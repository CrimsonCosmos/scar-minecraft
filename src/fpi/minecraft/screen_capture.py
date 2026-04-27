"""Screen capture process — grabs game frames via shared memory.

Runs a separate process that captures the screen at ~4 FPS, resizes to
84x84 grayscale, and writes the result to shared memory for the main
FPI loop to consume without blocking.

Shared memory layout (7064 bytes):
  Bytes 0-7:     frame counter (uint64 LE) — incremented each capture
  Bytes 8-7063:  84 * 84 = 7056 uint8 grayscale pixels (row-major)

Usage:
    capture = ScreenCaptureProcess(region=(x, y, w, h))
    capture.start()
    frame, frame_num = capture.read_frame()  # numpy (84, 84) uint8
    capture.stop()
"""

from __future__ import annotations

import struct
import time
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.typing import NDArray

FRAME_SIZE = 84
PIXEL_COUNT = FRAME_SIZE * FRAME_SIZE  # 7056
HEADER_SIZE = 8  # uint64 frame counter
SHM_SIZE = HEADER_SIZE + PIXEL_COUNT  # 7064
SHM_NAME_PREFIX = "fpi_vision_"


def _capture_loop(
    shm_name: str,
    region: tuple[int, int, int, int] | None,
    fps: int,
) -> None:
    """Main loop for the capture process. Runs until terminated."""
    try:
        import mss
        from PIL import Image
    except ImportError as exc:
        print(f"[screen_capture] Missing dependency: {exc}")
        print("[screen_capture] Install with: pip install mss Pillow")
        return

    shm = SharedMemory(name=shm_name, create=False)
    interval = 1.0 / fps
    frame_num = 0

    try:
        with mss.mss() as sct:
            # Determine capture region
            if region is not None:
                x, y, w, h = region
                monitor = {"left": x, "top": y, "width": w, "height": h}
            else:
                # Full primary monitor
                monitor = sct.monitors[1]

            while True:
                t0 = time.monotonic()

                try:
                    # Grab screen
                    shot = sct.grab(monitor)

                    # Convert to PIL, resize to 84x84, convert to grayscale
                    img = Image.frombytes("RGB", shot.size, shot.rgb)
                    img = img.resize((FRAME_SIZE, FRAME_SIZE), Image.BILINEAR)
                    img = img.convert("L")  # grayscale

                    pixels = np.frombuffer(img.tobytes(), dtype=np.uint8)
                except Exception:
                    # Capture failed — write zeros
                    pixels = np.zeros(PIXEL_COUNT, dtype=np.uint8)

                # Write frame counter + pixels to shared memory
                frame_num += 1
                struct.pack_into("<Q", shm.buf, 0, frame_num)
                shm.buf[HEADER_SIZE : HEADER_SIZE + PIXEL_COUNT] = pixels.tobytes()

                # Sleep to maintain target FPS
                elapsed = time.monotonic() - t0
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        shm.close()


class ScreenCaptureProcess:
    """Manages a background process that captures game frames.

    Frames are delivered through shared memory for zero-copy reads.

    Args:
        region: Capture region as (x, y, width, height) in screen coords.
            If None, captures the full primary monitor.
        fps: Target frames per second (default 4).
    """

    def __init__(
        self,
        region: tuple[int, int, int, int] | None = None,
        fps: int = 4,
    ) -> None:
        self._region = region
        self._fps = fps
        self._process: Process | None = None
        self._shm: SharedMemory | None = None
        self._last_frame_num: int = 0

    def start(self) -> None:
        """Start the capture process."""
        # Create shared memory
        self._shm = SharedMemory(
            name=SHM_NAME_PREFIX + str(int(time.time() * 1000) % 100000),
            create=True,
            size=SHM_SIZE,
        )
        # Zero out initial buffer
        self._shm.buf[:SHM_SIZE] = b"\x00" * SHM_SIZE

        # Start capture process
        self._process = Process(
            target=_capture_loop,
            args=(self._shm.name, self._region, self._fps),
            daemon=True,
        )
        self._process.start()

    def read_frame(self) -> tuple[NDArray[np.uint8], int]:
        """Read the latest frame from shared memory.

        Returns:
            (frame, frame_number) where frame is (84, 84) uint8 ndarray
            and frame_number is the capture sequence number (0 = no frame yet).
        """
        if self._shm is None:
            return np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8), 0

        frame_num = struct.unpack_from("<Q", self._shm.buf, 0)[0]
        pixels = bytes(self._shm.buf[HEADER_SIZE : HEADER_SIZE + PIXEL_COUNT])
        frame = np.frombuffer(pixels, dtype=np.uint8).reshape(FRAME_SIZE, FRAME_SIZE)
        self._last_frame_num = frame_num
        return frame, frame_num

    @property
    def has_new_frame(self) -> bool:
        """Check if a new frame is available since last read."""
        if self._shm is None:
            return False
        frame_num = struct.unpack_from("<Q", self._shm.buf, 0)[0]
        return frame_num > self._last_frame_num

    @property
    def is_alive(self) -> bool:
        """Check if the capture process is still running."""
        return self._process is not None and self._process.is_alive()

    def stop(self) -> None:
        """Stop the capture process and clean up shared memory."""
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)
        self._process = None

        if self._shm is not None:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            self._shm = None
