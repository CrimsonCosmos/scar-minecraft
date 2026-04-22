"""Data sources — abstract streams that yield raw observations.

Each stream produces string values at its own cadence. The monitor
polls all streams in a loop; streams return [] when they have no new data.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from typing import IO


class Stream(ABC):
    """Abstract data source that yields string values."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this stream."""
        ...

    @abstractmethod
    def poll(self) -> list[str]:
        """Return new data since last poll. Non-blocking. Returns [] if none."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        pass


class CommandStream(Stream):
    """Run a shell command periodically and yield its output.

    Args:
        command: Shell command to execute.
        interval_seconds: Seconds between executions.
    """

    def __init__(self, command: str, interval_seconds: float = 60.0) -> None:
        self._command = command
        self._interval = interval_seconds
        self._last_run: float = 0.0

    @property
    def name(self) -> str:
        return f"cmd:{self._command[:40]}"

    def poll(self) -> list[str]:
        """Run command if interval has elapsed. Return output lines."""
        now = time.monotonic()
        if now - self._last_run < self._interval:
            return []
        self._last_run = now

        try:
            result = subprocess.run(
                self._command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout.strip()
            if output:
                return [output]
        except (subprocess.TimeoutExpired, OSError):
            pass
        return []


class FileStream(Stream):
    """Tail a file and yield new lines (like tail -f).

    Opens the file and seeks to the end. Only reads content written
    after the stream was created. Handles file rotation by re-opening.

    Args:
        path: File path to watch.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._file: IO[str] | None = None
        self._open()

    @property
    def name(self) -> str:
        return f"file:{os.path.basename(self._path)}"

    def _open(self) -> None:
        """Open file and seek to end."""
        try:
            self._file = open(self._path, "r")
            self._file.seek(0, 2)  # Seek to end
        except OSError:
            self._file = None

    def poll(self) -> list[str]:
        """Read new lines since last poll."""
        if self._file is None:
            self._open()
            if self._file is None:
                return []

        try:
            lines = self._file.readlines()
            return [line.strip() for line in lines if line.strip()]
        except OSError:
            self._file = None
            return []

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class StdinStream(Stream):
    """Read from stdin line by line (non-blocking).

    Uses select() to check if stdin has data without blocking.
    Works on Unix/macOS for piped input.
    """

    @property
    def name(self) -> str:
        return "stdin"

    def poll(self) -> list[str]:
        """Read available lines from stdin without blocking."""
        import select

        lines: list[str] = []
        while select.select([sys.stdin], [], [], 0.0)[0]:
            line = sys.stdin.readline()
            if not line:  # EOF
                break
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
        return lines
