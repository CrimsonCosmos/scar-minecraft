"""Vision encoder — tiny CNN for 84x84 grayscale game frames.

Numpy-only inference. No torch/tensorflow required at runtime.
Weights can be initialized randomly (useful even untrained — random
CNN features still capture spatial structure that the history trace
can differentiate temporally) or loaded from a .npz file.

Architecture:
    Input:  84 x 84 x 1  (grayscale, float32 in [0, 1])
    Conv1:  32 filters, 8x8, stride 4, ReLU  → 20 x 20 x 32
    Conv2:  64 filters, 4x4, stride 2, ReLU  →  9 x  9 x 64
    Conv3: 128 filters, 3x3, stride 1, ReLU  →  7 x  7 x 128
    Global average pool                       → 128
    FC:    128 → 16                           →  16
    L2-normalize                              →  16

~55K parameters total.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

VISION_DIM = 16


def _conv2d(
    x: NDArray[np.float32],
    weights: NDArray[np.float32],
    bias: NDArray[np.float32],
    stride: int,
) -> NDArray[np.float32]:
    """2D convolution using numpy stride tricks.

    Args:
        x: Input (H, W, C_in).
        weights: Filters (C_out, kH, kW, C_in).
        bias: Bias (C_out,).
        stride: Stride in both dimensions.

    Returns:
        Output (H_out, W_out, C_out).
    """
    h, w, c_in = x.shape
    c_out, kh, kw, kc = weights.shape

    h_out = (h - kh) // stride + 1
    w_out = (w - kw) // stride + 1

    # Extract patches using stride tricks
    # Shape: (h_out, w_out, kh, kw, c_in)
    strides = x.strides
    patches = np.lib.stride_tricks.as_strided(
        x,
        shape=(h_out, w_out, kh, kw, c_in),
        strides=(
            strides[0] * stride,
            strides[1] * stride,
            strides[0],
            strides[1],
            strides[2],
        ),
    )

    # Reshape for matmul: (h_out * w_out, kh * kw * c_in) @ (kh * kw * c_in, c_out)
    patches_flat = patches.reshape(h_out * w_out, kh * kw * c_in)
    weights_flat = weights.reshape(c_out, kh * kw * kc).T  # (kh*kw*c_in, c_out)

    out = patches_flat @ weights_flat + bias  # (h_out * w_out, c_out)
    return out.reshape(h_out, w_out, c_out)


def _relu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """ReLU activation (in-place for efficiency)."""
    return np.maximum(x, 0, out=x)


class VisionEncoder:
    """Tiny CNN that encodes 84x84 grayscale frames into 16-dim vectors.

    Args:
        weights_path: Path to .npz file with pre-trained weights.
            If None, initializes with Kaiming-uniform random weights.
        seed: Random seed for weight initialization.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        seed: int = 42,
    ) -> None:
        if weights_path is not None:
            self.load_weights(weights_path)
        else:
            self._init_random_weights(seed)

    def _init_random_weights(self, seed: int) -> None:
        """Initialize weights with Kaiming-uniform distribution."""
        rng = np.random.default_rng(seed)

        def kaiming(shape: tuple[int, ...], fan_in: int) -> NDArray[np.float32]:
            std = np.sqrt(2.0 / fan_in)
            return (rng.standard_normal(shape) * std).astype(np.float32)

        # Conv1: 32 filters, 8x8, input channels 1
        self.conv1_w = kaiming((32, 8, 8, 1), fan_in=8 * 8 * 1)
        self.conv1_b = np.zeros(32, dtype=np.float32)

        # Conv2: 64 filters, 4x4, input channels 32
        self.conv2_w = kaiming((64, 4, 4, 32), fan_in=4 * 4 * 32)
        self.conv2_b = np.zeros(64, dtype=np.float32)

        # Conv3: 128 filters, 3x3, input channels 64
        self.conv3_w = kaiming((128, 3, 3, 64), fan_in=3 * 3 * 64)
        self.conv3_b = np.zeros(128, dtype=np.float32)

        # FC: 128 → 16
        self.fc_w = kaiming((128, VISION_DIM), fan_in=128)
        self.fc_b = np.zeros(VISION_DIM, dtype=np.float32)

    def encode(self, frame: NDArray[np.uint8]) -> NDArray[np.float64]:
        """Encode an 84x84 grayscale frame into a 16-dim L2-normalized vector.

        Args:
            frame: (84, 84) uint8 grayscale image.

        Returns:
            (16,) float64 L2-normalized feature vector.
        """
        # Normalize to [0, 1] float32, add channel dim
        x = frame.astype(np.float32) / 255.0
        if x.ndim == 2:
            x = x[:, :, np.newaxis]  # (84, 84, 1)

        # Conv1: 8x8, stride 4 → (20, 20, 32)
        x = _relu(_conv2d(x, self.conv1_w, self.conv1_b, stride=4))

        # Conv2: 4x4, stride 2 → (9, 9, 64)
        x = _relu(_conv2d(x, self.conv2_w, self.conv2_b, stride=2))

        # Conv3: 3x3, stride 1 → (7, 7, 128)
        x = _relu(_conv2d(x, self.conv3_w, self.conv3_b, stride=1))

        # Global average pool → (128,)
        x = x.mean(axis=(0, 1))

        # FC → (16,)
        features = (x @ self.fc_w + self.fc_b).astype(np.float64)

        # L2-normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm

        return features

    def save_weights(self, path: str) -> None:
        """Save weights to .npz file."""
        np.savez(
            path,
            conv1_w=self.conv1_w,
            conv1_b=self.conv1_b,
            conv2_w=self.conv2_w,
            conv2_b=self.conv2_b,
            conv3_w=self.conv3_w,
            conv3_b=self.conv3_b,
            fc_w=self.fc_w,
            fc_b=self.fc_b,
        )

    def load_weights(self, path: str) -> None:
        """Load weights from .npz file."""
        data = np.load(path)
        self.conv1_w = data["conv1_w"]
        self.conv1_b = data["conv1_b"]
        self.conv2_w = data["conv2_w"]
        self.conv2_b = data["conv2_b"]
        self.conv3_w = data["conv3_w"]
        self.conv3_b = data["conv3_b"]
        self.fc_w = data["fc_w"]
        self.fc_b = data["fc_b"]
