"""Signal encoders for real-world data.

Converts raw data (numbers, text) into Signals with meaningful cosine
similarity. This is the critical bridge between the real world and the
intelligence's pattern recognition.

Key insight: raw scalars like [50.0] and [95.0] have cosine similarity ~1.0
(they are colinear). Gaussian basis encoding makes nearby values similar and
distant values different — the same trick used for agent position encoding
in SurvivalEnv.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..primitives.signal import Signal


@dataclass
class NumericEncoder:
    """Encode numeric values using Gaussian basis functions.

    Same principle as SurvivalEnv position encoding: nearby values
    produce similar signals, distant values produce different signals.

    Attributes:
        num_bases: Number of Gaussian basis functions.
        range_min: Expected minimum value (auto-expands if exceeded).
        range_max: Expected maximum value (auto-expands if exceeded).
        modality: Signal modality label.
    """

    num_bases: int = 64
    range_min: float = 0.0
    range_max: float = 100.0
    modality: str = "numeric"

    def encode(self, value: float, timestamp: int = 0) -> Signal:
        """Encode a scalar value as a Gaussian-basis Signal."""
        # Auto-expand range
        if value < self.range_min:
            self.range_min = value - max(1.0, abs(value) * 0.1)
        if value > self.range_max:
            self.range_max = value + max(1.0, abs(value) * 0.1)

        span = self.range_max - self.range_min
        centers = np.linspace(self.range_min, self.range_max, self.num_bases)
        # With many bases over a wide range, sigma must be wide enough that
        # nearby values activate overlapping sets of bases (high cosine sim)
        # while distant values activate disjoint bases (low cosine sim).
        sigma = max(0.01, 3.0 * span / self.num_bases)
        basis = np.exp(-((value - centers) ** 2) / (2 * sigma ** 2))

        return Signal(
            data=basis.astype(np.float64),
            timestamp=timestamp,
            modality=self.modality,
        )


@dataclass
class TextEncoder:
    """Encode text lines as fixed-size vectors via character n-gram hashing.

    Similar log lines produce similar signals. Different log lines produce
    different signals. Uses the hashing trick for fixed dimensionality.

    Attributes:
        dim: Dimensionality of the output signal.
        ngram_sizes: Character n-gram sizes to extract.
        modality: Signal modality label.
    """

    dim: int = 64
    ngram_sizes: tuple[int, ...] = (2, 3, 4)
    modality: str = "text"

    def encode(self, text: str, timestamp: int = 0) -> Signal:
        """Encode a text line as a Signal."""
        data = np.zeros(self.dim, dtype=np.float64)

        for n in self.ngram_sizes:
            for i in range(len(text) - n + 1):
                ngram = text[i : i + n]
                h = hash(ngram) % self.dim
                data[h] += 1.0

        # L2 normalize so cosine similarity is well-defined
        norm = np.linalg.norm(data)
        if norm > 0:
            data /= norm

        return Signal(
            data=data,
            timestamp=timestamp,
            modality=self.modality,
        )


class AutoEncoder:
    """Automatically selects NumericEncoder or TextEncoder based on input.

    If input parses as a float, use NumericEncoder. Otherwise, TextEncoder.
    Handles common formats like "45%" by stripping trailing percent signs.
    """

    def __init__(
        self,
        numeric_kwargs: dict | None = None,
        text_kwargs: dict | None = None,
    ) -> None:
        self._numeric = NumericEncoder(**(numeric_kwargs or {}))
        self._text = TextEncoder(**(text_kwargs or {}))

    def encode(self, raw: str, timestamp: int = 0) -> Signal:
        """Encode raw string input, auto-detecting type."""
        cleaned = raw.strip().rstrip("%")
        try:
            value = float(cleaned)
            return self._numeric.encode(value, timestamp=timestamp)
        except ValueError:
            return self._text.encode(raw, timestamp=timestamp)
