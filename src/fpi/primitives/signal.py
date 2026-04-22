"""Signal — the atomic unit of information.

A Signal is a typed, timestamped value. It's the fundamental data element that
flows through the entire system. Signals are domain-agnostic: a pixel value,
a phoneme, a touch pressure reading — all are signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Signal:
    """An atomic unit of information — a timestamped vector with a modality label.

    Attributes:
        data: The signal content as a 1-D float array.
        timestamp: When this signal was produced (discrete tick).
        modality: Which channel produced this signal (e.g. "visual", "motor").
    """

    data: NDArray[np.float64]
    timestamp: int = 0
    modality: str = "default"

    @staticmethod
    def from_scalar(value: float, timestamp: int = 0, modality: str = "default") -> Signal:
        """Create a 1-element signal from a scalar."""
        return Signal(data=np.array([value], dtype=np.float64), timestamp=timestamp, modality=modality)

    @staticmethod
    def from_list(values: list[float], timestamp: int = 0, modality: str = "default") -> Signal:
        """Create a signal from a list of floats."""
        return Signal(data=np.array(values, dtype=np.float64), timestamp=timestamp, modality=modality)

    @property
    def dim(self) -> int:
        """Dimensionality of the signal."""
        return len(self.data)

    def distance(self, other: Self) -> float:
        """Euclidean distance to another signal."""
        return float(np.linalg.norm(self.data - other.data))

    def cosine_similarity(self, other: Self) -> float:
        """Cosine similarity to another signal. Returns 0 if either is zero."""
        norm_a = np.linalg.norm(self.data)
        norm_b = np.linalg.norm(other.data)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(self.data, other.data) / (norm_a * norm_b))


@dataclass(frozen=True, slots=True)
class SignalBundle:
    """A collection of simultaneous signals — a single 'moment' of experience.

    A bundle captures everything the agent senses (or generates internally)
    at a single timestep.
    """

    signals: tuple[Signal, ...] = field(default_factory=tuple)

    @staticmethod
    def from_signals(*signals: Signal) -> SignalBundle:
        return SignalBundle(signals=signals)

    @property
    def timestamp(self) -> int:
        """Timestamp of the bundle (from the first signal, or 0)."""
        return self.signals[0].timestamp if self.signals else 0

    def combined_vector(self) -> NDArray[np.float64]:
        """Concatenate all signal data into a single vector."""
        if not self.signals:
            return np.array([], dtype=np.float64)
        return np.concatenate([s.data for s in self.signals])

    def by_modality(self, modality: str) -> tuple[Signal, ...]:
        """Get all signals of a given modality."""
        return tuple(s for s in self.signals if s.modality == modality)
