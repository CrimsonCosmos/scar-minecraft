"""Affect — 2D emotional state (valence x arousal).

Emotions are NOT hardcoded categories. They are PATTERNS in affect space,
discovered by the same Distinction machinery used for everything else.
This implements Barrett's theory of constructed emotion: the brain
categorizes interoceptive signals (valence, arousal) into emotion
concepts using the same pattern-matching used for perception.

Russell's circumplex model provides the 2D space:
- Valence: [-1, 1] — good/bad (from Valence primitive)
- Arousal: [0, 1] — activation level (from surprise × urgency)

"Fear" = (negative valence, high arousal) — not a label, a region in
affect space that the agent discovers through experience.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .pattern import Distinction, Pattern
from .signal import Signal


@dataclass(slots=True)
class AffectState:
    """2D affect: valence + arousal."""

    valence: float = 0.0  # [-1, 1] good/bad
    arousal: float = 0.0  # [0, 1] activation level

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.valence, self.arousal], dtype=np.float64)


class AffectStream:
    """Encodes (valence, arousal) as Signal, discovers emotion patterns.

    Uses Gaussian basis encoding to create a signal from the 2D affect
    state, then feeds it to a Distinction to discover emotion patterns.
    Each emotion is just a region in affect space that the agent visits
    repeatedly.

    Args:
        bases_per_dim: Gaussian bases per affect dimension.
        similarity_threshold: For the internal Distinction.
        max_patterns: Capacity limit for emotion patterns.
    """

    def __init__(
        self,
        bases_per_dim: int = 6,
        similarity_threshold: float = 0.7,
        max_patterns: int = 10,
    ) -> None:
        self._bases_per_dim = bases_per_dim
        # Valence centers: [-1, 1]
        self._valence_centers = np.linspace(-1.0, 1.0, bases_per_dim)
        self._valence_sigma = max(0.01, 2.0 / bases_per_dim)
        # Arousal centers: [0, 1]
        self._arousal_centers = np.linspace(0.0, 1.0, bases_per_dim)
        self._arousal_sigma = max(0.01, 1.0 / bases_per_dim)

        self._distinction = Distinction(
            similarity_threshold=similarity_threshold,
            max_patterns=max_patterns,
        )
        self._current_state: AffectState = AffectState()
        self._current_pattern: Pattern | None = None

    @property
    def signal_dim(self) -> int:
        """Total signal dimensionality: 2 dims * bases_per_dim."""
        return 2 * self._bases_per_dim

    @property
    def current_state(self) -> AffectState:
        return self._current_state

    @property
    def current_pattern(self) -> Pattern | None:
        return self._current_pattern

    @property
    def pattern_count(self) -> int:
        return len(self._distinction.patterns)

    def encode(self, state: AffectState, timestamp: int) -> Signal:
        """Gaussian-basis encode valence and arousal."""
        val_basis = np.exp(
            -((np.clip(state.valence, -1.0, 1.0) - self._valence_centers) ** 2)
            / (2 * self._valence_sigma**2)
        )
        aro_basis = np.exp(
            -((np.clip(state.arousal, 0.0, 1.0) - self._arousal_centers) ** 2)
            / (2 * self._arousal_sigma**2)
        )
        data = np.concatenate([val_basis, aro_basis]).astype(np.float64)
        return Signal(data=data, timestamp=timestamp, modality="affect")

    def observe(
        self, state: AffectState, timestamp: int,
    ) -> tuple[Pattern, float]:
        """Encode affect state, distinguish, return (emotion_pattern, surprise).

        The returned pattern IS the discovered emotion — a region in
        affect space the agent has visited before.
        """
        self._current_state = state
        signal = self.encode(state, timestamp)
        pattern, similarity = self._distinction.distinguish(signal)
        self._current_pattern = pattern
        # Surprise: 1 - similarity (novel emotion = high surprise)
        surprise = 1.0 - similarity if similarity < 1.0 else 0.0
        return pattern, surprise

    def tick(self) -> None:
        """Advance the internal Distinction clock."""
        self._distinction.advance_tick()
