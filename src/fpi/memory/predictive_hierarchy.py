"""Predictive Hierarchy — true hierarchical predictive coding.

Implements Rao & Ballard (1999) and Friston's hierarchical predictive coding:

Level 0: processes raw signals (standard Distinction)
Level 1: processes prediction ERRORS from Level 0
Level 2: processes prediction ERRORS from Level 1
...

Each higher level:
1. Receives ERRORS from below (not raw patterns)
2. Generates top-down predictions that BIAS the level below
3. Operates at a SLOWER timescale (tick_divisor doubles per level)

This is NOT the same as TemporalHierarchy (which runs the same data at
different timescales). Here, each level processes fundamentally different
data: the RESIDUALS from the level below.

Top-down modulation: Level N's prediction modifies Level N-1's similarity
threshold — predicted patterns are easier to match, implementing perceptual
priors ("you see what you expect").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..primitives.signal import Signal
from ..primitives.pattern import Pattern, Distinction
from ..primitives.association import AssociationMap


@dataclass(slots=True)
class LevelState:
    """State of one level in the predictive coding hierarchy."""

    level: int
    distinction: Distinction
    associations: AssociationMap
    current_pattern: Pattern | None = None
    prev_pattern: Pattern | None = None
    last_prediction: tuple[Pattern, float] | None = None
    last_surprise: float = 1.0
    tick_divisor: int = 1  # Level N updates every tick_divisor ticks


class PredictiveHierarchy:
    """True hierarchical predictive coding.

    Level 0 processes raw signals. Each subsequent level processes the
    prediction errors from the level below. Higher levels operate at
    slower timescales and generate top-down predictions that bias lower
    levels (making expected patterns easier to match).

    Args:
        num_levels: Depth of hierarchy (2-4 recommended).
        base_threshold: Level 0 similarity threshold.
        base_max_patterns: Max patterns per level.
        top_down_strength: How much higher levels bias lower ones.
        association_decay_rate: Passive decay for associations.
    """

    def __init__(
        self,
        num_levels: int = 3,
        base_threshold: float = 0.7,
        base_max_patterns: int = 20,
        top_down_strength: float = 0.1,
        association_decay_rate: float = 0.01,
    ) -> None:
        self._num_levels = num_levels
        self._top_down_strength = top_down_strength
        self._global_tick = 0

        self._levels: list[LevelState] = []
        for i in range(num_levels):
            tick_div = 2 ** i  # Level 0: every tick, Level 1: every 2, Level 2: every 4
            level = LevelState(
                level=i,
                distinction=Distinction(
                    similarity_threshold=base_threshold,
                    max_patterns=base_max_patterns,
                ),
                associations=AssociationMap(
                    decay_rate=association_decay_rate,
                ),
                tick_divisor=tick_div,
            )
            self._levels.append(level)

        # Store base threshold so we can restore after top-down bias
        self._base_thresholds = [base_threshold] * num_levels

    @property
    def num_levels(self) -> int:
        return self._num_levels

    @property
    def levels(self) -> list[LevelState]:
        return self._levels

    def observe(self, signal: Signal, tick: int) -> dict[int, float]:
        """Process signal through the hierarchy. Returns {level: surprise}.

        Bottom-up pass:
            Level 0: distinguish(signal) -> pattern, compute error
            Level 1: distinguish(error_signal_from_0) -> error_pattern
            Level 2: distinguish(error_signal_from_1) -> error_error_pattern

        Top-down pass:
            Each level's prediction biases the level below.
        """
        self._global_tick = tick
        surprises: dict[int, float] = {}
        current_signal = signal

        # Bottom-up pass: each level processes the error from the level below
        for i, level in enumerate(self._levels):
            # Check if this level should update at this tick
            if tick % level.tick_divisor != 0:
                surprises[i] = level.last_surprise
                continue

            # Apply top-down bias from the level above (if any)
            if i > 0:
                bias = self._top_down_bias(i)
                level.distinction.similarity_threshold = max(
                    0.3, self._base_thresholds[i] - bias,
                )
            else:
                # Level 0 also gets top-down bias if there are higher levels
                if self._num_levels > 1:
                    bias = self._top_down_bias(0)
                    level.distinction.similarity_threshold = max(
                        0.3, self._base_thresholds[0] - bias,
                    )

            # Distinguish the current signal at this level
            level.prev_pattern = level.current_pattern
            pattern, sim = level.distinction.distinguish(current_signal)
            level.current_pattern = pattern

            # Compute surprise against prediction
            surprise = self._compute_level_surprise(level, pattern)
            level.last_surprise = surprise
            surprises[i] = surprise

            # Update associations
            if level.prev_pattern is not None:
                if surprise < 0.5:
                    assoc = level.associations.get_or_create(
                        level.prev_pattern.pattern_id,
                        pattern.pattern_id,
                    )
                    assoc.reinforce(tick=tick)
                else:
                    if level.last_prediction is not None:
                        old_assoc = level.associations.get(
                            level.prev_pattern.pattern_id,
                            level.last_prediction[0].pattern_id,
                        )
                        if old_assoc is not None:
                            old_assoc.weaken()
                    assoc = level.associations.get_or_create(
                        level.prev_pattern.pattern_id,
                        pattern.pattern_id,
                    )
                    assoc.reinforce(tick=tick)

            # Generate prediction for next timestep at this level
            strongest = level.associations.strongest_from(pattern.pattern_id)
            if strongest is not None:
                for p in level.distinction.patterns:
                    if p.pattern_id == strongest.target_id:
                        level.last_prediction = (p, strongest.strength)
                        break
                else:
                    level.last_prediction = None
            else:
                level.last_prediction = None

            # Compute error signal for the next level up
            if i < self._num_levels - 1:
                current_signal = self._compute_error_signal(level, pattern, current_signal)

        return surprises

    def _compute_error_signal(
        self, level: LevelState, actual_pattern: Pattern, raw_signal: Signal,
    ) -> Signal:
        """Compute the prediction error signal to pass to the next level.

        Error = |predicted_centroid - actual_centroid|. If no prediction,
        the error IS the raw signal (everything is surprising).
        """
        if level.last_prediction is not None:
            predicted_centroid = level.last_prediction[0].centroid
            actual_centroid = actual_pattern.centroid
            # Ensure same dimensionality
            if len(predicted_centroid) == len(actual_centroid):
                error_data = np.abs(predicted_centroid - actual_centroid)
            else:
                error_data = raw_signal.data.copy()
        else:
            # No prediction → full surprise → pass raw signal as error
            error_data = raw_signal.data.copy()

        return Signal(data=error_data, timestamp=raw_signal.timestamp)

    def _compute_level_surprise(
        self, level: LevelState, actual_pattern: Pattern,
    ) -> float:
        """Compute surprise at a given level."""
        if level.last_prediction is None:
            return 1.0

        predicted_pattern, confidence = level.last_prediction
        if predicted_pattern.pattern_id == actual_pattern.pattern_id:
            return 0.0

        norm_p = np.linalg.norm(predicted_pattern.centroid)
        norm_a = np.linalg.norm(actual_pattern.centroid)
        if norm_p == 0.0 or norm_a == 0.0:
            return 1.0
        sim = float(
            np.dot(predicted_pattern.centroid, actual_pattern.centroid)
            / (norm_p * norm_a)
        )
        return 1.0 - max(0.0, sim)

    def _top_down_bias(self, target_level: int) -> float:
        """Compute threshold reduction for target_level based on predictions
        from the level above.

        When the level above has a confident prediction, the target level's
        threshold is reduced (predicted patterns are easier to match).
        This implements "you see what you expect to see."
        """
        source_level_idx = target_level + 1
        if source_level_idx >= self._num_levels:
            return 0.0

        source_level = self._levels[source_level_idx]
        if source_level.last_prediction is None:
            return 0.0

        _, confidence = source_level.last_prediction
        return self._top_down_strength * confidence

    def get_surprise(self) -> dict[int, float]:
        """Per-level surprise values from the last observation."""
        return {level.level: level.last_surprise for level in self._levels}

    def aggregate_surprise(self) -> float:
        """Weighted average surprise across levels.

        Higher levels contribute less (they update less frequently and
        represent more abstract patterns).
        """
        total = 0.0
        weight_sum = 0.0
        for level in self._levels:
            weight = 1.0 / level.tick_divisor
            total += level.last_surprise * weight
            weight_sum += weight
        return total / weight_sum if weight_sum > 0 else 1.0

    def tick(self) -> None:
        """Advance all levels. Higher levels tick less frequently."""
        self._global_tick += 1
        for level in self._levels:
            if self._global_tick % level.tick_divisor == 0:
                level.distinction.advance_tick()
                level.associations.decay_all()
