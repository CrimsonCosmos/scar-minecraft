"""Valence — learned correlation between patterns and vitality change.

Valence is not hardcoded. A pattern has no intrinsic goodness or badness.
Valence is learned: when a pattern co-occurs with energy gain, it acquires
positive valence. When it co-occurs with energy loss, negative valence.

This is how the agent learns what matters — from experience, not design.
"Good" and "bad" are not assigned. They emerge from the correlation between
perception and survival.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Valence:
    """Tracks learned vitality-correlation for each pattern.

    Uses an exponential moving average so recent experience matters more
    than ancient history, allowing adaptation when the world changes.

    Attributes:
        _values: pattern_id → running average of vitality deltas.
        _counts: pattern_id → number of observations.
        learning_rate: How quickly valence updates (higher = more reactive).
    """

    _values: dict[int, float] = field(default_factory=dict)
    _counts: dict[int, int] = field(default_factory=dict)
    learning_rate: float = 0.3

    def update(self, pattern_id: int, vitality_delta: float) -> None:
        """Update valence for a pattern based on observed vitality change.

        Positive vitality_delta → pattern acquires positive valence.
        Negative vitality_delta → pattern acquires negative valence.
        """
        if pattern_id not in self._values:
            self._values[pattern_id] = vitality_delta
            self._counts[pattern_id] = 1
        else:
            alpha = self.learning_rate
            self._values[pattern_id] = (
                (1.0 - alpha) * self._values[pattern_id] + alpha * vitality_delta
            )
            self._counts[pattern_id] += 1

    def adjust_retroactive(
        self,
        pattern_ids: list[int],
        outcome_delta: float,
        decay: float = 0.85,
        strength: float = 0.1,
    ) -> None:
        """Adjust valence of patterns that preceded current outcome.

        When a bad outcome occurs (high surprise + vitality loss), the patterns
        that led to it should be penalized. When stability occurs (low surprise
        + vitality gain), predecessors get a boost.

        Args:
            pattern_ids: Recent pattern IDs, most recent last.
            outcome_delta: The vitality change that triggers adjustment.
                Negative = penalize predecessors, positive = boost them.
            decay: Geometric decay per step backward (0-1).
            strength: Overall scaling factor for adjustments.
        """
        if outcome_delta == 0.0:
            return
        for i, pid in enumerate(reversed(pattern_ids)):
            adjustment = outcome_delta * strength * (decay ** i)
            self.update(pid, adjustment)

    def get(self, pattern_id: int) -> float:
        """Get the valence of a pattern. Unknown patterns return 0 (neutral)."""
        return self._values.get(pattern_id, 0.0)

    def is_known(self, pattern_id: int) -> bool:
        """Whether this pattern has any valence history."""
        return pattern_id in self._values

    @property
    def known_count(self) -> int:
        """Number of patterns with learned valence."""
        return len(self._values)
