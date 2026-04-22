"""Associative Memory — storing patterns and recalling them via similarity.

The associative memory is where patterns and their associations live. It
provides the operations that the world model and agent need: store a new
pattern, recall similar patterns given a query, and form/strengthen
associations between patterns.

In Phase 3, the memory has finite capacity and maintenance costs. Patterns
and associations compete for resources — the brain is an ecosystem, not a
warehouse. Maintenance cost is returned to the agent as a vitality drain:
thinking itself costs energy.
"""

from __future__ import annotations

import numpy as np

from ..primitives.signal import Signal
from ..primitives.pattern import Pattern, Distinction
from ..primitives.association import Association, AssociationMap


class AssociativeMemory:
    """Stores patterns and associations, supports recall by similarity.

    This is the system's long-term memory. It wraps a Distinction (for
    pattern recognition) and an AssociationMap (for pattern-to-pattern links).

    Phase 3 additions:
    - Capacity limits on patterns and associations.
    - Passive decay on associations (use it or lose it).
    - Maintenance cost: each pattern and association costs energy to maintain.
    - Eviction cascade: when a pattern is evicted, its associations are cleaned up.

    Attributes:
        distinction: The pattern-matching machinery.
        associations: All learned associations between patterns.
        maintenance_cost_per_pattern: Energy cost per pattern per tick.
        maintenance_cost_per_association: Energy cost per association per tick.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        max_patterns: int | None = None,
        max_associations: int | None = None,
        association_decay_rate: float = 0.0,
        maintenance_cost_per_pattern: float = 0.0,
        maintenance_cost_per_association: float = 0.0,
        enable_salience: bool = False,
        modality_slices: list[tuple[int, int]] | None = None,
        enable_compositional: bool = False,
        patterns_per_modality: int = 15,
        modality_thresholds: list[float] | None = None,
    ) -> None:
        if enable_compositional and modality_slices:
            from ..primitives.compositional import CompositionalDistinction
            self.distinction = CompositionalDistinction(
                modality_slices=modality_slices,
                similarity_threshold=similarity_threshold,
                patterns_per_modality=patterns_per_modality,
                enable_salience=enable_salience,
                modality_thresholds=modality_thresholds,
            )
        else:
            self.distinction = Distinction(
                similarity_threshold=similarity_threshold,
                max_patterns=max_patterns,
                enable_salience=enable_salience,
                modality_slices=modality_slices,
            )
        self.associations = AssociationMap(
            max_associations=max_associations,
            decay_rate=association_decay_rate,
        )
        self._recent_pattern_ids: list[int] = []
        self.maintenance_cost_per_pattern = maintenance_cost_per_pattern
        self.maintenance_cost_per_association = maintenance_cost_per_association

    def store(self, signal: Signal) -> Pattern:
        """Process a signal and return its matched/created pattern.

        Also tracks the recent pattern sequence for building associations.
        If patterns were evicted (capacity limit), cascade-cleans their associations.
        """
        pattern, _ = self.distinction.distinguish(signal)

        # Cascade cleanup: remove associations for any evicted patterns
        for evicted_id in self.distinction.drain_evicted():
            self.associations.remove_associations_for_pattern(evicted_id)

        self._recent_pattern_ids.append(pattern.pattern_id)
        return pattern

    def recall(self, query: Signal, top_k: int = 5) -> list[tuple[Pattern, float]]:
        """Find the most similar patterns to a query signal.

        Returns a list of (pattern, similarity) sorted by descending similarity.
        """
        if not self.distinction.patterns:
            return []

        scored = [
            (p, self.distinction._similarity(p, query))
            for p in self.distinction.patterns
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def update_salience(self, error: float) -> None:
        """Update salience weights based on outcome error."""
        self.distinction.update_salience(error)

    def associate(self, source_id: int, target_id: int, reinforce: bool = True) -> Association:
        """Form or strengthen an association between two patterns.

        If reinforce is True, an existing association is strengthened.
        """
        assoc = self.associations.get_or_create(source_id, target_id)
        if reinforce:
            assoc.reinforce(tick=self.distinction._current_tick)
        return assoc

    def weaken(self, source_id: int, target_id: int) -> None:
        """Weaken an association (wrong prediction)."""
        assoc = self.associations.get(source_id, target_id)
        if assoc is not None:
            assoc.weaken()

    def predict_next(self, current_pattern_id: int) -> tuple[Pattern, float] | None:
        """Predict the most likely next pattern given the current one.

        Returns (predicted_pattern, confidence) or None if no associations exist.
        """
        strongest = self.associations.strongest_from(current_pattern_id)
        if strongest is None:
            return None

        # Find the target pattern by ID
        for p in self.distinction.patterns:
            if p.pattern_id == strongest.target_id:
                return p, strongest.strength
        return None

    def tick(self) -> float:
        """Advance one tick: decay associations, compute maintenance cost.

        Returns total maintenance cost (energy drain for having a brain).
        """
        self.distinction.advance_tick()

        # Passive decay — unused associations wither
        self.associations.decay_all()

        # Maintenance cost: thinking costs energy
        cost = (
            self.pattern_count * self.maintenance_cost_per_pattern
            + self.association_count * self.maintenance_cost_per_association
        )
        return cost

    def build_associations_from_sequence(self) -> None:
        """Form associations from the recent pattern sequence.

        For each consecutive pair (A, B) in the sequence, reinforce A → B.
        """
        for i in range(len(self._recent_pattern_ids) - 1):
            src = self._recent_pattern_ids[i]
            tgt = self._recent_pattern_ids[i + 1]
            self.associate(src, tgt, reinforce=True)

    def clear_recent(self) -> None:
        """Clear the recent pattern sequence (e.g. between episodes)."""
        self._recent_pattern_ids.clear()

    @property
    def pattern_count(self) -> int:
        return len(self.distinction.patterns)

    @property
    def association_count(self) -> int:
        return self.associations.count
