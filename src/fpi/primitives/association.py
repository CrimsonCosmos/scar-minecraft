"""Association — weighted, temporal links between patterns.

An Association records that when pattern A occurs, pattern B tends to follow.
This is the substrate of prediction: dense, accurate associations enable
the system to anticipate the future from the present.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Association:
    """A directed, weighted link between two patterns.

    In Phase 3, associations have a last_activated tick and passively decay.
    Use it or lose it — connections that aren't reinforced wither and die.

    Attributes:
        source_id: Pattern ID of the antecedent.
        target_id: Pattern ID of the consequent.
        strength: How reliably source predicts target (0.0 to 1.0).
        temporal_delay: Typical number of timesteps between source and target.
        activation_count: How many times this association has been reinforced.
        last_activated: Tick when this association was last reinforced.
    """

    source_id: int
    target_id: int
    strength: float = 0.0
    temporal_delay: int = 1
    activation_count: int = 0
    last_activated: int = 0

    def reinforce(self, amount: float = 0.1, tick: int = 0) -> None:
        """Strengthen this association (Hebbian-style).

        Strength is clamped to [0, 1].
        """
        self.activation_count += 1
        self.strength = min(1.0, self.strength + amount * (1.0 - self.strength))
        self.last_activated = tick

    def weaken(self, amount: float = 0.05) -> None:
        """Weaken this association (prediction failed).

        Strength is clamped to [0, 1].
        """
        self.strength = max(0.0, self.strength - amount * self.strength)

    def decay(self, rate: float = 0.01) -> None:
        """Passive strength erosion — use it or lose it.

        This is separate from weaken() (which is active punishment for wrong
        predictions). Decay is entropy: connections that aren't maintained
        deteriorate.
        """
        self.strength = max(0.0, self.strength - rate)

    def fitness(self, current_tick: int) -> float:
        """Fitness score for eviction decisions.

        Combines strength with recency: strong-but-stale associations
        score lower than weak-but-active ones. Mirrors pattern eviction
        which uses exposure_count × recency.
        """
        age = max(0, current_tick - self.last_activated)
        recency = 1.0 / (1.0 + age * 0.01)
        return self.strength * recency

    @property
    def key(self) -> tuple[int, int]:
        """Unique identifier for this association."""
        return (self.source_id, self.target_id)


@dataclass(slots=True)
class AssociationMap:
    """A collection of associations, indexed for fast lookup.

    Provides O(1) lookup by (source, target) and O(1) lookup of all
    associations from a given source.

    In Phase 3, supports capacity limits, passive decay, and cascade removal
    when patterns are evicted. The associative network is an ecosystem —
    connections compete for limited resources.

    Attributes:
        max_associations: Capacity limit (None = unlimited).
        decay_rate: Passive strength erosion per tick (0.0 = no decay).
    """

    _by_key: dict[tuple[int, int], Association] = field(default_factory=dict)
    _by_source: dict[int, list[Association]] = field(default_factory=dict)
    max_associations: int | None = None
    decay_rate: float = 0.0
    _current_tick: int = 0

    def get_or_create(self, source_id: int, target_id: int) -> Association:
        """Get an existing association or create a new one."""
        key = (source_id, target_id)
        if key not in self._by_key:
            # Evict weakest if at capacity
            if self.max_associations is not None and len(self._by_key) >= self.max_associations:
                self._evict_weakest()
            assoc = Association(source_id=source_id, target_id=target_id)
            self._by_key[key] = assoc
            self._by_source.setdefault(source_id, []).append(assoc)
        return self._by_key[key]

    def get(self, source_id: int, target_id: int) -> Association | None:
        """Get an association if it exists."""
        return self._by_key.get((source_id, target_id))

    def from_source(self, source_id: int) -> list[Association]:
        """Get all associations originating from a pattern."""
        return self._by_source.get(source_id, [])

    def strongest_from(self, source_id: int) -> Association | None:
        """Get the strongest association from a source pattern."""
        assocs = self.from_source(source_id)
        if not assocs:
            return None
        return max(assocs, key=lambda a: a.strength)

    def remove(self, assoc: Association) -> None:
        """Remove a single association from both indices."""
        self._by_key.pop(assoc.key, None)
        source_list = self._by_source.get(assoc.source_id)
        if source_list is not None:
            try:
                source_list.remove(assoc)
            except ValueError:
                pass
            if not source_list:
                del self._by_source[assoc.source_id]

    def remove_associations_for_pattern(self, pattern_id: int) -> None:
        """Remove all associations involving a pattern (cascade cleanup on eviction)."""
        # Remove associations FROM this pattern
        to_remove = list(self._by_source.get(pattern_id, []))
        for assoc in to_remove:
            self.remove(assoc)

        # Remove associations TO this pattern
        to_remove = [a for a in self._by_key.values() if a.target_id == pattern_id]
        for assoc in to_remove:
            self.remove(assoc)

    def decay_all(self) -> list[Association]:
        """Apply passive decay to all associations. Prune dead ones.

        Returns the list of pruned (dead) associations.
        """
        self._current_tick += 1
        if self.decay_rate <= 0.0:
            return []

        pruned: list[Association] = []
        for assoc in list(self._by_key.values()):
            assoc.decay(self.decay_rate)
            if assoc.strength <= 0.0:
                pruned.append(assoc)
                self.remove(assoc)
        return pruned

    def _evict_weakest(self) -> None:
        """Evict the least-fit association to make room.

        Uses fitness (strength × recency) rather than strength alone,
        so stale associations are evicted before active ones.
        """
        if not self._by_key:
            return
        weakest = min(
            self._by_key.values(),
            key=lambda a: a.fitness(self._current_tick),
        )
        self.remove(weakest)

    @property
    def count(self) -> int:
        return len(self._by_key)
