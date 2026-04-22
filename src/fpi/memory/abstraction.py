"""AbstractionLayer — patterns-of-patterns.

The same FPI loop applied recursively: instead of categorizing raw signals
into patterns, this layer categorizes *patterns* into meta-patterns.
This produces abstraction, generalization, and proto-symbolic rules
without adding any new primitives.

Layer 0 (WorldModel): Signal → Pattern
Layer 1 (AbstractionLayer): Pattern centroid → Meta-Pattern

Meta-patterns ARE abstract categories. Meta-associations ARE if-then rules.
Meta-valence IS generalized evaluation. All from the same six primitives.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..primitives.signal import Signal
from ..primitives.pattern import Pattern
from ..primitives.valence import Valence
from ..world_model.model import WorldModel


class AbstractionLayer:
    """Higher-order WorldModel that categorizes base-patterns into meta-patterns.

    Same mechanism as Layer 0, but input = pattern centroids (not raw signals).
    Produces meta-patterns (abstract categories), meta-associations (abstract
    rules), and meta-valence (category-level evaluation).

    Attributes:
        world_model: The meta-level WorldModel (patterns over patterns).
        valence: Meta-valence — which abstract categories are good/bad.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_meta_patterns: int = 20,
        max_meta_associations: int = 100,
        association_decay_rate: float = 0.001,
    ) -> None:
        self.world_model = WorldModel(
            similarity_threshold=similarity_threshold,
            max_patterns=max_meta_patterns,
            max_associations=max_meta_associations,
            association_decay_rate=association_decay_rate,
        )
        self.valence = Valence()
        self._tick = 0

    def observe(self, pattern_centroid: NDArray[np.float64], actual_delta: float) -> float:
        """Observe a base-pattern centroid, categorize into meta-pattern.

        This is the recursive application: same observe() call, but on
        pattern data instead of raw signals. Returns meta-surprise.
        """
        signal = Signal(data=pattern_centroid.copy(), timestamp=self._tick)
        meta_surprise = self.world_model.observe(signal)

        # Update meta-valence: this category inherits the outcome (weaker)
        if self.world_model.current_pattern is not None:
            self.valence.update(
                self.world_model.current_pattern.pattern_id,
                actual_delta * 0.5,
            )

        self._tick += 1
        return meta_surprise

    @property
    def current_meta_pattern(self) -> Pattern | None:
        """The current meta-pattern (abstract category)."""
        return self.world_model.current_pattern

    @property
    def meta_pattern_count(self) -> int:
        """Number of abstract categories discovered."""
        return len(self.world_model.memory.distinction.patterns)

    @property
    def meta_association_count(self) -> int:
        """Number of abstract rules learned."""
        return self.world_model.memory.association_count

    def meta_valence_for(self, centroid: NDArray[np.float64]) -> float:
        """Get the abstract category-level valence for a base pattern.

        Used for generalization: even if a specific base-pattern has
        no direct valence, its category might. This is how the agent
        handles novel situations — by recognizing their category.
        """
        signal = Signal(data=centroid, timestamp=0)
        closest = self.world_model.memory.distinction.find_closest(signal)
        if closest is None:
            return 0.0
        pattern, sim = closest
        if sim < self.world_model.memory.distinction.similarity_threshold:
            return 0.0
        return self.valence.get(pattern.pattern_id)

    def predict_meta_transition(self) -> tuple[Pattern, float] | None:
        """Predict the next meta-pattern (abstract category transition).

        This is a proto-symbolic rule: "from category X, category Y follows."
        """
        return self.world_model.predict()

    def tick(self) -> float:
        """Advance one tick: decay meta-associations, compute cost."""
        return self.world_model.tick()
