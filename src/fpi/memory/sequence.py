"""SequenceMemory — hierarchical pattern composition.

Level 2 of the intelligence: recognizing patterns OF patterns.

Level 1: Signal -> Distinction -> Pattern -> Association -> Prediction
Level 2: [Pattern, Pattern, Pattern] -> encode as Signal -> Distinction ->
         SequencePattern -> Association -> Prediction

This reuses ALL existing primitives. No changes to Signal, Pattern,
Distinction, or Association. Just a new composition layer that watches
Level 1's output and finds higher-order structure.

This is the foundation of abstraction: a "word" is a sequence-pattern,
a "sentence" is a sequence of sequence-patterns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..primitives.signal import Signal
from ..primitives.pattern import Pattern, Distinction
from ..primitives.association import AssociationMap


@dataclass
class SequencePattern:
    """A recognized sequence — a Pattern at Level 2.

    Wraps a Level 2 Pattern with a record of which Level 1 pattern IDs
    compose it. This enables decomposition: given a sequence prediction,
    we can say "the next N patterns will be [A, B, C]."

    Attributes:
        pattern: The Level 2 Pattern (centroid in sequence-signal space).
        constituent_ids: The Level 1 pattern IDs this sequence was first
            recognized from (the prototype composition).
    """

    pattern: Pattern
    constituent_ids: tuple[int, ...]


class SequenceMemory:
    """Recognizes recurring multi-step patterns in the Level 1 pattern stream.

    Watches a sliding window of Level 1 patterns. When the window is full,
    encodes the sequence as a Signal (by concatenating the centroids of
    the constituent patterns), feeds it to a Level 2 Distinction, and
    builds Level 2 associations.

    Args:
        window_size: How many Level 1 patterns compose one sequence.
        similarity_threshold: For Level 2 Distinction.
        max_patterns: Capacity limit for Level 2 patterns.
        max_associations: Capacity limit for Level 2 associations.
        association_decay_rate: Decay rate for Level 2 associations.
    """

    def __init__(
        self,
        window_size: int = 3,
        similarity_threshold: float = 0.7,
        max_patterns: int = 20,
        max_associations: int = 60,
        association_decay_rate: float = 0.005,
    ) -> None:
        self.window_size = window_size
        self._distinction = Distinction(
            similarity_threshold=similarity_threshold,
            max_patterns=max_patterns,
        )
        self._associations = AssociationMap(
            max_associations=max_associations,
            decay_rate=association_decay_rate,
        )

        # Sliding window of recent (pattern_id, centroid) pairs
        self._window: list[tuple[int, NDArray[np.float64]]] = []

        # Level 2 state
        self._current_sequence: SequencePattern | None = None
        self._prev_sequence: SequencePattern | None = None
        self._last_prediction: tuple[SequencePattern, float] | None = None
        self.last_surprise: float = 1.0
        self.observation_count: int = 0

        # Map Level 2 pattern_id -> SequencePattern (for decomposition)
        self._sequence_registry: dict[int, SequencePattern] = {}

    def observe(self, pattern: Pattern) -> float | None:
        """Feed a Level 1 pattern into the sequence detector.

        Returns surprise if a full window was processed, None otherwise.
        The window slides by 1 each call (overlapping windows).
        """
        self._window.append((pattern.pattern_id, pattern.centroid.copy()))

        if len(self._window) < self.window_size:
            return None

        # Encode window as a signal
        signal = self._encode_window()

        # Track previous sequence
        self._prev_sequence = self._current_sequence

        # Categorize via Level 2 Distinction
        l2_pattern, _similarity = self._distinction.distinguish(signal)

        # Build or retrieve SequencePattern
        if l2_pattern.pattern_id not in self._sequence_registry:
            constituent_ids = tuple(
                pid for pid, _ in self._window[-self.window_size:]
            )
            seq_pat = SequencePattern(
                pattern=l2_pattern,
                constituent_ids=constituent_ids,
            )
            self._sequence_registry[l2_pattern.pattern_id] = seq_pat

        self._current_sequence = self._sequence_registry[l2_pattern.pattern_id]

        # Compute surprise
        surprise = self._compute_surprise(l2_pattern)
        self.last_surprise = surprise
        self.observation_count += 1

        # Update associations
        if self._prev_sequence is not None:
            prev_id = self._prev_sequence.pattern.pattern_id
            curr_id = l2_pattern.pattern_id
            if surprise < 0.5:
                assoc = self._associations.get_or_create(prev_id, curr_id)
                assoc.reinforce(tick=self._distinction._current_tick)
            else:
                if self._last_prediction is not None:
                    pred_id = self._last_prediction[0].pattern.pattern_id
                    old_assoc = self._associations.get(prev_id, pred_id)
                    if old_assoc is not None:
                        old_assoc.weaken()
                assoc = self._associations.get_or_create(prev_id, curr_id)
                assoc.reinforce(tick=self._distinction._current_tick)

        # Generate prediction
        self._last_prediction = self._predict_next(l2_pattern.pattern_id)

        # Slide window by 1 (keep overlap)
        self._window = self._window[1:]

        return surprise

    def _encode_window(self) -> Signal:
        """Encode the current window as a Signal by concatenating centroids."""
        centroids = [c for _, c in self._window[-self.window_size:]]
        data = np.concatenate(centroids).astype(np.float64)
        return Signal(
            data=data,
            timestamp=self._distinction._current_tick,
            modality="sequence",
        )

    def _compute_surprise(self, actual: Pattern) -> float:
        """Compute surprise at Level 2."""
        if self._last_prediction is None:
            return 1.0
        predicted_seq, _confidence = self._last_prediction
        if predicted_seq.pattern.pattern_id == actual.pattern_id:
            return 0.0
        # Cosine similarity between centroids
        norm_p = np.linalg.norm(predicted_seq.pattern.centroid)
        norm_a = np.linalg.norm(actual.centroid)
        if norm_p == 0.0 or norm_a == 0.0:
            return 1.0
        sim = float(
            np.dot(predicted_seq.pattern.centroid, actual.centroid)
            / (norm_p * norm_a)
        )
        return 1.0 - max(0.0, sim)

    def _predict_next(
        self, current_id: int
    ) -> tuple[SequencePattern, float] | None:
        """Predict the next sequence pattern."""
        strongest = self._associations.strongest_from(current_id)
        if strongest is None:
            return None
        seq = self._sequence_registry.get(strongest.target_id)
        if seq is None:
            return None
        return seq, strongest.strength

    def predict(self) -> tuple[SequencePattern, float] | None:
        """Get the current Level 2 prediction."""
        return self._last_prediction

    def predict_constituent_ids(self) -> tuple[int, ...] | None:
        """Decompose the predicted next sequence into Level 1 pattern IDs.

        This is the key capability: "after this sequence, the next N
        patterns will be [A, B, C]."
        """
        if self._last_prediction is None:
            return None
        return self._last_prediction[0].constituent_ids

    def tick(self) -> None:
        """Advance one tick: decay associations, advance Distinction clock."""
        self._distinction.advance_tick()
        self._associations.decay_all()

        # Clean up registry for evicted patterns
        live_ids = {p.pattern_id for p in self._distinction.patterns}
        dead_ids = [k for k in self._sequence_registry if k not in live_ids]
        for k in dead_ids:
            del self._sequence_registry[k]

    @property
    def current_sequence(self) -> SequencePattern | None:
        return self._current_sequence

    @property
    def pattern_count(self) -> int:
        return len(self._distinction.patterns)

    @property
    def association_count(self) -> int:
        return self._associations.count
