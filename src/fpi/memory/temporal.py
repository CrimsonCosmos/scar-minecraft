"""TemporalHierarchy — multiple timescales in parallel.

Wraps multiple SequenceMemory instances at different window sizes.
Short-term catches A->B->C, long-term catches the 15-step rhythm
underlying them. This is how humans hear both individual notes AND
the melody.

This is NOT a deep stack (Level 3, Level 4...) — it's parallel
resolution at different window sizes.
"""

from __future__ import annotations

from ..primitives.pattern import Pattern
from .sequence import SequenceMemory, SequencePattern


class TemporalHierarchy:
    """Multiple SequenceMemory instances at different window sizes.

    Args:
        scales: Tuple of window sizes for each timescale.
        similarity_threshold: For each scale's Distinction.
        max_patterns_per_scale: Capacity per scale.
        max_associations_per_scale: Association limit per scale.
    """

    def __init__(
        self,
        scales: tuple[int, ...] = (3, 7, 15),
        similarity_threshold: float = 0.7,
        max_patterns_per_scale: int = 15,
        max_associations_per_scale: int = 40,
    ) -> None:
        self.scales = scales
        self._memories: dict[int, SequenceMemory] = {}
        for scale in scales:
            self._memories[scale] = SequenceMemory(
                window_size=scale,
                similarity_threshold=similarity_threshold,
                max_patterns=max_patterns_per_scale,
                max_associations=max_associations_per_scale,
            )

    def observe(self, pattern: Pattern) -> dict[int, float | None]:
        """Feed a Level 1 pattern to all scales.

        Returns {scale: surprise | None} for each scale.
        None means the scale's window isn't full yet.
        """
        results: dict[int, float | None] = {}
        for scale, mem in self._memories.items():
            results[scale] = mem.observe(pattern)
        return results

    def predict(self) -> dict[int, tuple[SequencePattern, float] | None]:
        """Get predictions from each scale.

        Returns {scale: (prediction, confidence) | None}.
        """
        results: dict[int, tuple[SequencePattern, float] | None] = {}
        for scale, mem in self._memories.items():
            results[scale] = mem.predict()
        return results

    def tick(self) -> None:
        """Advance all scales one tick."""
        for mem in self._memories.values():
            mem.tick()

    def get_status(self) -> dict:
        """Per-scale status: pattern count, association count, predictions."""
        status: dict[str, object] = {"scales": {}}
        for scale, mem in self._memories.items():
            pred = mem.predict()
            pred_ids = mem.predict_constituent_ids()
            scale_status: dict[str, object] = {
                "pattern_count": mem.pattern_count,
                "association_count": mem.association_count,
                "observation_count": mem.observation_count,
                "predicted_next": list(pred_ids) if pred_ids else None,
            }
            if pred is not None:
                scale_status["prediction_confidence"] = pred[1]
            else:
                scale_status["prediction_confidence"] = None
            status["scales"][scale] = scale_status  # type: ignore[index]
        return status
