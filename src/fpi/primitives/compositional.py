"""Compositional Distinction — per-modality pattern recognition with exponential capacity.

Instead of matching full signals against monolithic patterns (50 slots = 50
distinguishable situations), this splits signals by modality and matches each
independently. A 'composite pattern' is a combination of per-modality patterns.

With 15 patterns per modality across 6 modalities:
- 90 base patterns stored
- Up to 15^6 ≈ 11 million distinguishable situations (implicit)
- In practice ~200-500 active composites (only observed combinations exist)

This solves FPI limitation #1 (pattern capacity) while using only existing
primitives: Signal, Pattern, Distinction — applied per-modality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .pattern import Distinction, Pattern
from .signal import Signal


class CompositionalDistinction:
    """Per-modality pattern recognition with compositional combination.

    Same interface as Distinction so it can be used as a drop-in replacement
    in AssociativeMemory. Internally manages one Distinction per modality
    slice, producing composite patterns from their combination.

    Attributes:
        modality_slices: List of (start, end) index pairs for each modality.
        similarity_threshold: Passed to per-modality Distinction instances.
        patterns_per_modality: Max patterns in each modality Distinction.
        enable_salience: Whether per-modality salience learning is enabled.
    """

    def __init__(
        self,
        modality_slices: list[tuple[int, int]],
        similarity_threshold: float = 0.8,
        patterns_per_modality: int = 15,
        enable_salience: bool = False,
        modality_thresholds: list[float] | None = None,
        adaptive_thresholds: bool = False,
    ) -> None:
        self.modality_slices = modality_slices
        self.similarity_threshold = similarity_threshold
        self.max_patterns: int | None = None  # No hard limit on composites
        self.enable_salience = enable_salience

        # One Distinction per modality — each operates on its own signal slice
        # Per-modality thresholds let volatile modalities (terrain, entities,
        # history) use looser matching to produce fewer, broader patterns.
        self._modal_distinctions: list[Distinction] = []
        for i, (_start, _end) in enumerate(modality_slices):
            threshold = (
                modality_thresholds[i]
                if modality_thresholds is not None
                else similarity_threshold
            )
            d = Distinction(
                similarity_threshold=threshold,
                max_patterns=patterns_per_modality,
                enable_salience=enable_salience,
            )
            if adaptive_thresholds:
                d.enable_adaptive(
                    threshold_min=max(0.0, threshold - 0.15),
                    threshold_max=min(1.0, threshold + 0.10),
                )
            self._modal_distinctions.append(d)

        # Composite patterns: keyed by tuple of per-modality pattern IDs
        self._composites: dict[tuple[int, ...], Pattern] = {}
        # Reverse lookup: composite pattern_id -> modal key tuple
        self._composite_to_key: dict[int, tuple[int, ...]] = {}
        self._next_id: int = 0
        self._current_tick: int = 0

        # Eviction tracking — modal evictions cascade to composite evictions
        self._evicted_ids: list[int] = []

        # Interface compat with Distinction
        self._last_evicted: int | None = None

        # Deviation buffer for salience (aggregated across modalities)
        self._last_deviation: NDArray[np.float64] | None = None

    @property
    def patterns(self) -> list[Pattern]:
        """All active composite patterns."""
        return list(self._composites.values())

    def distinguish(self, signal: Signal) -> tuple[Pattern, float]:
        """Map a signal to a composite pattern by per-modality matching.

        Splits the signal into modality slices, matches each independently,
        then creates/retrieves the composite pattern for that combination.

        Returns:
            (composite_pattern, average_similarity) where the similarity is
            the mean of per-modality similarities.
        """
        self._last_evicted = None
        self._evicted_ids.clear()
        self._last_deviation = None

        # 1. Per-modality distinction
        modal_patterns: list[Pattern] = []
        modal_sims: list[float] = []

        for i, (start, end) in enumerate(self.modality_slices):
            modal_signal = Signal(
                data=signal.data[start:end],
                timestamp=signal.timestamp,
            )
            pattern, sim = self._modal_distinctions[i].distinguish(modal_signal)
            modal_patterns.append(pattern)
            modal_sims.append(sim)

        # 2. Cascade any modal evictions to composite evictions
        self._cascade_evictions()

        # 3. Get or create composite pattern
        key = tuple(p.pattern_id for p in modal_patterns)

        if key in self._composites:
            composite = self._composites[key]
            # Refresh centroid from current modal centroids (they may have shifted)
            composite.centroid = np.concatenate(
                [p.centroid for p in modal_patterns]
            )
            composite.exposure_count += 1
            composite.last_activated = self._current_tick
        else:
            centroid = np.concatenate([p.centroid for p in modal_patterns])
            composite = Pattern(
                centroid=centroid,
                pattern_id=self._next_id,
                exposure_count=1,
                last_activated=self._current_tick,
            )
            self._next_id += 1
            self._composites[key] = composite
            self._composite_to_key[composite.pattern_id] = key

        # Track deviation for salience (aggregate across modalities)
        deviations = []
        for i, md in enumerate(self._modal_distinctions):
            if md._last_deviation is not None:
                deviations.append(md._last_deviation)
            else:
                start, end = self.modality_slices[i]
                deviations.append(np.zeros(end - start))
        self._last_deviation = np.concatenate(deviations)

        # Set _last_evicted for backward compat (first evicted ID, if any)
        if self._evicted_ids:
            self._last_evicted = self._evicted_ids[0]

        avg_sim = sum(modal_sims) / len(modal_sims)
        return composite, avg_sim

    def find_closest(self, signal: Signal) -> tuple[Pattern, float] | None:
        """Find the closest composite pattern without creating or updating.

        Searches existing composites by cosine similarity against the full signal.
        """
        if not self._composites:
            return None

        best_pattern: Pattern | None = None
        best_sim = -1.0
        for composite in self._composites.values():
            sim = composite.similarity(signal)
            if sim > best_sim:
                best_sim = sim
                best_pattern = composite

        if best_pattern is None:
            return None
        return best_pattern, best_sim

    def advance_tick(self) -> None:
        """Advance the internal clock for all modality distinctions."""
        self._current_tick += 1
        for md in self._modal_distinctions:
            md.advance_tick()

    def update_salience(self, error: float) -> None:
        """Update salience weights in all modality distinctions."""
        for md in self._modal_distinctions:
            md.update_salience(error)

    def drain_evicted(self) -> list[int]:
        """Return and clear all composite pattern IDs evicted since last call.

        When a modality pattern is evicted, ALL composite patterns that
        included it become invalid. This returns those invalidated IDs
        so AssociativeMemory can clean up their associations.
        """
        result = list(self._evicted_ids)
        self._evicted_ids.clear()
        self._last_evicted = None
        return result

    def get_threshold_report(self) -> list[dict]:
        """Return current threshold status for each modality (diagnostics)."""
        return [
            {
                "modality": i,
                "threshold": md.similarity_threshold,
                "min": md._threshold_min,
                "max": md._threshold_max,
                "patterns": len(md.patterns),
                "adaptive": md._adaptive,
            }
            for i, md in enumerate(self._modal_distinctions)
        ]

    def _cascade_evictions(self) -> None:
        """Check all modality distinctions for evictions, cascade to composites.

        When a modality distinction evicts a pattern (because it's at capacity
        and a new pattern is more interesting), every composite pattern that
        referenced that modal pattern becomes invalid and must be removed.
        """
        for i, md in enumerate(self._modal_distinctions):
            if md._last_evicted is not None:
                evicted_modal_id = md._last_evicted
                md._last_evicted = None

                # Find all composite patterns that include this modal pattern
                to_remove: list[tuple[int, ...]] = []
                for key in self._composites:
                    if key[i] == evicted_modal_id:
                        to_remove.append(key)

                for key in to_remove:
                    composite = self._composites.pop(key)
                    self._composite_to_key.pop(composite.pattern_id, None)
                    self._evicted_ids.append(composite.pattern_id)

    @property
    def _salience(self):
        """Aggregate salience weights across all modalities (for compatibility)."""
        if not self.enable_salience:
            return None
        # Check if any modality has initialized salience
        weights = []
        for md in self._modal_distinctions:
            if md._salience is not None:
                weights.append(md._salience.weights)
            else:
                return None
        return type('_AggSalience', (), {'weights': np.concatenate(weights)})()

    def _similarity(self, pattern: Pattern, signal: Signal) -> float:
        """Compute similarity between a composite pattern and a signal.

        Uses per-modality cosine similarity averaged equally, matching
        the modal_similarity approach.
        """
        total = 0.0
        n = 0
        for start, end in self.modality_slices:
            pc = pattern.centroid[start:end]
            sd = signal.data[start:end]
            nc = np.linalg.norm(pc)
            ns = np.linalg.norm(sd)
            if nc > 0.0 and ns > 0.0:
                total += float(np.dot(pc, sd) / (nc * ns))
                n += 1
        return total / n if n > 0 else 0.0
