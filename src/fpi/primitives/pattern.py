"""Pattern & Distinction — recognizing structure in the signal stream.

A Pattern is a recognized regularity: a centroid vector that summarizes a
cluster of similar signals. Distinction is the machinery that maps new signals
to existing patterns (or decides they're novel).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .salience import Salience
from .signal import Signal


@dataclass(slots=True, eq=False)
class Pattern:
    """A recognized regularity in the signal stream.

    The centroid is a running average of all signals that matched this pattern.
    exposure_count tracks how many times this pattern has been activated —
    more-exposed patterns are more stable and harder to shift.

    In Phase 3, patterns compete for survival: each has a last_activated tick
    and a fitness score. Under capacity pressure, the least-fit patterns are
    evicted — Neural Darwinism at the pattern level.

    Attributes:
        centroid: The representative vector for this pattern.
        pattern_id: Unique identifier.
        exposure_count: How many signals have contributed to this centroid.
        last_activated: Tick when this pattern was last matched.
    """

    centroid: NDArray[np.float64]
    pattern_id: int
    exposure_count: int = 0
    last_activated: int = 0

    @property
    def dim(self) -> int:
        return len(self.centroid)

    def similarity(self, signal: Signal) -> float:
        """Cosine similarity between this pattern's centroid and a signal."""
        norm_c = np.linalg.norm(self.centroid)
        norm_s = np.linalg.norm(signal.data)
        if norm_c == 0.0 and norm_s == 0.0:
            return 1.0  # Two zero vectors are identical (e.g., no entities nearby)
        if norm_c == 0.0 or norm_s == 0.0:
            return 0.0  # One is zero, other is not — maximally different
        return float(np.dot(self.centroid, signal.data) / (norm_c * norm_s))

    def weighted_similarity(
        self, signal: Signal, weights: NDArray[np.float64],
    ) -> float:
        """Cosine similarity with per-dimension weighting."""
        wc = self.centroid * weights
        ws = signal.data * weights
        norm_c = np.linalg.norm(wc)
        norm_s = np.linalg.norm(ws)
        if norm_c == 0.0 or norm_s == 0.0:
            return 0.0
        return float(np.dot(wc, ws) / (norm_c * norm_s))

    def modal_similarity(
        self,
        signal: Signal,
        weights: NDArray[np.float64],
        slices: list[tuple[int, int]],
    ) -> float:
        """Per-modality weighted cosine similarity, averaged across modalities.

        Each modality contributes equally regardless of dimension count.
        """
        total = 0.0
        n = 0
        for start, end in slices:
            wc = self.centroid[start:end] * weights[start:end]
            ws = signal.data[start:end] * weights[start:end]
            nc = np.linalg.norm(wc)
            ns = np.linalg.norm(ws)
            if nc > 0.0 and ns > 0.0:
                total += float(np.dot(wc, ws) / (nc * ns))
                n += 1
        return total / n if n > 0 else 0.0

    def specificity(self) -> float:
        """How specific is this pattern? High exposure + sharp centroid = specific.

        Specific patterns are expensive to maintain (Occam's razor / FEP).
        Sharpness = max(|centroid|) / mean(|centroid|): a pattern with one
        dominant dimension is more specific than a flat one.
        """
        abs_centroid = np.abs(self.centroid)
        mean_abs = float(np.mean(abs_centroid))
        if mean_abs < 1e-10:
            return 0.0
        sharpness = float(np.max(abs_centroid)) / mean_abs
        exposure_factor = min(self.exposure_count, 20) / 20.0
        return sharpness * exposure_factor

    def fitness(self, current_tick: int) -> float:
        """How valuable is this pattern? Used for eviction decisions.

        Fitness = exposure_count * recency. Patterns that are both frequently
        used and recently active survive. Unused or stale patterns die.
        """
        age = max(1, current_tick - self.last_activated)
        recency = 1.0 / age
        return self.exposure_count * recency

    def update_centroid(self, signal: Signal, learning_rate: float | None = None, tick: int = 0) -> None:
        """Shift the centroid toward a matching signal.

        If no explicit learning_rate is given, we use 1/(exposure_count+1)
        so early signals have a large effect and the centroid stabilizes
        over time.
        """
        if learning_rate is None:
            learning_rate = 1.0 / (self.exposure_count + 1)
        self.centroid = self.centroid + learning_rate * (signal.data - self.centroid)
        self.exposure_count += 1
        self.last_activated = tick


@dataclass(slots=True)
class Distinction:
    """The machinery of categorization — mapping signals to patterns.

    Distinction maintains a set of known patterns and, given a new signal,
    either matches it to an existing pattern or creates a new one.

    In Phase 3, Distinction has a capacity limit. When full, the least-fit
    pattern is evicted to make room — Neural Darwinism. The brain can't
    remember everything; useful patterns survive, useless ones die.

    Attributes:
        patterns: All known patterns.
        similarity_threshold: Minimum cosine similarity to count as a match.
        max_patterns: Capacity limit (None = unlimited, backward compatible).
        _next_id: Counter for assigning pattern IDs.
        _current_tick: Internal clock for fitness calculation.
        _last_evicted: Pattern ID of most recently evicted pattern (for cascade cleanup).
    """

    patterns: list[Pattern] = field(default_factory=list)
    similarity_threshold: float = 0.8
    max_patterns: int | None = None
    enable_salience: bool = False
    modality_slices: list[tuple[int, int]] | None = None
    _next_id: int = 0
    _current_tick: int = 0
    _last_evicted: int | None = None
    _salience: Salience | None = field(default=None, repr=False)
    _last_deviation: NDArray[np.float64] | None = field(default=None, repr=False)
    _salience_window: int = 6
    _deviation_buffer: list[NDArray[np.float64]] = field(default_factory=list, repr=False)

    # Adaptive threshold fields
    _adaptive: bool = False
    _threshold_min: float = 0.0
    _threshold_max: float = 1.0
    _adapt_interval: int = 100
    _adapt_rate: float = 0.05
    _obs_since_adapt: int = 0
    _creation_count: int = 0
    _match_sims: list[float] = field(default_factory=list, repr=False)
    _active_in_window: set[int] = field(default_factory=set, repr=False)

    def advance_tick(self) -> None:
        """Advance the internal clock."""
        self._current_tick += 1

    def _similarity(self, pattern: Pattern, signal: Signal) -> float:
        """Compute similarity, weighted by salience if enabled."""
        if self._salience is not None and self.modality_slices is not None:
            return pattern.modal_similarity(
                signal, self._salience.weights, self.modality_slices,
            )
        if self._salience is not None:
            return pattern.weighted_similarity(signal, self._salience.weights)
        return pattern.similarity(signal)

    def distinguish(self, signal: Signal) -> tuple[Pattern, float]:
        """Map a signal to its best-matching pattern, or create a new one.

        Returns:
            (matched_pattern, similarity_score).
            If a new pattern was created, similarity is 1.0 (exact match to itself).
        """
        self._last_evicted = None  # Reset per call
        self._last_deviation = None

        # Lazy-init salience on first signal
        if self.enable_salience and self._salience is None:
            self._salience = Salience.uniform(signal.dim)

        if not self.patterns:
            return self._create_pattern(signal), 1.0

        best_pattern = self.patterns[0]
        best_sim = self._similarity(best_pattern, signal)
        for p in self.patterns[1:]:
            sim = self._similarity(p, signal)
            if sim > best_sim:
                best_sim = sim
                best_pattern = p

        if best_sim >= self.similarity_threshold:
            # Compute deviation BEFORE centroid update (for salience learning)
            if self._salience is not None:
                dev = np.abs(signal.data - best_pattern.centroid)
                self._last_deviation = dev
                self._deviation_buffer.append(dev)
                if len(self._deviation_buffer) > self._salience_window:
                    self._deviation_buffer.pop(0)
            best_pattern.update_centroid(signal, tick=self._current_tick)
            # Track for adaptive thresholds
            if self._adaptive:
                self._match_sims.append(best_sim)
                self._active_in_window.add(best_pattern.pattern_id)
                self._maybe_adapt_threshold()
            return best_pattern, best_sim

        # New pattern created
        result = self._create_pattern(signal), 1.0
        if self._adaptive:
            self._creation_count += 1
            self._maybe_adapt_threshold()
        return result

    def find_closest(self, signal: Signal) -> tuple[Pattern, float] | None:
        """Find the closest pattern without creating or updating anything."""
        if not self.patterns:
            return None
        best_pattern = self.patterns[0]
        best_sim = self._similarity(best_pattern, signal)
        for p in self.patterns[1:]:
            sim = self._similarity(p, signal)
            if sim > best_sim:
                best_sim = sim
                best_pattern = p
        return best_pattern, best_sim

    def update_salience(self, error: float) -> None:
        """Update salience weights retroactively for all recent deviations.

        Mirrors Valence.adjust_retroactive(): when a significant outcome
        occurs, ALL recent deviations are updated with geometric decay so
        that older deviations receive less credit.
        """
        if self._salience is None or not self._deviation_buffer:
            return
        for i, dev in enumerate(reversed(self._deviation_buffer)):
            decayed_error = error * (0.85 ** i)
            self._salience.update(dev, decayed_error)

    def drain_evicted(self) -> list[int]:
        """Return and clear list of evicted pattern IDs since last drain.

        Used by AssociativeMemory to cascade-clean associations for evicted
        patterns. Supports both single evictions (Distinction) and multi-
        evictions (CompositionalDistinction, where one modal eviction
        cascades to multiple composite evictions).
        """
        if self._last_evicted is not None:
            result = [self._last_evicted]
            self._last_evicted = None
            return result
        return []

    def enable_adaptive(
        self,
        threshold_min: float | None = None,
        threshold_max: float | None = None,
        adapt_interval: int = 100,
        adapt_rate: float = 0.05,
    ) -> None:
        """Enable adaptive threshold adjustment.

        Args:
            threshold_min: Lower bound (default: initial - 0.15).
            threshold_max: Upper bound (default: initial + 0.10).
            adapt_interval: Observations between adaptations.
            adapt_rate: EMA rate for threshold changes.
        """
        self._adaptive = True
        self._threshold_min = (
            threshold_min if threshold_min is not None
            else max(0.0, self.similarity_threshold - 0.15)
        )
        self._threshold_max = (
            threshold_max if threshold_max is not None
            else min(1.0, self.similarity_threshold + 0.10)
        )
        self._adapt_interval = adapt_interval
        self._adapt_rate = adapt_rate

    def _maybe_adapt_threshold(self) -> None:
        """Check utilization metrics and adjust threshold if interval reached."""
        self._obs_since_adapt += 1
        if self._obs_since_adapt < self._adapt_interval:
            return

        interval = self._adapt_interval

        # Metrics
        creation_rate = self._creation_count / interval
        avg_match_sim = (
            sum(self._match_sims) / len(self._match_sims)
            if self._match_sims else 0.9
        )
        utilization = (
            len(self._active_in_window) / len(self.patterns)
            if self.patterns else 1.0
        )

        # Decision: compute target delta
        delta = 0.0
        if (creation_rate > 0.08
                and self.max_patterns is not None
                and len(self.patterns) >= self.max_patterns):
            delta += 0.02   # too many new patterns at capacity → raise threshold
        if utilization < 0.3:
            delta += 0.01   # many dead patterns → raise threshold
        if avg_match_sim > 0.97:
            delta -= 0.02   # everything matches too easily → lower threshold
        if avg_match_sim < 0.85:
            delta += 0.01   # matches are marginal → raise threshold

        # Apply with EMA and bounds
        target = self.similarity_threshold + delta
        self.similarity_threshold += self._adapt_rate * (target - self.similarity_threshold)
        self.similarity_threshold = max(
            self._threshold_min,
            min(self._threshold_max, self.similarity_threshold),
        )

        # Reset counters
        self._obs_since_adapt = 0
        self._creation_count = 0
        self._match_sims.clear()
        self._active_in_window.clear()

    def _evict_weakest(self) -> int:
        """Evict the least-fit pattern. Returns the evicted pattern_id."""
        weakest = min(self.patterns, key=lambda p: p.fitness(self._current_tick))
        evicted_id = weakest.pattern_id
        self.patterns.remove(weakest)
        return evicted_id

    def _create_pattern(self, signal: Signal) -> Pattern:
        # Evict if at capacity
        if self.max_patterns is not None and len(self.patterns) >= self.max_patterns:
            self._last_evicted = self._evict_weakest()

        pid = self._next_id
        self._next_id += 1
        pattern = Pattern(
            centroid=signal.data.copy(), pattern_id=pid,
            exposure_count=1, last_activated=self._current_tick,
        )
        self.patterns.append(pattern)
        return pattern
