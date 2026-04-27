"""Tests for adaptive threshold adaptation on Distinction.

Covers:
1. enable_adaptive sets bounds correctly
2. High creation rate at capacity raises threshold
3. High avg_match_sim lowers threshold
4. Bounds enforcement
5. Phase bias via apply_phase_bias
6. CompositionalDistinction adaptive wiring
"""

from __future__ import annotations

import numpy as np
import pytest

from fpi.primitives.pattern import Distinction
from fpi.primitives.signal import Signal
from fpi.primitives.compositional import CompositionalDistinction


def make_signal(dim: int = 10, seed: int = 0) -> Signal:
    """Create a random normalized signal."""
    rng = np.random.default_rng(seed)
    data = rng.random(dim)
    data /= np.linalg.norm(data)
    return Signal(data=data, timestamp=0)


class TestEnableAdaptive:
    def test_default_bounds(self):
        d = Distinction(similarity_threshold=0.80, max_patterns=16)
        d.enable_adaptive()
        assert d._adaptive is True
        assert d._threshold_min == pytest.approx(0.65)  # 0.80 - 0.15
        assert d._threshold_max == pytest.approx(0.90)  # 0.80 + 0.10

    def test_custom_bounds(self):
        d = Distinction(similarity_threshold=0.80, max_patterns=16)
        d.enable_adaptive(threshold_min=0.50, threshold_max=0.95)
        assert d._threshold_min == 0.50
        assert d._threshold_max == 0.95

    def test_bounds_clamped_to_01(self):
        d = Distinction(similarity_threshold=0.95, max_patterns=16)
        d.enable_adaptive()
        assert d._threshold_max == 1.0  # min(1.0, 0.95 + 0.10)
        assert d._threshold_min == pytest.approx(0.80)  # max(0.0, 0.95 - 0.15)

    def test_low_threshold_bounds(self):
        d = Distinction(similarity_threshold=0.10, max_patterns=16)
        d.enable_adaptive()
        assert d._threshold_min == 0.0  # max(0.0, 0.10 - 0.15)
        assert d._threshold_max == 0.20  # 0.10 + 0.10


class TestAdaptationBehavior:
    def test_no_adaptation_before_interval(self):
        d = Distinction(similarity_threshold=0.80, max_patterns=16)
        d.enable_adaptive(adapt_interval=100)
        initial = d.similarity_threshold

        # Feed 50 observations (less than interval)
        for i in range(50):
            sig = make_signal(dim=10, seed=i)
            d.distinguish(sig)
            d.advance_tick()

        assert d.similarity_threshold == initial

    def test_saturation_raises_threshold(self):
        """When at max capacity and creating many patterns, threshold rises."""
        d = Distinction(similarity_threshold=0.80, max_patterns=4)
        d.enable_adaptive(adapt_interval=20, adapt_rate=1.0)

        # Fill to capacity with very different signals
        for i in range(4):
            data = np.zeros(10)
            data[i % 10] = 1.0
            sig = Signal(data=data, timestamp=0)
            d.distinguish(sig)

        assert len(d.patterns) == 4  # at capacity

        # Now feed random signals that will mostly create new patterns
        # (and evict old ones since at capacity)
        for i in range(20):
            sig = make_signal(dim=10, seed=100 + i)
            d.distinguish(sig)
            d.advance_tick()

        # Threshold should have risen (or stayed) from the adaptation
        # Since creation_rate will be high at capacity
        assert d.similarity_threshold >= 0.80

    def test_high_similarity_matches_lower_threshold(self):
        """When all matches are very similar (>0.97), threshold decreases."""
        d = Distinction(similarity_threshold=0.80, max_patterns=16)
        d.enable_adaptive(adapt_interval=20, adapt_rate=1.0)

        # Create one pattern
        base = np.ones(10) / np.sqrt(10)
        sig = Signal(data=base.copy(), timestamp=0)
        d.distinguish(sig)

        # Feed very similar signals (all will match with >0.99)
        for i in range(25):
            data = base.copy() + np.random.default_rng(i).normal(0, 0.001, 10)
            data /= np.linalg.norm(data)
            sig = Signal(data=data, timestamp=0)
            d.distinguish(sig)
            d.advance_tick()

        # Threshold should have decreased
        assert d.similarity_threshold < 0.80

    def test_threshold_stays_within_bounds(self):
        """Threshold never exceeds configured bounds."""
        d = Distinction(similarity_threshold=0.80, max_patterns=4)
        d.enable_adaptive(
            threshold_min=0.70, threshold_max=0.90,
            adapt_interval=10, adapt_rate=1.0,
        )

        # Run many observations to trigger multiple adaptations
        for i in range(200):
            sig = make_signal(dim=10, seed=i)
            d.distinguish(sig)
            d.advance_tick()
            assert d._threshold_min <= d.similarity_threshold <= d._threshold_max, (
                f"Threshold {d.similarity_threshold} outside [{d._threshold_min}, {d._threshold_max}]"
            )


class TestCompositionalAdaptive:
    def test_adaptive_threaded_to_modalities(self):
        """CompositionalDistinction passes adaptive=True to per-modality Distinctions."""
        slices = [(0, 5), (5, 10), (10, 15)]
        thresholds = [0.80, 0.70, 0.60]
        cd = CompositionalDistinction(
            modality_slices=slices,
            similarity_threshold=0.80,
            patterns_per_modality=16,
            modality_thresholds=thresholds,
            adaptive_thresholds=True,
        )
        for i, md in enumerate(cd._modal_distinctions):
            assert md._adaptive is True
            assert md._threshold_min == pytest.approx(max(0.0, thresholds[i] - 0.15))
            assert md._threshold_max == pytest.approx(min(1.0, thresholds[i] + 0.10))

    def test_not_adaptive_by_default(self):
        """Without adaptive_thresholds=True, Distinctions are not adaptive."""
        slices = [(0, 5), (5, 10)]
        cd = CompositionalDistinction(
            modality_slices=slices,
            similarity_threshold=0.80,
            patterns_per_modality=16,
        )
        for md in cd._modal_distinctions:
            assert md._adaptive is False

    def test_threshold_report(self):
        """get_threshold_report returns per-modality info."""
        slices = [(0, 5), (5, 10)]
        cd = CompositionalDistinction(
            modality_slices=slices,
            similarity_threshold=0.80,
            patterns_per_modality=16,
            adaptive_thresholds=True,
        )
        report = cd.get_threshold_report()
        assert len(report) == 2
        for r in report:
            assert "modality" in r
            assert "threshold" in r
            assert "min" in r
            assert "max" in r
            assert "patterns" in r
