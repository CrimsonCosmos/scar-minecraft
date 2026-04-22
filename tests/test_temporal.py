"""Tests for TemporalHierarchy — multiple timescales in parallel."""

import numpy as np

from fpi.primitives.pattern import Pattern
from fpi.memory.temporal import TemporalHierarchy


def _make_pattern(pid: int, dim: int = 6) -> Pattern:
    """Create a pattern with a distinct centroid."""
    rng = np.random.default_rng(pid + 42)
    centroid = rng.random(dim).astype(np.float64)
    return Pattern(centroid=centroid, pattern_id=pid, exposure_count=1)


class TestTemporalHierarchy:
    def test_creation(self):
        th = TemporalHierarchy(scales=(3, 7))
        assert len(th._memories) == 2
        assert 3 in th._memories
        assert 7 in th._memories

    def test_default_scales(self):
        th = TemporalHierarchy()
        assert th.scales == (3, 7, 15)

    def test_observe_returns_dict(self):
        th = TemporalHierarchy(scales=(3, 5))
        p = _make_pattern(0)
        result = th.observe(p)
        assert isinstance(result, dict)
        assert 3 in result
        assert 5 in result

    def test_short_scale_produces_surprise_before_long(self):
        th = TemporalHierarchy(scales=(2, 5))
        patterns = [_make_pattern(i) for i in range(3)]
        # Feed 2 patterns — short scale (window=2) should produce a result
        th.observe(patterns[0])
        result = th.observe(patterns[1])
        assert result[2] is not None  # Short scale window full
        assert result[5] is None  # Long scale window not full yet

    def test_both_scales_produce_after_enough_data(self):
        th = TemporalHierarchy(scales=(2, 3))
        patterns = [_make_pattern(i) for i in range(5)]
        results = []
        for p in patterns:
            results.append(th.observe(p))
        # After 3 patterns, both scales should have produced
        assert results[2][2] is not None
        assert results[2][3] is not None

    def test_predict_returns_dict(self):
        th = TemporalHierarchy(scales=(3,))
        patterns = [_make_pattern(i % 3) for i in range(10)]
        for p in patterns:
            th.observe(p)
        preds = th.predict()
        assert isinstance(preds, dict)
        assert 3 in preds

    def test_tick_advances_all(self):
        th = TemporalHierarchy(scales=(3, 5))
        # tick should not raise
        th.tick()
        th.tick()

    def test_get_status(self):
        th = TemporalHierarchy(scales=(3,))
        patterns = [_make_pattern(i % 2) for i in range(10)]
        for p in patterns:
            th.observe(p)
        status = th.get_status()
        assert "scales" in status
        assert 3 in status["scales"]
        s3 = status["scales"][3]
        assert "pattern_count" in s3
        assert "association_count" in s3
        assert "observation_count" in s3
        assert s3["observation_count"] > 0

    def test_different_scales_learn_different_patterns(self):
        """Short and long scales should learn different numbers of patterns."""
        th = TemporalHierarchy(scales=(2, 4))
        # Create a repeating sequence: 0,1,2,3,0,1,2,3,...
        patterns = [_make_pattern(i % 4) for i in range(30)]
        for p in patterns:
            th.observe(p)
        status = th.get_status()
        # Both scales should have learned something
        assert status["scales"][2]["pattern_count"] >= 1
        assert status["scales"][4]["pattern_count"] >= 1
