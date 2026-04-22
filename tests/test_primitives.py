"""Tests for the five primitives: Signal, Pattern, Distinction, Association."""

import numpy as np
import pytest

from fpi.primitives.signal import Signal, SignalBundle
from fpi.primitives.pattern import Pattern, Distinction
from fpi.primitives.association import Association, AssociationMap


class TestSignal:
    def test_from_scalar(self):
        s = Signal.from_scalar(3.14, timestamp=1, modality="test")
        assert s.dim == 1
        assert s.data[0] == pytest.approx(3.14)
        assert s.timestamp == 1
        assert s.modality == "test"

    def test_from_list(self):
        s = Signal.from_list([1.0, 2.0, 3.0])
        assert s.dim == 3
        np.testing.assert_array_equal(s.data, [1.0, 2.0, 3.0])

    def test_distance(self):
        a = Signal.from_list([1.0, 0.0])
        b = Signal.from_list([0.0, 1.0])
        assert a.distance(b) == pytest.approx(np.sqrt(2))

    def test_distance_to_self(self):
        s = Signal.from_list([1.0, 2.0, 3.0])
        assert s.distance(s) == pytest.approx(0.0)

    def test_cosine_similarity_identical(self):
        s = Signal.from_list([1.0, 2.0, 3.0])
        assert s.cosine_similarity(s) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = Signal.from_list([1.0, 0.0])
        b = Signal.from_list([0.0, 1.0])
        assert a.cosine_similarity(b) == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        a = Signal.from_list([0.0, 0.0])
        b = Signal.from_list([1.0, 2.0])
        assert a.cosine_similarity(b) == 0.0


class TestSignalBundle:
    def test_from_signals(self):
        s1 = Signal.from_scalar(1.0, modality="a")
        s2 = Signal.from_scalar(2.0, modality="b")
        bundle = SignalBundle.from_signals(s1, s2)
        assert len(bundle.signals) == 2

    def test_combined_vector(self):
        s1 = Signal.from_list([1.0, 2.0])
        s2 = Signal.from_list([3.0])
        bundle = SignalBundle.from_signals(s1, s2)
        np.testing.assert_array_equal(bundle.combined_vector(), [1.0, 2.0, 3.0])

    def test_by_modality(self):
        s1 = Signal.from_scalar(1.0, modality="visual")
        s2 = Signal.from_scalar(2.0, modality="audio")
        s3 = Signal.from_scalar(3.0, modality="visual")
        bundle = SignalBundle.from_signals(s1, s2, s3)
        visual = bundle.by_modality("visual")
        assert len(visual) == 2

    def test_empty_bundle(self):
        bundle = SignalBundle()
        assert bundle.timestamp == 0
        assert len(bundle.combined_vector()) == 0


class TestPattern:
    def test_similarity_identical(self):
        p = Pattern(centroid=np.array([1.0, 0.0, 0.0]), pattern_id=0)
        s = Signal.from_list([1.0, 0.0, 0.0])
        assert p.similarity(s) == pytest.approx(1.0)

    def test_similarity_orthogonal(self):
        p = Pattern(centroid=np.array([1.0, 0.0]), pattern_id=0)
        s = Signal.from_list([0.0, 1.0])
        assert p.similarity(s) == pytest.approx(0.0)

    def test_update_centroid(self):
        p = Pattern(centroid=np.array([1.0, 0.0]), pattern_id=0, exposure_count=1)
        s = Signal.from_list([0.0, 1.0])
        p.update_centroid(s)
        # After update, centroid should move toward [0, 1]
        assert p.centroid[0] < 1.0
        assert p.centroid[1] > 0.0
        assert p.exposure_count == 2


class TestDistinction:
    def test_creates_pattern_from_nothing(self):
        d = Distinction()
        s = Signal.from_list([1.0, 0.0, 0.0])
        pattern, sim = d.distinguish(s)
        assert pattern.pattern_id == 0
        assert sim == 1.0
        assert len(d.patterns) == 1

    def test_matches_similar_signal(self):
        d = Distinction(similarity_threshold=0.9)
        s1 = Signal.from_list([1.0, 0.0, 0.0])
        s2 = Signal.from_list([0.99, 0.01, 0.0])
        p1, _ = d.distinguish(s1)
        p2, _ = d.distinguish(s2)
        assert p1.pattern_id == p2.pattern_id  # Same pattern
        assert len(d.patterns) == 1

    def test_creates_new_pattern_for_different_signal(self):
        d = Distinction(similarity_threshold=0.9)
        s1 = Signal.from_list([1.0, 0.0, 0.0])
        s2 = Signal.from_list([0.0, 1.0, 0.0])
        p1, _ = d.distinguish(s1)
        p2, _ = d.distinguish(s2)
        assert p1.pattern_id != p2.pattern_id
        assert len(d.patterns) == 2

    def test_find_closest(self):
        d = Distinction()
        d.distinguish(Signal.from_list([1.0, 0.0]))
        d.distinguish(Signal.from_list([0.0, 1.0]))
        result = d.find_closest(Signal.from_list([0.9, 0.1]))
        assert result is not None
        pattern, sim = result
        assert pattern.pattern_id == 0  # Closer to [1, 0]


class TestAssociation:
    def test_reinforce(self):
        a = Association(source_id=0, target_id=1, strength=0.0)
        a.reinforce(amount=0.5)
        assert a.strength > 0.0
        assert a.activation_count == 1

    def test_reinforce_caps_at_one(self):
        a = Association(source_id=0, target_id=1, strength=0.9)
        for _ in range(100):
            a.reinforce(amount=0.5)
        assert a.strength <= 1.0

    def test_weaken(self):
        a = Association(source_id=0, target_id=1, strength=0.8)
        a.weaken(amount=0.5)
        assert a.strength < 0.8

    def test_weaken_caps_at_zero(self):
        a = Association(source_id=0, target_id=1, strength=0.1)
        for _ in range(100):
            a.weaken(amount=0.5)
        assert a.strength >= 0.0


class TestAssociationMap:
    def test_get_or_create(self):
        am = AssociationMap()
        a = am.get_or_create(0, 1)
        assert a.source_id == 0
        assert a.target_id == 1
        # Getting again returns same object
        b = am.get_or_create(0, 1)
        assert a is b

    def test_from_source(self):
        am = AssociationMap()
        am.get_or_create(0, 1)
        am.get_or_create(0, 2)
        am.get_or_create(1, 2)
        assert len(am.from_source(0)) == 2
        assert len(am.from_source(1)) == 1

    def test_strongest_from(self):
        am = AssociationMap()
        a1 = am.get_or_create(0, 1)
        a2 = am.get_or_create(0, 2)
        a1.reinforce(0.3)
        a2.reinforce(0.8)
        strongest = am.strongest_from(0)
        assert strongest is not None
        assert strongest.target_id == 2
