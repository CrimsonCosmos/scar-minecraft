"""Tests for AssociativeMemory."""

import numpy as np
import pytest

from fpi.primitives.signal import Signal
from fpi.memory.associative import AssociativeMemory


class TestAssociativeMemory:
    def test_store_creates_pattern(self):
        mem = AssociativeMemory()
        s = Signal.from_list([1.0, 0.0, 0.0])
        p = mem.store(s)
        assert p.pattern_id == 0
        assert mem.pattern_count == 1

    def test_store_reuses_similar_pattern(self):
        mem = AssociativeMemory(similarity_threshold=0.9)
        s1 = Signal.from_list([1.0, 0.0, 0.0])
        s2 = Signal.from_list([0.99, 0.01, 0.0])
        p1 = mem.store(s1)
        p2 = mem.store(s2)
        assert p1.pattern_id == p2.pattern_id
        assert mem.pattern_count == 1

    def test_recall(self):
        mem = AssociativeMemory()
        mem.store(Signal.from_list([1.0, 0.0, 0.0]))
        mem.store(Signal.from_list([0.0, 1.0, 0.0]))
        mem.store(Signal.from_list([0.0, 0.0, 1.0]))

        results = mem.recall(Signal.from_list([0.9, 0.1, 0.0]))
        assert len(results) == 3
        # Best match should be the first pattern
        assert results[0][0].pattern_id == 0
        assert results[0][1] > results[1][1]

    def test_associate_and_predict(self):
        mem = AssociativeMemory(similarity_threshold=0.9)
        # Store two distinct patterns
        s1 = Signal.from_list([1.0, 0.0, 0.0, 0.0])
        s2 = Signal.from_list([0.0, 1.0, 0.0, 0.0])
        p1 = mem.store(s1)
        p2 = mem.store(s2)

        # Associate p1 → p2
        mem.associate(p1.pattern_id, p2.pattern_id)

        # Predict from p1
        prediction = mem.predict_next(p1.pattern_id)
        assert prediction is not None
        assert prediction[0].pattern_id == p2.pattern_id

    def test_build_associations_from_sequence(self):
        mem = AssociativeMemory(similarity_threshold=0.9)
        signals = [
            Signal.from_list([1.0, 0.0, 0.0, 0.0]),
            Signal.from_list([0.0, 1.0, 0.0, 0.0]),
            Signal.from_list([0.0, 0.0, 1.0, 0.0]),
        ]
        for s in signals:
            mem.store(s)

        mem.build_associations_from_sequence()
        assert mem.association_count == 2  # 0→1, 1→2

    def test_weaken(self):
        mem = AssociativeMemory()
        s1 = Signal.from_list([1.0, 0.0])
        s2 = Signal.from_list([0.0, 1.0])
        p1 = mem.store(s1)
        p2 = mem.store(s2)
        assoc = mem.associate(p1.pattern_id, p2.pattern_id)
        initial_strength = assoc.strength

        mem.weaken(p1.pattern_id, p2.pattern_id)
        assert assoc.strength < initial_strength

    def test_clear_recent(self):
        mem = AssociativeMemory()
        mem.store(Signal.from_list([1.0, 0.0]))
        mem.store(Signal.from_list([0.0, 1.0]))
        assert len(mem._recent_pattern_ids) == 2
        mem.clear_recent()
        assert len(mem._recent_pattern_ids) == 0
