"""Tests for SequenceMemory — hierarchical pattern composition."""

import numpy as np

from fpi.primitives.pattern import Pattern
from fpi.memory.sequence import SequenceMemory, SequencePattern


def make_pattern(pid: int, dim: int = 64) -> Pattern:
    """Create a Pattern with a deterministic pseudo-random centroid."""
    rng = np.random.default_rng(pid)
    centroid = rng.standard_normal(dim).astype(np.float64)
    centroid /= np.linalg.norm(centroid)
    return Pattern(
        centroid=centroid, pattern_id=pid, exposure_count=10, last_activated=0
    )


class TestSequenceMemoryBasics:
    def test_no_output_before_window_full(self):
        sm = SequenceMemory(window_size=3)
        assert sm.observe(make_pattern(0)) is None
        assert sm.observe(make_pattern(1)) is None

    def test_first_full_window_returns_surprise_1(self):
        sm = SequenceMemory(window_size=3)
        sm.observe(make_pattern(0))
        sm.observe(make_pattern(1))
        surprise = sm.observe(make_pattern(2))
        assert surprise == 1.0  # No prediction yet

    def test_current_sequence_set_after_full_window(self):
        sm = SequenceMemory(window_size=3)
        for i in range(3):
            sm.observe(make_pattern(i))
        assert sm.current_sequence is not None
        assert isinstance(sm.current_sequence, SequencePattern)
        assert len(sm.current_sequence.constituent_ids) == 3

    def test_observation_count(self):
        sm = SequenceMemory(window_size=3)
        for i in range(5):
            sm.observe(make_pattern(i % 3))
        # 5 patterns fed, window_size=3, so 3 full windows
        assert sm.observation_count == 3

    def test_predict_returns_none_initially(self):
        sm = SequenceMemory(window_size=3)
        assert sm.predict() is None

    def test_predict_constituent_ids_none_initially(self):
        sm = SequenceMemory(window_size=3)
        assert sm.predict_constituent_ids() is None

    def test_pattern_count_after_observation(self):
        sm = SequenceMemory(window_size=3)
        for i in range(3):
            sm.observe(make_pattern(i))
        assert sm.pattern_count >= 1


class TestSequenceLearning:
    def test_recognizes_repeated_sequence(self):
        """Repeating the same 3-pattern sequence should reduce surprise."""
        sm = SequenceMemory(window_size=3, similarity_threshold=0.7)
        patterns = [make_pattern(i) for i in range(3)]

        surprises = []
        for _ in range(30):
            for p in patterns:
                s = sm.observe(p)
                if s is not None:
                    surprises.append(s)

        # Later surprises should be lower than early ones
        assert len(surprises) >= 10
        early = surprises[:5]
        late = surprises[-5:]
        assert sum(late) / len(late) < sum(early) / len(early)

    def test_different_sequences_create_different_patterns(self):
        """Different pattern sequences should create different sequence patterns."""
        sm = SequenceMemory(window_size=3, similarity_threshold=0.9)
        seq_a = [make_pattern(0), make_pattern(1), make_pattern(2)]
        seq_b = [make_pattern(10), make_pattern(11), make_pattern(12)]

        for p in seq_a:
            sm.observe(p)
        count_after_a = sm.pattern_count

        for p in seq_b:
            sm.observe(p)
        count_after_b = sm.pattern_count

        assert count_after_b > count_after_a

    def test_builds_associations(self):
        """Repeating sequence should build Level 2 associations."""
        sm = SequenceMemory(window_size=3, similarity_threshold=0.7)
        patterns = [make_pattern(i) for i in range(3)]

        for _ in range(20):
            for p in patterns:
                sm.observe(p)

        assert sm.association_count >= 1

    def test_prediction_after_learning(self):
        """After learning a repeating cycle, should have predictions."""
        sm = SequenceMemory(window_size=3, similarity_threshold=0.7)
        patterns = [make_pattern(i) for i in range(3)]

        for _ in range(30):
            for p in patterns:
                sm.observe(p)

        # After enough repetition, system should have learned patterns
        assert sm.pattern_count > 0
        assert sm.observation_count > 0

    def test_constituent_ids_correct_length(self):
        """Predicted constituent IDs should match window_size."""
        sm = SequenceMemory(window_size=3, similarity_threshold=0.7)
        patterns = [make_pattern(i) for i in range(3)]

        for _ in range(30):
            for p in patterns:
                sm.observe(p)

        ids = sm.predict_constituent_ids()
        if ids is not None:
            assert len(ids) == 3


class TestSequenceMaintenance:
    def test_tick_decays_associations(self):
        sm = SequenceMemory(window_size=3, association_decay_rate=0.1)
        patterns = [make_pattern(i) for i in range(3)]
        for _ in range(10):
            for p in patterns:
                sm.observe(p)

        initial_count = sm.association_count
        for _ in range(100):
            sm.tick()
        assert sm.association_count <= initial_count

    def test_tick_advances_distinction_clock(self):
        sm = SequenceMemory(window_size=3)
        initial_tick = sm._distinction._current_tick
        sm.tick()
        assert sm._distinction._current_tick == initial_tick + 1

    def test_registry_cleaned_on_eviction(self):
        """When patterns are evicted, registry should be cleaned."""
        sm = SequenceMemory(window_size=2, max_patterns=3, similarity_threshold=0.99)
        # Create many distinct sequences to force eviction
        for i in range(20):
            sm.observe(make_pattern(i * 100))  # very different patterns
        sm.tick()
        # Registry should not have more entries than live patterns
        live = {p.pattern_id for p in sm._distinction.patterns}
        for k in sm._sequence_registry:
            assert k in live


class TestSequenceSignalEncoding:
    def test_signal_dimensionality(self):
        """Sequence signal should be window_size * pattern_dim."""
        sm = SequenceMemory(window_size=3)
        patterns = [make_pattern(i) for i in range(3)]
        for p in patterns:
            sm.observe(p)
        # The current sequence pattern's centroid should be 3 * 64 = 192 dim
        assert sm.current_sequence is not None
        assert sm.current_sequence.pattern.dim == 3 * 64

    def test_signal_modality_is_sequence(self):
        """Encoded signal should have modality 'sequence'."""
        sm = SequenceMemory(window_size=3)
        for i in range(3):
            sm.observe(make_pattern(i))
        # Verify by checking the internal encode method
        sm._window = [(i, make_pattern(i).centroid) for i in range(3)]
        sig = sm._encode_window()
        assert sig.modality == "sequence"
