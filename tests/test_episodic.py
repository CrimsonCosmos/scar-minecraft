"""Tests for EpisodicMemory — remembering specific events."""

import numpy as np

from fpi.memory.episodic import Episode, EpisodicMemory


def _make_episode(
    tick: int = 0,
    pattern_id: int = 0,
    centroid: list[float] | None = None,
    surprise: float = 0.9,
    vitality: float = 0.8,
    context_ids: tuple[int, ...] = (),
    valence: float = -0.1,
) -> Episode:
    if centroid is None:
        centroid = [1.0, 0.0, 0.0]
    return Episode(
        tick=tick,
        pattern_id=pattern_id,
        centroid=np.array(centroid, dtype=np.float64),
        surprise=surprise,
        vitality=vitality,
        context_ids=context_ids,
        valence=valence,
    )


class TestEpisodicMemory:
    def test_record_and_count(self):
        mem = EpisodicMemory(capacity=10)
        assert mem.count == 0
        mem.record(_make_episode(tick=0))
        assert mem.count == 1
        mem.record(_make_episode(tick=1))
        assert mem.count == 2

    def test_fifo_eviction(self):
        mem = EpisodicMemory(capacity=3)
        for i in range(5):
            mem.record(_make_episode(tick=i, pattern_id=i))
        assert mem.count == 3
        # Oldest (tick=0,1) should be evicted
        ticks = [ep.tick for ep in mem.get_recent(10)]
        assert ticks == [2, 3, 4]

    def test_recall_by_similarity(self):
        mem = EpisodicMemory(capacity=10, recall_threshold=0.7)
        # Store episodes with distinct centroids
        mem.record(_make_episode(centroid=[1.0, 0.0, 0.0], tick=0, pattern_id=0))
        mem.record(_make_episode(centroid=[0.0, 1.0, 0.0], tick=1, pattern_id=1))
        mem.record(_make_episode(centroid=[0.0, 0.0, 1.0], tick=2, pattern_id=2))

        # Query similar to first episode
        query = np.array([0.9, 0.1, 0.0], dtype=np.float64)
        results = mem.recall(query, k=3)
        assert len(results) >= 1
        assert results[0].pattern_id == 0  # Most similar to [1,0,0]

    def test_recall_respects_threshold(self):
        mem = EpisodicMemory(capacity=10, recall_threshold=0.99)
        mem.record(_make_episode(centroid=[1.0, 0.0, 0.0]))
        # Query orthogonal — should not match
        query = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        results = mem.recall(query, k=3)
        assert len(results) == 0

    def test_recall_k_limit(self):
        mem = EpisodicMemory(capacity=10, recall_threshold=0.5)
        # Store 5 similar episodes
        for i in range(5):
            mem.record(_make_episode(centroid=[1.0, 0.1 * i, 0.0], tick=i))
        query = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        results = mem.recall(query, k=2)
        assert len(results) <= 2

    def test_recall_sorted_by_similarity(self):
        mem = EpisodicMemory(capacity=10, recall_threshold=0.5)
        mem.record(_make_episode(centroid=[1.0, 0.5, 0.0], tick=0, pattern_id=0))
        mem.record(_make_episode(centroid=[1.0, 0.0, 0.0], tick=1, pattern_id=1))
        query = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        results = mem.recall(query, k=10)
        assert len(results) >= 2
        # Second episode (exact match) should come first
        assert results[0].pattern_id == 1

    def test_recall_empty_memory(self):
        mem = EpisodicMemory(capacity=10)
        query = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        results = mem.recall(query)
        assert results == []

    def test_get_recent(self):
        mem = EpisodicMemory(capacity=10)
        for i in range(7):
            mem.record(_make_episode(tick=i))
        recent = mem.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].tick == 6

    def test_episode_frozen(self):
        ep = _make_episode()
        # Episode is frozen dataclass — should be immutable
        try:
            ep.tick = 99  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass
