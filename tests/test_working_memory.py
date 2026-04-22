"""Tests for WorkingMemory (Baddeley's capacity-limited active buffer)."""

import numpy as np

from fpi.memory.working_memory import WorkingMemory, WorkingMemoryItem


class TestWorkingMemory:
    def test_hold_and_retrieve(self):
        wm = WorkingMemory(capacity=4)
        wm.hold(1, np.array([1.0, 0.0]), valence=0.5)
        assert wm.count == 1
        assert wm.contains(1)

    def test_capacity_limit(self):
        wm = WorkingMemory(capacity=4)
        for i in range(5):
            wm.hold(i, np.zeros(2), valence=float(i))
        # Should have exactly 4 items (5th evicted lowest-valence)
        assert wm.count == 4
        # Item 0 (valence=0.0, lowest) should have been evicted
        assert not wm.contains(0)
        assert wm.contains(4)

    def test_evicts_lowest_abs_valence(self):
        wm = WorkingMemory(capacity=2)
        wm.hold(1, np.zeros(2), valence=0.8)
        wm.hold(2, np.zeros(2), valence=-0.1)
        # Add third → evicts item 2 (|valence|=0.1 is lowest)
        wm.hold(3, np.zeros(2), valence=0.5)
        assert wm.contains(1)
        assert not wm.contains(2)
        assert wm.contains(3)

    def test_decay_evicts_unrehearsed(self):
        wm = WorkingMemory(capacity=4, decay_ticks=3)
        wm.hold(1, np.zeros(2), valence=0.5)
        # Age it past decay_ticks
        wm.tick()  # age=1
        wm.tick()  # age=2
        wm.tick()  # age=3
        assert wm.contains(1)  # age == decay_ticks, still alive
        wm.tick()  # age=4 > decay_ticks
        assert not wm.contains(1)

    def test_refresh_prevents_decay(self):
        wm = WorkingMemory(capacity=4, decay_ticks=3)
        wm.hold(1, np.zeros(2), valence=0.5)
        wm.tick()  # age=1
        wm.tick()  # age=2
        wm.refresh(1)  # reset age to 0
        wm.tick()  # age=1
        wm.tick()  # age=2
        wm.tick()  # age=3
        assert wm.contains(1)  # Still alive because of refresh

    def test_hold_existing_refreshes(self):
        wm = WorkingMemory(capacity=4, decay_ticks=3)
        wm.hold(1, np.zeros(2), valence=0.5)
        wm.tick()  # age=1
        wm.tick()  # age=2
        # Hold again should refresh
        wm.hold(1, np.zeros(2), valence=0.6)
        assert wm.count == 1  # Not duplicated
        wm.tick()  # age=1 (reset by hold)
        wm.tick()  # age=2
        wm.tick()  # age=3
        assert wm.contains(1)  # Still alive

    def test_maintenance_cost(self):
        wm = WorkingMemory(capacity=4, maintenance_cost=0.01)
        wm.hold(1, np.zeros(2), valence=0.5)
        wm.hold(2, np.zeros(2), valence=0.3)
        cost = wm.tick()
        assert cost == 0.02  # 2 items * 0.01

    def test_best_item(self):
        wm = WorkingMemory(capacity=4)
        wm.hold(1, np.zeros(2), valence=-0.3)
        wm.hold(2, np.zeros(2), valence=0.8)
        wm.hold(3, np.zeros(2), valence=0.5)
        best = wm.best_item()
        assert best is not None
        assert best.pattern_id == 2

    def test_best_item_empty(self):
        wm = WorkingMemory(capacity=4)
        assert wm.best_item() is None

    def test_contents_sorted_by_age(self):
        wm = WorkingMemory(capacity=4, decay_ticks=20)
        wm.hold(1, np.zeros(2), valence=0.5)
        wm.tick()
        wm.hold(2, np.zeros(2), valence=0.3)
        contents = wm.contents()
        # Item 2 is newer (age=0), item 1 is older (age=1)
        assert contents[0].pattern_id == 2
        assert contents[1].pattern_id == 1

    def test_clear(self):
        wm = WorkingMemory(capacity=4)
        wm.hold(1, np.zeros(2), valence=0.5)
        wm.hold(2, np.zeros(2), valence=0.3)
        wm.clear()
        assert wm.count == 0

    def test_refresh_returns_false_for_missing(self):
        wm = WorkingMemory(capacity=4)
        assert wm.refresh(99) is False
