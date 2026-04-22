"""Tests for AttentionGate (Global Workspace Theory implementation)."""

import numpy as np
import pytest

from fpi.primitives.attention import AttentionChannel, AttentionGate
from fpi.primitives.signal import Signal


class TestAttentionChannel:
    def test_creation(self):
        ch = AttentionChannel(name="position", slice_start=0, slice_end=6)
        assert ch.name == "position"
        assert not ch.suppressed

    def test_priority_default(self):
        ch = AttentionChannel(name="test", slice_start=0, slice_end=4)
        assert ch.priority == 1.0


class TestAttentionGate:
    def test_compete_marks_losers(self):
        gate = AttentionGate(capacity=1)
        channels = [
            AttentionChannel("a", 0, 4, priority=0.5),
            AttentionChannel("b", 4, 8, priority=0.9),
            AttentionChannel("c", 8, 12, priority=0.3),
        ]
        result = gate.compete(channels)
        # Highest priority (b) should be winner
        winners = [ch for ch in result if not ch.suppressed]
        losers = [ch for ch in result if ch.suppressed]
        assert len(winners) == 1
        assert winners[0].name == "b"
        assert len(losers) == 2

    def test_capacity_limits_winners(self):
        gate = AttentionGate(capacity=2)
        channels = [
            AttentionChannel("a", 0, 4, priority=0.9),
            AttentionChannel("b", 4, 8, priority=0.8),
            AttentionChannel("c", 8, 12, priority=0.1),
        ]
        gate.compete(channels)
        winners = [ch for ch in channels if not ch.suppressed]
        assert len(winners) == 2

    def test_gate_attenuates_low_priority(self):
        gate = AttentionGate(capacity=1, suppress_factor=0.1)
        obs = Signal(data=np.ones(8))
        slices = [(0, 4), (4, 8)]
        priorities = [0.9, 0.1]

        gated = gate.gate(obs, slices, priorities)

        # First modality (high priority) should be unchanged
        assert np.allclose(gated.data[:4], 1.0)
        # Second modality (low priority) should be attenuated
        assert np.allclose(gated.data[4:], 0.1)

    def test_gate_preserves_winner_fully(self):
        gate = AttentionGate(capacity=2, suppress_factor=0.0)
        obs = Signal(data=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        slices = [(0, 2), (2, 4), (4, 6)]
        priorities = [0.8, 0.9, 0.1]

        gated = gate.gate(obs, slices, priorities)

        # Top 2 priorities: mod 1 (0.9) and mod 0 (0.8) should be preserved
        assert gated.data[0] == 1.0
        assert gated.data[1] == 2.0
        assert gated.data[2] == 3.0
        assert gated.data[3] == 4.0
        # Loser (mod 2) should be zeroed (suppress_factor=0.0)
        assert gated.data[4] == 0.0
        assert gated.data[5] == 0.0

    def test_gate_mismatched_lengths_returns_original(self):
        gate = AttentionGate(capacity=1)
        obs = Signal(data=np.ones(8))
        slices = [(0, 4), (4, 8)]
        priorities = [0.5]  # Wrong length
        gated = gate.gate(obs, slices, priorities)
        assert np.array_equal(gated.data, obs.data)

    def test_gate_preserves_metadata(self):
        gate = AttentionGate(capacity=1)
        obs = Signal(data=np.ones(4), timestamp=42, modality="test")
        slices = [(0, 2), (2, 4)]
        priorities = [1.0, 0.0]
        gated = gate.gate(obs, slices, priorities)
        assert gated.timestamp == 42
        assert gated.modality == "test"

    def test_all_winners_when_capacity_exceeds_channels(self):
        gate = AttentionGate(capacity=5)
        channels = [
            AttentionChannel("a", 0, 4, priority=0.5),
            AttentionChannel("b", 4, 8, priority=0.3),
        ]
        gate.compete(channels)
        assert all(not ch.suppressed for ch in channels)

    def test_effective_capacity_no_arousal(self):
        gate = AttentionGate(capacity=3)
        assert gate.effective_capacity(0.0) == 3

    def test_effective_capacity_high_arousal(self):
        gate = AttentionGate(capacity=3)
        assert gate.effective_capacity(1.0) == 1  # Tunnel vision

    def test_effective_capacity_medium_arousal(self):
        gate = AttentionGate(capacity=4)
        # arousal=0.5 -> reduction = int(0.5 * 3) = 1 -> 3
        assert gate.effective_capacity(0.5) == 3

    def test_compete_with_arousal_narrows(self):
        gate = AttentionGate(capacity=3)
        channels = [
            AttentionChannel("a", 0, 4, priority=0.9),
            AttentionChannel("b", 4, 8, priority=0.5),
            AttentionChannel("c", 8, 12, priority=0.1),
        ]
        # No arousal: all 3 win
        gate.compete(channels, arousal=0.0)
        winners_0 = [ch for ch in channels if not ch.suppressed]
        assert len(winners_0) == 3

        # High arousal: only 1 wins (tunnel vision)
        gate.compete(channels, arousal=1.0)
        winners_1 = [ch for ch in channels if not ch.suppressed]
        assert len(winners_1) == 1
        assert winners_1[0].name == "a"

    def test_gate_with_arousal(self):
        gate = AttentionGate(capacity=2, suppress_factor=0.1)
        obs = Signal(data=np.ones(6))
        slices = [(0, 2), (2, 4), (4, 6)]
        priorities = [0.9, 0.5, 0.1]

        # No arousal: top 2 win
        gated_0 = gate.gate(obs, slices, priorities, arousal=0.0)
        assert np.allclose(gated_0.data[:4], 1.0)  # Both winners
        assert np.allclose(gated_0.data[4:], 0.1)  # Loser attenuated

        # High arousal: only top 1 wins
        gated_1 = gate.gate(obs, slices, priorities, arousal=1.0)
        assert np.allclose(gated_1.data[:2], 1.0)  # Winner
        assert np.allclose(gated_1.data[2:4], 0.1)  # Now suppressed
        assert np.allclose(gated_1.data[4:], 0.1)  # Still suppressed
