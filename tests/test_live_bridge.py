"""Tests for WatcherBridge — society-level signal encoding for watchers."""

import numpy as np
import pytest

from fpi.live.bridge import WatcherBridge


class TestWatcherBridge:
    def test_encoding_shape(self):
        bridge = WatcherBridge(n_watchers=3, bases_per_dim=6)
        sig = bridge.encode(
            {"a": 0.5, "b": 0.3, "c": 0.8},
            {"a": 0.9, "b": 0.7, "c": 0.5},
            timestamp=0,
        )
        assert sig.dim == 36  # 3 * 2 * 6

    def test_similar_states_similar_signals(self):
        bridge = WatcherBridge(n_watchers=2)
        s1 = bridge.encode({"a": 0.1, "b": 0.1}, {"a": 0.9, "b": 0.9}, 0)
        s2 = bridge.encode({"a": 0.15, "b": 0.12}, {"a": 0.88, "b": 0.87}, 0)
        sim = s1.cosine_similarity(s2)
        assert sim > 0.9, f"Similar states should be similar, got {sim}"

    def test_different_states_different_signals(self):
        bridge = WatcherBridge(n_watchers=2)
        s1 = bridge.encode({"a": 0.1, "b": 0.1}, {"a": 0.9, "b": 0.9}, 0)
        s2 = bridge.encode({"a": 0.9, "b": 0.9}, {"a": 0.1, "b": 0.1}, 0)
        sim = s1.cosine_similarity(s2)
        assert sim < 0.5, f"Different states should be different, got {sim}"

    def test_modality_is_collective(self):
        bridge = WatcherBridge(n_watchers=1)
        sig = bridge.encode({"a": 0.5}, {"a": 0.5}, 0)
        assert sig.modality == "collective"

    def test_signal_dim_property(self):
        bridge = WatcherBridge(n_watchers=4, bases_per_dim=8)
        assert bridge.signal_dim == 64  # 4 * 2 * 8

    def test_decode_summary(self):
        bridge = WatcherBridge(n_watchers=2)
        summary = bridge.decode_summary(
            {"a": 0.3, "b": 0.7},
            {"a": 0.9, "b": 0.4},
        )
        assert summary["a"]["surprise"] == 0.3
        assert summary["b"]["vitality"] == 0.4

    def test_deterministic_ordering(self):
        """Sorted watcher names ensure deterministic encoding."""
        bridge = WatcherBridge(n_watchers=2)
        s1 = bridge.encode({"z": 0.1, "a": 0.9}, {"z": 0.5, "a": 0.5}, 0)
        s2 = bridge.encode({"a": 0.9, "z": 0.1}, {"a": 0.5, "z": 0.5}, 0)
        assert s1.cosine_similarity(s2) == pytest.approx(1.0)
