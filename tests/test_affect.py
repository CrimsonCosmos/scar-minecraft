"""Tests for AffectState and AffectStream (emotion as pattern in affect space)."""

import numpy as np
import pytest

from fpi.primitives.affect import AffectState, AffectStream


class TestAffectState:
    def test_creation(self):
        state = AffectState(valence=0.5, arousal=0.8)
        assert state.valence == 0.5
        assert state.arousal == 0.8

    def test_defaults(self):
        state = AffectState()
        assert state.valence == 0.0
        assert state.arousal == 0.0

    def test_as_array(self):
        state = AffectState(valence=-0.3, arousal=0.7)
        arr = state.as_array()
        assert arr.shape == (2,)
        assert arr[0] == pytest.approx(-0.3)
        assert arr[1] == pytest.approx(0.7)


class TestAffectStream:
    def test_encode_produces_correct_dim(self):
        stream = AffectStream(bases_per_dim=6)
        state = AffectState(valence=0.5, arousal=0.3)
        signal = stream.encode(state, timestamp=0)
        assert signal.dim == 12  # 6 + 6
        assert signal.modality == "affect"

    def test_encode_different_states_produce_different_signals(self):
        stream = AffectStream(bases_per_dim=6)
        s1 = stream.encode(AffectState(0.9, 0.1), timestamp=0)
        s2 = stream.encode(AffectState(-0.9, 0.9), timestamp=0)
        # Very different affect states should produce different signals
        cos_sim = float(np.dot(s1.data, s2.data) / (
            np.linalg.norm(s1.data) * np.linalg.norm(s2.data)
        ))
        assert cos_sim < 0.5

    def test_distinct_emotions_create_distinct_patterns(self):
        """Opposite affect regions → different emotion patterns."""
        stream = AffectStream(bases_per_dim=6, similarity_threshold=0.7)
        # "Fear": negative valence, high arousal
        p1, _ = stream.observe(AffectState(-0.8, 0.9), timestamp=0)
        # "Contentment": positive valence, low arousal
        p2, _ = stream.observe(AffectState(0.8, 0.1), timestamp=1)
        assert p1.pattern_id != p2.pattern_id

    def test_similar_emotions_merge(self):
        """Nearby affect states → same emotion pattern."""
        stream = AffectStream(bases_per_dim=6, similarity_threshold=0.7)
        p1, _ = stream.observe(AffectState(0.5, 0.3), timestamp=0)
        p2, _ = stream.observe(AffectState(0.55, 0.35), timestamp=1)
        assert p1.pattern_id == p2.pattern_id

    def test_observe_returns_surprise(self):
        stream = AffectStream(bases_per_dim=6)
        # First observation: creates new pattern → surprise = 0.0 (exact match to self)
        _, surprise1 = stream.observe(AffectState(0.5, 0.5), timestamp=0)
        assert surprise1 == 0.0  # New pattern = exact match

    def test_current_state_tracks(self):
        stream = AffectStream()
        stream.observe(AffectState(0.3, 0.7), timestamp=0)
        assert stream.current_state.valence == pytest.approx(0.3)
        assert stream.current_state.arousal == pytest.approx(0.7)

    def test_pattern_count_increases(self):
        stream = AffectStream(bases_per_dim=6, similarity_threshold=0.7)
        assert stream.pattern_count == 0
        stream.observe(AffectState(-0.9, 0.9), timestamp=0)
        assert stream.pattern_count == 1
        stream.observe(AffectState(0.9, 0.1), timestamp=1)
        assert stream.pattern_count == 2

    def test_tick_advances_clock(self):
        stream = AffectStream()
        stream.observe(AffectState(0.0, 0.0), timestamp=0)
        tick_before = stream._distinction._current_tick
        stream.tick()
        assert stream._distinction._current_tick == tick_before + 1


class TestAgentWithAffect:
    def test_agent_with_affect_runs(self):
        """Agent with affect enabled completes a survival episode."""
        from fpi.agent.core import Agent
        from fpi.env.base import SurvivalEnv

        agent = Agent(similarity_threshold=0.7, seed=42, enable_affect=True)
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])

        results = agent.run_survival_episode(env, max_steps=50)
        assert len(results) > 0
        assert agent._affect is not None
        assert agent._affect.pattern_count > 0

    def test_agent_without_affect_unchanged(self):
        """Agent without affect is backward compatible."""
        from fpi.agent.core import Agent

        agent = Agent(similarity_threshold=0.7, seed=42)
        assert agent._affect is None
