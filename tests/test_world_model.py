"""Tests for WorldModel and the integration with Agent."""

import numpy as np
import pytest

from fpi.primitives.signal import Signal
from fpi.world_model.model import WorldModel
from fpi.agent.core import Agent
from fpi.env.base import SequencePredictionEnv


class TestWorldModel:
    def test_first_observation_is_full_surprise(self):
        wm = WorldModel()
        s = Signal.from_list([1.0, 0.0, 0.0])
        surprise = wm.observe(s)
        assert surprise == 1.0  # No prediction yet

    def test_surprise_decreases_with_repetition(self):
        wm = WorldModel(similarity_threshold=0.9)
        signals = [
            Signal.from_list([1.0, 0.0, 0.0, 0.0]),
            Signal.from_list([0.0, 1.0, 0.0, 0.0]),
        ]

        # Run through the sequence many times
        surprises_early = []
        surprises_late = []
        for cycle in range(20):
            for s in signals:
                surprise = wm.observe(s)
                if cycle < 3:
                    surprises_early.append(surprise)
                elif cycle >= 17:
                    surprises_late.append(surprise)

        avg_early = sum(surprises_early) / len(surprises_early)
        avg_late = sum(surprises_late) / len(surprises_late)
        assert avg_late < avg_early, f"Late surprise {avg_late} should be < early {avg_early}"

    def test_predict_returns_none_initially(self):
        wm = WorldModel()
        assert wm.predict() is None

    def test_predict_returns_pattern_after_learning(self):
        wm = WorldModel(similarity_threshold=0.9)
        signals = [
            Signal.from_list([1.0, 0.0, 0.0, 0.0]),
            Signal.from_list([0.0, 1.0, 0.0, 0.0]),
        ]
        # Train
        for _ in range(10):
            for s in signals:
                wm.observe(s)

        # After observing signal[0], the model should predict signal[1]
        wm.observe(signals[0])
        prediction = wm.predict()
        assert prediction is not None
        pattern, confidence = prediction
        assert confidence > 0.0

    def test_reset_stats(self):
        wm = WorldModel()
        wm.observe(Signal.from_list([1.0, 0.0]))
        wm.observe(Signal.from_list([0.0, 1.0]))
        assert wm.observation_count == 2
        wm.reset_stats()
        assert wm.observation_count == 0
        assert wm.total_surprise == 0.0


class TestAgentIntegration:
    def test_agent_learns_repeating_sequence(self):
        """The key integration test: an agent in a repeating-sequence
        environment should have decreasing surprise over episodes."""
        env = SequencePredictionEnv(
            sequence=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            num_steps=50,
        )
        agent = Agent(similarity_threshold=0.9)

        episode_surprises = []
        for _ in range(10):
            agent.world_model.reset_stats()
            results = agent.run_episode(env, max_steps=50)
            avg = sum(r.surprise for r in results) / len(results)
            episode_surprises.append(avg)

        # Surprise in last episodes should be lower than first episodes
        early_avg = sum(episode_surprises[:3]) / 3
        late_avg = sum(episode_surprises[-3:]) / 3
        assert late_avg < early_avg, (
            f"Agent didn't learn: early={early_avg:.3f}, late={late_avg:.3f}"
        )

    def test_agent_discovers_patterns(self):
        env = SequencePredictionEnv(
            sequence=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            num_steps=50,
        )
        agent = Agent(similarity_threshold=0.9)
        agent.run_episode(env)
        # Should discover 4 distinct patterns
        assert agent.pattern_count == 4

    def test_agent_forms_associations(self):
        env = SequencePredictionEnv(
            sequence=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            num_steps=50,
        )
        agent = Agent(similarity_threshold=0.9)
        agent.run_episode(env)
        # Should form associations between consecutive patterns
        assert agent.association_count >= 4  # At least 0→1, 1→2, 2→3, 3→0


class TestForwardSimulation:
    """Tests for stateless multi-step prediction (generative model)."""

    def _build_model_with_transitions(self):
        """Build a WorldModel with known action-conditional transitions.

        Creates patterns A, B, C and teaches:
          A --(action 0)--> B (vitality +0.1)
          B --(action 1)--> C (vitality +0.2)
        """
        wm = WorldModel(similarity_threshold=0.7)
        sig_a = Signal(data=np.array([1.0, 0.0, 0.0, 0.0]))
        sig_b = Signal(data=np.array([0.0, 1.0, 0.0, 0.0]))
        sig_c = Signal(data=np.array([0.0, 0.0, 1.0, 0.0]))

        # Observe to create patterns
        wm.observe(sig_a)
        pat_a = wm.current_pattern
        wm.observe(sig_b)
        pat_b = wm.current_pattern
        wm.observe(sig_c)
        pat_c = wm.current_pattern

        # Teach action transitions by recording multiple times
        for _ in range(5):
            wm.current_pattern = pat_a
            wm.prev_pattern = pat_a
            wm.record_action_outcome(0, pat_b, 0.1)
            wm.current_pattern = pat_b
            wm.prev_pattern = pat_b
            wm.record_action_outcome(1, pat_c, 0.2)

        return wm, pat_a, pat_b, pat_c

    def test_predict_from_explicit_pattern(self):
        wm, pat_a, pat_b, _pat_c = self._build_model_with_transitions()
        result = wm.predict_from(pat_a.pattern_id, 0)
        assert result is not None
        assert result[0].pattern_id == pat_b.pattern_id

    def test_predict_vitality_from_explicit_pattern(self):
        wm, pat_a, _pat_b, _pat_c = self._build_model_with_transitions()
        vit = wm.predict_vitality_from(pat_a.pattern_id, 0)
        assert vit is not None
        assert vit > 0.0

    def test_simulate_trajectory(self):
        wm, pat_a, pat_b, pat_c = self._build_model_with_transitions()
        total_vit, pids = wm.simulate_trajectory(pat_a.pattern_id, [0, 1])
        assert pids == [pat_a.pattern_id, pat_b.pattern_id, pat_c.pattern_id]
        assert total_vit > 0.0

    def test_simulate_trajectory_unknown_action(self):
        wm, pat_a, _pat_b, _pat_c = self._build_model_with_transitions()
        # Action 99 has no learned data
        total_vit, pids = wm.simulate_trajectory(pat_a.pattern_id, [99])
        # No prediction -> stays at start, no vitality change
        assert pids == [pat_a.pattern_id, pat_a.pattern_id]
        assert total_vit == 0.0

    def test_simulate_trajectory_stateless(self):
        wm, pat_a, _pat_b, pat_c = self._build_model_with_transitions()
        # Set current_pattern to C
        wm.current_pattern = pat_c
        # Simulate from A — should NOT change current_pattern
        wm.simulate_trajectory(pat_a.pattern_id, [0, 1])
        assert wm.current_pattern.pattern_id == pat_c.pattern_id
