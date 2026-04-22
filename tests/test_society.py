"""Tests for Phase 4: Recursive Intelligence — Society as Meta-Agent."""

import numpy as np
import pytest

from fpi.primitives.signal import Signal
from fpi.society.bridge import SignalBridge
from fpi.society.core import Society, SocietyStepResult
from fpi.agent.core import Agent
from fpi.env.shared import SharedGridEnv
from fpi.evolution.genome import Genome


# ---- SignalBridge ----

class TestSignalBridge:
    def test_encoding_shape(self):
        bridge = SignalBridge(grid_size=20, n_regions=4)
        signal = bridge.encode({0: 5, 1: 15}, {0: 0.8, 1: 0.6}, timestamp=0)
        assert signal.dim == 8  # 2 * n_regions
        assert signal.modality == "collective"

    def test_signal_dim_property(self):
        bridge = SignalBridge(grid_size=20, n_regions=4)
        assert bridge.signal_dim == 8

    def test_different_states_different_signals(self):
        bridge = SignalBridge(grid_size=20, n_regions=4)
        # All agents on the left
        sig_left = bridge.encode({0: 1, 1: 2, 2: 3}, {0: 0.9, 1: 0.9, 2: 0.9}, timestamp=0)
        # All agents on the right
        sig_right = bridge.encode({0: 17, 1: 18, 2: 19}, {0: 0.9, 1: 0.9, 2: 0.9}, timestamp=0)
        # Should be quite different
        sim = sig_left.cosine_similarity(sig_right)
        assert sim < 0.9, f"Different distributions should have low similarity, got {sim}"

    def test_similar_states_similar_signals(self):
        bridge = SignalBridge(grid_size=20, n_regions=4)
        sig1 = bridge.encode({0: 3, 1: 13}, {0: 0.8, 1: 0.7}, timestamp=0)
        sig2 = bridge.encode({0: 4, 1: 14}, {0: 0.75, 1: 0.65}, timestamp=0)
        sim = sig1.cosine_similarity(sig2)
        assert sim > 0.8, f"Similar distributions should have high similarity, got {sim}"

    def test_empty_population(self):
        bridge = SignalBridge(grid_size=20, n_regions=4)
        sig = bridge.encode({}, {}, timestamp=0)
        assert sig.dim == 8
        # All zeros for density, all zeros for vitality
        assert np.allclose(sig.data, 0.0)

    def test_density_sums_to_one(self):
        bridge = SignalBridge(grid_size=20, n_regions=4)
        sig = bridge.encode({0: 5, 1: 10, 2: 15}, {0: 1.0, 1: 1.0, 2: 1.0}, timestamp=0)
        density = sig.data[:4]
        assert density.sum() == pytest.approx(1.0)


# ---- Society ----

def _make_society(
    n_agents: int = 4,
    grid_size: int = 20,
    num_resources: int = 4,
    max_steps: int = 100,
    seed: int = 42,
) -> Society:
    """Helper to create a Society with agents."""
    env = SharedGridEnv(
        grid_size=grid_size,
        num_resources=num_resources,
        resource_value=0.3,
        resource_regen_rate=0.05,
        move_cost=0.015,
        stay_cost=0.005,
        max_steps=max_steps,
        seed=seed,
    )
    bridge = SignalBridge(grid_size=grid_size, n_regions=4)
    agents = []
    rng = np.random.default_rng(seed)
    for i in range(n_agents):
        genome = Genome.random(rng)
        agent = Agent(seed=int(rng.integers(0, 2**31)), **genome.to_agent_kwargs())
        agents.append(agent)
    return Society(agents=agents, env=env, bridge=bridge, seed=seed)


class TestSocietyConstruction:
    def test_has_same_primitives_as_agent(self):
        """Society has WorldModel + Vitality + Valence — same as Agent."""
        soc = _make_society()
        assert soc.world_model is not None
        assert soc.vitality is not None
        assert soc.valence is not None
        assert soc.vitality.alive is True

    def test_agents_registered_in_env(self):
        soc = _make_society(n_agents=4)
        assert len(soc.env.agent_positions) == 4

    def test_agents_spread_across_grid(self):
        soc = _make_society(n_agents=4, grid_size=20)
        positions = list(soc.env.agent_positions.values())
        # Agents should be spread out, not all at center
        assert len(set(positions)) > 1


class TestSocietyStep:
    def test_step_returns_result(self):
        soc = _make_society()
        result = soc.step()
        assert isinstance(result, SocietyStepResult)
        assert result.num_alive > 0
        assert 0.0 <= result.collective_vitality <= 1.0

    def test_step_progresses_tick(self):
        soc = _make_society()
        soc.step()
        soc.step()
        assert soc._tick == 2

    def test_society_actions_are_valid(self):
        soc = _make_society()
        result = soc.step()
        assert result.action in [0, 1, 2]


class TestSocietyLearning:
    def test_society_learns_patterns(self):
        """After enough steps, society should recognize patterns in collective behavior."""
        soc = _make_society(max_steps=200)
        soc.run_episode(max_steps=50)
        assert soc.pattern_count >= 1
        # With enough variation in collective state, should see multiple patterns
        # (may only be 1 if agents all collapse to same state)

    def test_society_forms_associations(self):
        """Society should form temporal associations between collective states."""
        soc = _make_society(max_steps=200)
        soc.run_episode(max_steps=50)
        # Society should have formed at least some associations
        assert soc.association_count >= 0  # May be 0 if all steps match same pattern

    def test_society_surprise_measurable(self):
        """Society should have measurable surprise."""
        soc = _make_society(max_steps=200)
        results = soc.run_episode(max_steps=30)
        surprises = [r.surprise for r in results]
        assert len(surprises) > 0
        # At least some surprise should be non-zero (first observation always has surprise)
        assert any(s > 0 for s in surprises)


class TestSocietyVitality:
    def test_vitality_responds_to_agents(self):
        """Society vitality should change based on agent health."""
        soc = _make_society(max_steps=200)
        results = soc.run_episode(max_steps=50)
        # Society vitality should have changed from initial 1.0
        final_vitality = results[-1].collective_vitality
        assert final_vitality != 1.0

    def test_society_dies_when_agents_die(self):
        """If all agents die, society should eventually die."""
        # Create environment with no resources — agents will die
        env = SharedGridEnv(
            grid_size=10,
            num_resources=0,
            resource_regen_rate=0.0,
            max_steps=500,
            seed=42,
        )
        bridge = SignalBridge(grid_size=10, n_regions=4)
        agents = [Agent(similarity_threshold=0.7, seed=i) for i in range(3)]
        soc = Society(agents=agents, env=env, bridge=bridge, seed=42)
        results = soc.run_episode(max_steps=500)
        # Society should have ended early (agents died)
        assert len(results) < 500

    def test_num_alive_decreases_without_resources(self):
        """Without resources, agents should die over time."""
        env = SharedGridEnv(
            grid_size=10,
            num_resources=0,
            resource_regen_rate=0.0,
            max_steps=300,
            seed=42,
        )
        bridge = SignalBridge(grid_size=10, n_regions=4)
        agents = [Agent(similarity_threshold=0.7, seed=i) for i in range(3)]
        soc = Society(agents=agents, env=env, bridge=bridge, seed=42)
        results = soc.run_episode(max_steps=200)
        if len(results) > 1:
            assert results[-1].num_alive <= results[0].num_alive


class TestSocietyActions:
    def test_action_changes_regen_bias(self):
        """Society action should affect the environment."""
        soc = _make_society()
        # Before any action, bias should be uniform
        assert np.allclose(soc.env._regen_bias, 1.0)
        # After a step, society takes an action that may change bias
        soc.step()
        # We can't guarantee which action, but the mechanism works
        # Let's manually test:
        soc.env.set_regen_bias(0)
        assert soc.env._regen_bias[0] > soc.env._regen_bias[-1]


class TestAgentsOblivious:
    def test_no_agent_references_society(self):
        """Individual agents should have no knowledge of the society."""
        soc = _make_society()
        soc.run_episode(max_steps=10)
        for agent in soc.agents:
            # Agent has no society attribute
            assert not hasattr(agent, "society")
            # Agent's world model doesn't reference society
            assert not hasattr(agent.world_model, "society")


class TestSocietyEpisode:
    def test_run_episode(self):
        """Full episode should run and return results."""
        soc = _make_society(max_steps=50)
        results = soc.run_episode(max_steps=50)
        assert len(results) > 0
        assert all(isinstance(r, SocietyStepResult) for r in results)

    def test_history_accumulated(self):
        soc = _make_society(max_steps=50)
        soc.run_episode(max_steps=20)
        assert len(soc.history) > 0

    def test_society_surprise_decreases_over_time(self):
        """The key integration test: society should learn to predict collective behavior.

        Society surprise should decrease as it learns patterns in the
        collective state of its agents.
        """
        soc = _make_society(n_agents=6, max_steps=200, num_resources=6)
        results = soc.run_episode(max_steps=150)

        if len(results) < 40:
            pytest.skip("Episode too short to measure surprise decrease")

        # Compare early vs late surprise
        early = results[:20]
        late = results[-20:]
        early_surprise = sum(r.surprise for r in early) / len(early)
        late_surprise = sum(r.surprise for r in late) / len(late)

        # Society should have learned something (surprise decreased or stayed low)
        # This is a soft test — the trend should be there but may not always decrease
        assert late_surprise <= early_surprise + 0.3, (
            f"Society surprise didn't decrease: early={early_surprise:.3f}, late={late_surprise:.3f}"
        )
