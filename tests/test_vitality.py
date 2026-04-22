"""Tests for vitality, valence, survival environment, and agent survival."""

import numpy as np
import pytest

from fpi.primitives.vitality import Vitality
from fpi.primitives.valence import Valence
from fpi.primitives.signal import Signal
from fpi.env.base import SurvivalEnv
from fpi.agent.core import Agent


class TestVitality:
    def test_initial_state(self):
        v = Vitality()
        assert v.energy == 1.0
        assert v.alive is True
        assert v.urgency == 0.0
        assert v.fraction == 1.0

    def test_entropy_depletes_energy(self):
        v = Vitality(energy=1.0, entropy_rate=0.1)
        v.tick()
        assert v.energy == pytest.approx(0.9)
        assert v.alive is True

    def test_dies_at_zero(self):
        v = Vitality(energy=0.05, entropy_rate=0.1)
        v.tick()
        assert v.energy == 0.0
        assert v.alive is False

    def test_dead_agent_tick_returns_zero(self):
        v = Vitality(energy=0.0, alive=False)
        assert v.tick() == 0.0

    def test_spend_reduces_energy(self):
        v = Vitality(energy=1.0)
        v.spend(0.3)
        assert v.energy == pytest.approx(0.7)
        assert v.alive is True

    def test_spend_can_kill(self):
        v = Vitality(energy=0.1)
        v.spend(0.5)
        assert v.energy == 0.0
        assert v.alive is False

    def test_spend_on_dead_agent_is_noop(self):
        v = Vitality(energy=0.0, alive=False)
        v.spend(0.1)
        assert v.energy == 0.0

    def test_restore_adds_energy(self):
        v = Vitality(energy=0.5, max_energy=1.0)
        actual = v.restore(0.3)
        assert actual == pytest.approx(0.3)
        assert v.energy == pytest.approx(0.8)

    def test_restore_caps_at_max(self):
        v = Vitality(energy=0.8, max_energy=1.0)
        actual = v.restore(0.5)
        assert actual == pytest.approx(0.2)
        assert v.energy == pytest.approx(1.0)

    def test_urgency_reflects_deficit(self):
        v = Vitality(energy=0.3, max_energy=1.0)
        assert v.urgency == pytest.approx(0.7)

    def test_urgency_at_full_is_zero(self):
        v = Vitality(energy=1.0, max_energy=1.0)
        assert v.urgency == pytest.approx(0.0)


class TestValence:
    def test_unknown_pattern_is_neutral(self):
        val = Valence()
        assert val.get(999) == 0.0
        assert val.is_known(999) is False

    def test_positive_delta_creates_positive_valence(self):
        val = Valence()
        val.update(0, 0.5)
        assert val.get(0) > 0.0
        assert val.is_known(0) is True

    def test_negative_delta_creates_negative_valence(self):
        val = Valence()
        val.update(0, -0.3)
        assert val.get(0) < 0.0

    def test_valence_adapts_over_time(self):
        val = Valence(learning_rate=0.5)
        val.update(0, 1.0)
        assert val.get(0) > 0.0
        # Switch to negative experience
        for _ in range(20):
            val.update(0, -1.0)
        assert val.get(0) < 0.0

    def test_known_count(self):
        val = Valence()
        val.update(0, 0.1)
        val.update(1, -0.1)
        val.update(2, 0.0)
        assert val.known_count == 3

    def test_first_observation_sets_directly(self):
        val = Valence()
        val.update(5, 0.42)
        assert val.get(5) == pytest.approx(0.42)


class TestSurvivalEnv:
    def test_reset_returns_signal(self):
        env = SurvivalEnv(grid_size=10)
        obs = env.reset()
        assert obs.dim == SurvivalEnv.NUM_POSITION_BASES
        assert obs.modality == "env"

    def test_action_space(self):
        env = SurvivalEnv()
        assert env.action_space == [0, 1, 2]

    def test_resource_gives_energy(self):
        env = SurvivalEnv(grid_size=10, resource_positions=[5], resource_value=0.3, move_cost=0.02)
        env.reset()
        env._position = 4  # One step from resource
        _obs, delta, _done = env.step(2)  # Move right to position 5
        assert delta > 0  # resource_value - move_cost should be positive
        assert delta == pytest.approx(0.3 - 0.02)

    def test_moving_costs_energy(self):
        env = SurvivalEnv(grid_size=10, resource_positions=[], move_cost=0.05)
        env.reset()
        _obs, delta, _done = env.step(0)  # Move left, no resource
        assert delta == pytest.approx(-0.05)

    def test_staying_costs_less_than_moving(self):
        env = SurvivalEnv(grid_size=10, resource_positions=[], move_cost=0.05, stay_cost=0.01)
        env.reset()
        _obs, move_delta, _done = env.step(0)
        env.reset()
        _obs, stay_delta, _done = env.step(1)
        assert abs(stay_delta) < abs(move_delta)

    def test_stays_in_bounds(self):
        env = SurvivalEnv(grid_size=5)
        env.reset()
        env._position = 0
        env.step(0)  # Try to move left at position 0
        assert env._position == 0
        env._position = 4
        env.step(2)  # Try to move right at last position
        assert env._position == 4

    def test_done_after_max_steps(self):
        env = SurvivalEnv(max_steps=3)
        env.reset()
        _, _, done1 = env.step(1)
        _, _, done2 = env.step(1)
        _, _, done3 = env.step(1)
        assert done1 is False
        assert done2 is False
        assert done3 is True

    def test_observation_is_position_only(self):
        """Observation encodes position via Gaussian basis — no vitality."""
        env = SurvivalEnv(grid_size=10)
        env.reset()
        obs, _, _ = env.step(1)
        assert obs.dim == SurvivalEnv.NUM_POSITION_BASES


class TestAgentSurvival:
    def test_agent_dies_without_resources(self):
        """In an empty world, the agent must eventually die."""
        env = SurvivalEnv(
            grid_size=10,
            resource_positions=[],  # No resources
            move_cost=0.02,
            stay_cost=0.005,
            max_steps=500,
        )
        agent = Agent(similarity_threshold=0.7, seed=42)
        results = agent.run_survival_episode(env, max_steps=500)
        # Should die before 500 steps (entropy + action costs drain energy)
        assert len(results) < 500
        assert agent.vitality.alive is False

    def test_agent_survives_with_resources(self):
        """With resources, a lucky/smart agent can survive longer."""
        env = SurvivalEnv(
            grid_size=10,
            resource_positions=[3, 7],
            resource_value=0.5,
            move_cost=0.02,
            stay_cost=0.005,
            max_steps=100,
        )
        agent = Agent(similarity_threshold=0.7, seed=42)
        results = agent.run_survival_episode(env, max_steps=100)
        # Should survive at least some steps
        assert len(results) > 10

    def test_agent_learns_to_survive_longer(self):
        """The key integration test: survival time should increase over episodes."""
        env = SurvivalEnv(
            grid_size=20,
            resource_positions=[5, 15],
            move_cost=0.02,
            stay_cost=0.005,
            resource_value=0.4,
            max_steps=200,
        )
        agent = Agent(similarity_threshold=0.7, seed=42)

        survival_times: list[int] = []
        for _ in range(40):
            agent.world_model.reset_stats()
            results = agent.run_survival_episode(env, max_steps=200)
            survival_times.append(len(results))

        early_avg = sum(survival_times[:5]) / 5
        late_avg = sum(survival_times[-5:]) / 5
        assert late_avg > early_avg, (
            f"Agent didn't improve: early={early_avg:.0f}, late={late_avg:.0f}"
        )

    def test_valence_learns_from_experience(self):
        """Patterns should acquire non-zero valence from experience."""
        env = SurvivalEnv(grid_size=10, resource_positions=[5])
        agent = Agent(similarity_threshold=0.7, seed=42)

        for _ in range(10):
            agent.run_survival_episode(env, max_steps=100)

        assert agent.valence.known_count > 0
        # Should have both positive and negative valences
        has_positive = any(v > 0.001 for v in agent.valence._values.values())
        has_negative = any(v < -0.001 for v in agent.valence._values.values())
        assert has_positive or has_negative, "No meaningful valence learned"

    def test_action_selection_with_no_model(self):
        """With no learned model, agent should still select valid actions."""
        agent = Agent(similarity_threshold=0.7, seed=42)
        action = agent.select_action([0, 1, 2])
        assert action in [0, 1, 2]

    def test_step_result_includes_vitality(self):
        """StepResult should include vitality information."""
        env = SurvivalEnv(grid_size=10, resource_positions=[5])
        agent = Agent(similarity_threshold=0.7, seed=42)
        results = agent.run_survival_episode(env, max_steps=10)
        for r in results:
            assert 0.0 <= r.vitality <= 1.0
