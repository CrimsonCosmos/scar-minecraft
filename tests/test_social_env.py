"""Tests for SocialGridEnv — leaky embodiment."""

import numpy as np
import pytest

from fpi.env.social import SocialGridEnv
from fpi.agent.core import Agent


class TestConstruction:
    """SocialGridEnv produces correct observation shape and modality."""

    def test_observation_dim_is_21(self):
        env = SocialGridEnv(grid_size=20)
        env.register_agent(0)
        obs = env._make_social_observation(0)
        assert obs.data.shape[0] == 21

    def test_observation_dim_property(self):
        env = SocialGridEnv(grid_size=20)
        assert env.observation_dim == 21  # 6 + 4 + 4 + 3 + 4

    def test_modality_is_env(self):
        env = SocialGridEnv(grid_size=20)
        obs = env.register_agent(0)
        assert obs.modality == "env"

    def test_register_returns_21_dim(self):
        env = SocialGridEnv(grid_size=20)
        obs = env.register_agent(0)
        assert obs.data.shape[0] == 21


class TestPerception:
    """Agents perceive nearby others but not distant ones."""

    def test_alone_produces_zeros_in_social_dims(self):
        env = SocialGridEnv(grid_size=20, perception_radius=5)
        env.register_agent(0, position=0)
        obs = env._make_social_observation(0)
        # Social dims [6:21] should be all zeros when alone
        social_part = obs.data[6:]
        assert np.all(social_part == 0.0)

    def test_nearby_agent_produces_nonzero_social(self):
        env = SocialGridEnv(grid_size=20, perception_radius=5)
        env.register_agent(0, position=5)
        env.register_agent(1, position=7)  # distance 2, within radius 5
        obs = env._make_social_observation(0)
        social_part = obs.data[6:]
        assert np.any(social_part > 0.0)

    def test_distant_agent_not_perceived(self):
        env = SocialGridEnv(grid_size=20, perception_radius=3)
        env.register_agent(0, position=0)
        env.register_agent(1, position=15)  # distance 15, outside radius 3
        obs = env._make_social_observation(0)
        social_part = obs.data[6:]
        assert np.all(social_part == 0.0)

    def test_nearest_selected_when_multiple(self):
        env = SocialGridEnv(grid_size=20, perception_radius=10)
        env.register_agent(0, position=10)
        env.register_agent(1, position=8)   # distance 2
        env.register_agent(2, position=15)  # distance 5
        nearest = env._find_nearest_other(0)
        assert nearest == 1

    def test_perception_radius_boundary(self):
        """Agent at exactly perception_radius is still perceived."""
        env = SocialGridEnv(grid_size=20, perception_radius=5)
        env.register_agent(0, position=5)
        env.register_agent(1, position=10)  # distance = 5 = radius
        nearest = env._find_nearest_other(0)
        assert nearest == 1


class TestLeaking:
    """Leaked state appears in observation correctly."""

    def test_default_leaked_state(self):
        """Before update_agent_state, defaults are used."""
        env = SocialGridEnv(grid_size=20, perception_radius=10)
        env.register_agent(0, position=5)
        env.register_agent(1, position=7)
        # Default: vitality=0.5, surprise=0.5, action=stay(1)
        state = env._leaked_state[1]
        assert state == (0.5, 0.5, 1)

    def test_update_agent_state_changes_leaked(self):
        env = SocialGridEnv(grid_size=20, perception_radius=10)
        env.register_agent(0, position=5)
        env.register_agent(1, position=7)
        env.update_agent_state(1, vitality=0.9, surprise=0.1)
        v, s, a = env._leaked_state[1]
        assert v == pytest.approx(0.9)
        assert s == pytest.approx(0.1)

    def test_high_vs_low_vitality_different_signal(self):
        """Nearby agent with high vs low vitality produces different obs."""
        env = SocialGridEnv(grid_size=20, perception_radius=10)
        env.register_agent(0, position=5)
        env.register_agent(1, position=7)

        env.update_agent_state(1, vitality=0.9, surprise=0.5)
        obs_high = env._make_social_observation(0)

        env.update_agent_state(1, vitality=0.1, surprise=0.5)
        obs_low = env._make_social_observation(0)

        # Vitality dims [6:10] should differ
        assert not np.allclose(obs_high.data[6:10], obs_low.data[6:10])
        # Surprise dims [10:14] should be the same (both 0.5)
        assert np.allclose(obs_high.data[10:14], obs_low.data[10:14])

    def test_direction_encoding(self):
        """Left/stay/right actions produce different direction encodings."""
        env = SocialGridEnv(grid_size=20, perception_radius=10)
        env.register_agent(0, position=5)
        env.register_agent(1, position=7)

        directions = {}
        for action in [0, 1, 2]:
            env._leaked_state[1] = (0.5, 0.5, action)
            obs = env._make_social_observation(0)
            directions[action] = obs.data[14:17].copy()

        # All three should be distinct
        assert not np.allclose(directions[0], directions[1])
        assert not np.allclose(directions[1], directions[2])
        assert not np.allclose(directions[0], directions[2])

    def test_step_agent_records_action(self):
        """step_agent should record the action in leaked state."""
        env = SocialGridEnv(grid_size=20, perception_radius=10)
        env.register_agent(0, position=10)
        env.step_agent(0, action=2)  # move right
        _v, _s, recorded_action = env._leaked_state[0]
        assert recorded_action == 2


class TestPatternQuality:
    """Social observations have good properties for pattern matching."""

    def test_all_values_in_01(self):
        """Gaussian basis values should be in [0, 1]."""
        env = SocialGridEnv(grid_size=20, perception_radius=10)
        env.register_agent(0, position=5)
        env.register_agent(1, position=7)
        env.update_agent_state(1, vitality=0.8, surprise=0.3)
        obs = env._make_social_observation(0)
        assert np.all(obs.data >= 0.0)
        assert np.all(obs.data <= 1.0)

    def test_alone_vs_accompanied_different(self):
        """Alone and accompanied observations differ in social dims."""
        env = SocialGridEnv(grid_size=20, perception_radius=10)
        env.register_agent(0, position=5)

        obs_alone = env._make_social_observation(0)

        env.register_agent(1, position=7)
        obs_with = env._make_social_observation(0)

        # Position dims should be the same
        assert np.allclose(obs_alone.data[:6], obs_with.data[:6])
        # Social dims should differ
        assert not np.allclose(obs_alone.data[6:], obs_with.data[6:])

    def test_different_distances_distinguishable(self):
        """Distance encoding should differ for near vs far neighbors."""
        env = SocialGridEnv(grid_size=20, perception_radius=10)
        env.register_agent(0, position=10)
        env.register_agent(1, position=11)  # distance 1

        obs_near = env._make_social_observation(0)

        env._agent_positions[1] = 18  # distance 8
        obs_far = env._make_social_observation(0)

        # Distance dims [17:21] should differ
        assert not np.allclose(obs_near.data[17:21], obs_far.data[17:21])


class TestSelfEmission:
    """Proprioceptive self-emission (include_self_emission=True)."""

    def test_disabled_by_default(self):
        env = SocialGridEnv(grid_size=20)
        assert not env.include_self_emission
        assert env.observation_dim == 21

    def test_obs_dim_32_when_enabled(self):
        env = SocialGridEnv(grid_size=20, include_self_emission=True)
        assert env.include_self_emission
        assert env.observation_dim == 32

    def test_register_returns_32_dim(self):
        env = SocialGridEnv(grid_size=20, include_self_emission=True)
        obs = env.register_agent(0, position=5)
        assert obs.data.shape[0] == 32

    def test_step_returns_32_dim(self):
        env = SocialGridEnv(grid_size=20, include_self_emission=True)
        env.register_agent(0, position=10)
        obs, delta, done = env.step_agent(0, 1)
        assert obs.data.shape[0] == 32

    def test_self_vitality_appears_in_dims_21_25(self):
        env = SocialGridEnv(grid_size=20, include_self_emission=True, perception_radius=10)
        env.register_agent(0, position=5)
        env.register_agent(1, position=7)

        env.update_agent_state(0, vitality=0.9, surprise=0.5)
        obs_high = env._make_social_observation(0)

        env.update_agent_state(0, vitality=0.1, surprise=0.5)
        obs_low = env._make_social_observation(0)

        # Self vitality dims [21:25] should differ
        assert not np.allclose(obs_high.data[21:25], obs_low.data[21:25])
        # Self surprise dims [25:29] should be the same
        assert np.allclose(obs_high.data[25:29], obs_low.data[25:29])

    def test_self_surprise_appears_in_dims_25_29(self):
        env = SocialGridEnv(grid_size=20, include_self_emission=True, perception_radius=10)
        env.register_agent(0, position=5)

        env.update_agent_state(0, vitality=0.5, surprise=0.9)
        obs_high = env._make_social_observation(0)

        env.update_agent_state(0, vitality=0.5, surprise=0.1)
        obs_low = env._make_social_observation(0)

        # Self surprise dims [25:29] should differ
        assert not np.allclose(obs_high.data[25:29], obs_low.data[25:29])

    def test_self_direction_appears_in_dims_29_32(self):
        env = SocialGridEnv(grid_size=20, include_self_emission=True, perception_radius=10)
        env.register_agent(0, position=10)

        directions = {}
        for action in [0, 1, 2]:
            env._leaked_state[0] = (0.5, 0.5, action)
            obs = env._make_social_observation(0)
            directions[action] = obs.data[29:32].copy()

        assert not np.allclose(directions[0], directions[1])
        assert not np.allclose(directions[1], directions[2])
        assert not np.allclose(directions[0], directions[2])

    def test_self_encoding_is_centered_version_of_other(self):
        """Self-emission is the mean-centered form of the same Gaussian encoding."""
        env = SocialGridEnv(grid_size=20, include_self_emission=True, perception_radius=10)
        env.register_agent(0, position=5)
        env.register_agent(1, position=7)

        # Set both agents to same vitality and surprise
        env.update_agent_state(0, vitality=0.8, surprise=0.3)
        env.update_agent_state(1, vitality=0.8, surprise=0.3)
        obs = env._make_social_observation(0)

        # Other's vitality [6:10] is uncentered; self vitality [21:25] is centered
        other_vit = obs.data[6:10]
        self_vit = obs.data[21:25]
        assert np.allclose(self_vit, other_vit - np.mean(other_vit))

        # Other's surprise [10:14] is uncentered; self surprise [25:29] is centered
        other_surp = obs.data[10:14]
        self_surp = obs.data[25:29]
        assert np.allclose(self_surp, other_surp - np.mean(other_surp))

    def test_value_ranges(self):
        """Non-self dims in [0,1]; self-emission dims are mean-centered."""
        env = SocialGridEnv(grid_size=20, include_self_emission=True, perception_radius=10)
        env.register_agent(0, position=5)
        env.register_agent(1, position=7)
        env.update_agent_state(0, vitality=0.8, surprise=0.3)
        obs = env._make_social_observation(0)

        # Position + social dims [0:21]: standard Gaussian, in [0, 1]
        assert np.all(obs.data[:21] >= 0.0)
        assert np.all(obs.data[:21] <= 1.0)

        # Self-emission dims [21:32]: mean-centered, in [-1, 1]
        assert np.all(obs.data[21:] >= -1.0)
        assert np.all(obs.data[21:] <= 1.0)

        # Verify centering: each self-emission group has mean ~0
        assert abs(np.mean(obs.data[21:25])) < 1e-10  # vitality group
        assert abs(np.mean(obs.data[25:29])) < 1e-10  # surprise group
        assert abs(np.mean(obs.data[29:32])) < 1e-10  # direction group

    def test_alone_social_zeros_but_self_nonzero(self):
        """When alone, social dims [6:21] are zeros but self-emission [21:32] nonzero."""
        env = SocialGridEnv(grid_size=20, include_self_emission=True, perception_radius=5)
        env.register_agent(0, position=5)
        obs = env._make_social_observation(0)

        # Social dims should be zeros (alone)
        assert np.all(obs.data[6:21] == 0.0)
        # Self-emission dims should be nonzero (centered, but not all zero)
        assert not np.allclose(obs.data[21:32], 0.0)

    def test_self_emission_uses_one_tick_delay(self):
        """Self-emission reads from _leaked_state, which has one-tick delay."""
        env = SocialGridEnv(grid_size=20, include_self_emission=True, perception_radius=10)
        env.register_agent(0, position=10)

        # Before any update, default leaked state (0.5, 0.5, stay)
        obs_before = env._make_social_observation(0)

        # Update state (simulating SocialSociety callback)
        env.update_agent_state(0, vitality=0.9, surprise=0.1)
        obs_after = env._make_social_observation(0)

        # Self-emission should change after update
        assert not np.allclose(obs_before.data[21:32], obs_after.data[21:32])

    def test_agent_processes_32_dim_obs(self):
        """Agent can process 32-dim observations without error."""
        env = SocialGridEnv(grid_size=10, include_self_emission=True, perception_radius=5)
        agent = Agent(similarity_threshold=0.7)

        obs = env.register_agent(0, position=5)
        env.register_agent(1, position=7)

        result = agent.step_with_action(obs, 0.0, None)
        assert result.surprise >= 0.0

        # Run a few more steps
        for _ in range(10):
            action = agent.select_action(env.action_space)
            obs, delta, _done = env.step_agent(0, action)
            agent.step_with_action(obs, delta, action)

        assert agent.world_model.memory.pattern_count > 0


class TestIntegration:
    """SocialGridEnv works with existing Agent (no crashes)."""

    def test_agent_processes_social_obs(self):
        """Agent can process 21-dim social observations without error."""
        env = SocialGridEnv(grid_size=10, perception_radius=5)
        agent = Agent(similarity_threshold=0.7)

        obs = env.register_agent(0, position=5)
        env.register_agent(1, position=7)

        # Agent should handle 21-dim signal fine
        result = agent.step_with_action(obs, 0.0, None)
        assert result.surprise >= 0.0

    def test_agent_forms_patterns_from_social_obs(self):
        """Agent forms patterns from 21-dim social observations."""
        env = SocialGridEnv(grid_size=10, perception_radius=5)
        agent = Agent(similarity_threshold=0.7)

        env.register_agent(0, position=5)
        env.register_agent(1, position=7)

        # Run several steps
        for _ in range(20):
            action = agent.select_action(env.action_space)
            obs, delta, _done = env.step_agent(0, action)
            agent.step_with_action(obs, delta, action)

        assert agent.world_model.memory.pattern_count > 0

    def test_step_returns_correct_shape(self):
        """step_agent returns 21-dim observation."""
        env = SocialGridEnv(grid_size=20, perception_radius=5)
        env.register_agent(0, position=10)
        obs, delta, done = env.step_agent(0, 1)  # stay
        assert obs.data.shape[0] == 21

    def test_reset_clears_leaked_state(self):
        """reset() clears leaked state and agent positions."""
        env = SocialGridEnv(grid_size=20, perception_radius=5)
        env.register_agent(0, position=5)
        env.update_agent_state(0, vitality=0.9, surprise=0.1)
        env.reset()
        assert len(env._leaked_state) == 0
        assert len(env._agent_positions) == 0
