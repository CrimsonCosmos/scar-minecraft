"""Tests for SharedGrid2DEnv and SocialGrid2DEnv."""

import numpy as np
import pytest

from fpi.env.shared_2d import SharedGrid2DEnv, ACTION_DELTAS
from fpi.env.social_2d import SocialGrid2DEnv


class TestSharedGrid2DEnv:
    def test_register_agent(self):
        env = SharedGrid2DEnv(grid_size=10)
        obs = env.register_agent(0)
        assert obs.dim == SharedGrid2DEnv.NUM_POSITION_BASES * 2  # row + col
        assert obs.modality == "env"
        assert 0 in env.agent_positions

    def test_register_agent_at_position(self):
        env = SharedGrid2DEnv(grid_size=10)
        env.register_agent(0, position=(3, 7))
        assert env.agent_positions[0] == (3, 7)

    def test_default_center_position(self):
        env = SharedGrid2DEnv(grid_size=10)
        env.register_agent(0)
        assert env.agent_positions[0] == (5, 5)

    def test_step_moves_up(self):
        env = SharedGrid2DEnv(grid_size=10, num_resources=0)
        env.register_agent(0, position=(5, 5))
        obs, delta, done = env.step_agent(0, 0)  # up
        assert env.agent_positions[0] == (4, 5)
        assert delta == pytest.approx(-0.015)

    def test_step_moves_down(self):
        env = SharedGrid2DEnv(grid_size=10, num_resources=0)
        env.register_agent(0, position=(5, 5))
        env.step_agent(0, 1)  # down
        assert env.agent_positions[0] == (6, 5)

    def test_step_moves_left(self):
        env = SharedGrid2DEnv(grid_size=10, num_resources=0)
        env.register_agent(0, position=(5, 5))
        env.step_agent(0, 2)  # left
        assert env.agent_positions[0] == (5, 4)

    def test_step_moves_right(self):
        env = SharedGrid2DEnv(grid_size=10, num_resources=0)
        env.register_agent(0, position=(5, 5))
        env.step_agent(0, 3)  # right
        assert env.agent_positions[0] == (5, 6)

    def test_stay_costs_less(self):
        env = SharedGrid2DEnv(grid_size=10, num_resources=0, move_cost=0.02, stay_cost=0.005)
        env.register_agent(0, position=(5, 5))
        _, move_delta, _ = env.step_agent(0, 0)  # move up
        env._agent_positions[0] = (5, 5)
        _, stay_delta, _ = env.step_agent(0, 4)  # stay
        assert abs(stay_delta) < abs(move_delta)

    def test_resource_depletion(self):
        env = SharedGrid2DEnv(grid_size=10, num_resources=0, resource_value=0.5, resource_regen_rate=0.0)
        env._resources.add((5, 5))
        env.register_agent(0, position=(4, 5))

        _, delta, _ = env.step_agent(0, 1)  # move down onto (5, 5)
        assert delta > 0
        assert (5, 5) not in env.resources

    def test_resource_regeneration(self):
        env = SharedGrid2DEnv(grid_size=5, num_resources=0, resource_regen_rate=1.0, seed=42)
        assert len(env.resources) == 0
        env.tick()
        assert len(env.resources) > 0

    def test_bounds_top_left(self):
        env = SharedGrid2DEnv(grid_size=5)
        env.register_agent(0, position=(0, 0))
        env.step_agent(0, 0)  # up at row 0
        assert env.agent_positions[0][0] == 0
        env.step_agent(0, 2)  # left at col 0
        assert env.agent_positions[0][1] == 0

    def test_bounds_bottom_right(self):
        env = SharedGrid2DEnv(grid_size=5)
        env.register_agent(0, position=(4, 4))
        env.step_agent(0, 1)  # down at row 4
        assert env.agent_positions[0][0] == 4
        env.step_agent(0, 3)  # right at col 4
        assert env.agent_positions[0][1] == 4

    def test_multiple_agents(self):
        env = SharedGrid2DEnv(grid_size=10)
        obs0 = env.register_agent(0, position=(2, 2))
        obs1 = env.register_agent(1, position=(8, 8))
        assert len(env.agent_positions) == 2
        assert obs0.dim == obs1.dim

    def test_resource_competition(self):
        env = SharedGrid2DEnv(grid_size=10, num_resources=0, resource_value=0.5, resource_regen_rate=0.0)
        env._resources.add((5, 5))
        env.register_agent(0, position=(4, 5))
        env.register_agent(1, position=(6, 5))

        _, delta0, _ = env.step_agent(0, 1)  # down onto (5, 5)
        _, delta1, _ = env.step_agent(1, 0)  # up onto (5, 5), resource gone
        assert delta0 > 0
        assert delta1 < 0

    def test_unregistered_agent_raises(self):
        env = SharedGrid2DEnv(grid_size=10)
        with pytest.raises(ValueError):
            env.step_agent(99, 0)

    def test_done_after_max_steps(self):
        env = SharedGrid2DEnv(grid_size=10, max_steps=3)
        env.register_agent(0)
        for _ in range(3):
            env.tick()
        _, _, done = env.step_agent(0, 4)
        assert done is True

    def test_reset(self):
        env = SharedGrid2DEnv(grid_size=10)
        env.register_agent(0, position=(3, 3))
        env.tick()
        env.reset()
        assert len(env.agent_positions) == 0
        assert env.step_count == 0
        assert len(env.resources) > 0

    def test_action_space(self):
        env = SharedGrid2DEnv(grid_size=10)
        assert env.action_space == [0, 1, 2, 3, 4]

    def test_observation_dim(self):
        env = SharedGrid2DEnv(grid_size=10)
        obs = env.register_agent(0)
        assert obs.dim == 12  # 6 row + 6 col


class TestClustering2D:
    def test_default_no_clustering(self):
        env = SharedGrid2DEnv(grid_size=10)
        assert env._resource_cluster_prob == 0.0

    def test_clustering_produces_more_resources(self):
        total_clustered = 0
        total_plain = 0
        for trial in range(30):
            env_c = SharedGrid2DEnv(
                grid_size=10, num_resources=0,
                resource_regen_rate=0.1, resource_cluster_prob=1.0, seed=trial,
            )
            env_c._regenerate_resources()
            total_clustered += len(env_c.resources)

            env_p = SharedGrid2DEnv(
                grid_size=10, num_resources=0,
                resource_regen_rate=0.1, resource_cluster_prob=0.0, seed=trial,
            )
            env_p._regenerate_resources()
            total_plain += len(env_p.resources)
        assert total_clustered > total_plain

    def test_cluster_respects_grid_bounds(self):
        env = SharedGrid2DEnv(
            grid_size=5, num_resources=0,
            resource_regen_rate=1.0, resource_cluster_prob=1.0, seed=42,
        )
        env._regenerate_resources()
        for r, c in env.resources:
            assert 0 <= r < 5
            assert 0 <= c < 5


class TestSocialGrid2DEnv:
    def test_register_agent(self):
        env = SocialGrid2DEnv(grid_size=10)
        obs = env.register_agent(0)
        assert obs.dim == env.observation_dim
        assert obs.modality == "env"

    def test_observation_dim_without_self_emission(self):
        env = SocialGrid2DEnv(grid_size=10, include_self_emission=False)
        assert env.observation_dim == 30  # 12 pos + 4 vit + 4 surp + 6 dir + 4 dist

    def test_observation_dim_with_self_emission(self):
        env = SocialGrid2DEnv(grid_size=10, include_self_emission=True)
        assert env.observation_dim == 44  # 30 + 14

    def test_step_agent_moves(self):
        env = SocialGrid2DEnv(grid_size=10, num_resources=0)
        env.register_agent(0, position=(5, 5))
        obs, delta, done = env.step_agent(0, 3)  # right
        assert env.agent_positions[0] == (5, 6)
        assert obs.dim == env.observation_dim

    def test_social_dims_zero_when_alone(self):
        env = SocialGrid2DEnv(grid_size=10, include_self_emission=False)
        obs = env.register_agent(0, position=(5, 5))
        # Social dims (indices 12-29) should all be zero when alone
        social_part = obs.data[12:30]
        assert np.allclose(social_part, 0.0)

    def test_social_dims_nonzero_with_neighbor(self):
        env = SocialGrid2DEnv(grid_size=10, perception_radius=5)
        env.register_agent(0, position=(5, 5))
        env.register_agent(1, position=(5, 7))  # 2 cells away
        env.update_agent_state(1, vitality=0.8, surprise=0.3)

        obs, _, _ = env.step_agent(0, 4)  # stay
        social_part = obs.data[12:30]
        assert not np.allclose(social_part, 0.0)

    def test_out_of_radius_not_perceived(self):
        env = SocialGrid2DEnv(grid_size=20, perception_radius=3)
        env.register_agent(0, position=(0, 0))
        env.register_agent(1, position=(10, 10))  # dist=20, way outside radius

        obs, _, _ = env.step_agent(0, 4)
        social_part = obs.data[12:30]
        assert np.allclose(social_part, 0.0)

    def test_self_emission_mean_centered(self):
        env = SocialGrid2DEnv(grid_size=10, include_self_emission=True)
        env.register_agent(0, position=(5, 5))
        env.update_agent_state(0, vitality=0.7, surprise=0.2)

        obs, _, _ = env.step_agent(0, 4)
        # Self-emission: [30:34] vit, [34:38] surp, [38:41] dx, [41:44] dy
        own_vit = obs.data[30:34]
        own_surp = obs.data[34:38]
        # Mean-centered: should sum to ~0
        assert abs(np.sum(own_vit)) < 0.01
        assert abs(np.sum(own_surp)) < 0.01

    def test_resource_depletion(self):
        env = SocialGrid2DEnv(grid_size=10, num_resources=0, resource_value=0.5, resource_regen_rate=0.0)
        env._resources.add((5, 5))
        env.register_agent(0, position=(4, 5))

        _, delta, _ = env.step_agent(0, 1)  # down onto (5, 5)
        assert delta > 0
        assert (5, 5) not in env.resources

    def test_update_agent_state(self):
        env = SocialGrid2DEnv(grid_size=10)
        env.register_agent(0)
        env.update_agent_state(0, vitality=0.3, surprise=0.9)
        assert env._leaked_state[0][0] == 0.3
        assert env._leaked_state[0][1] == 0.9

    def test_reset_clears_leaked_state(self):
        env = SocialGrid2DEnv(grid_size=10)
        env.register_agent(0)
        env.update_agent_state(0, vitality=0.5, surprise=0.5)
        env.reset()
        assert len(env._leaked_state) == 0

    def test_direction_encoding(self):
        env = SocialGrid2DEnv(grid_size=10)
        # up: dr=-1 → 0.0, dc=0 → 0.5
        dx, dy = env._action_to_direction(0)
        assert dx == 0.0
        assert dy == 0.5
        # down: dr=1 → 1.0, dc=0 → 0.5
        dx, dy = env._action_to_direction(1)
        assert dx == 1.0
        assert dy == 0.5
        # right: dr=0 → 0.5, dc=1 → 1.0
        dx, dy = env._action_to_direction(3)
        assert dx == 0.5
        assert dy == 1.0
        # stay: dr=0 → 0.5, dc=0 → 0.5
        dx, dy = env._action_to_direction(4)
        assert dx == 0.5
        assert dy == 0.5

    def test_unregistered_agent_raises(self):
        env = SocialGrid2DEnv(grid_size=10)
        with pytest.raises(ValueError):
            env.step_agent(99, 0)

    def test_works_with_agent(self):
        """SocialGrid2DEnv observations work with Agent's dimension-agnostic WorldModel."""
        from fpi.agent.core import Agent

        env = SocialGrid2DEnv(grid_size=10, include_self_emission=True)
        agent = Agent(similarity_threshold=0.7, seed=42)

        obs = env.register_agent(0, position=(5, 5))
        agent.step_with_action(obs, 0.0, None)

        for _ in range(20):
            action = agent.select_action(env.action_space)
            obs, delta, done = env.step_agent(0, action)
            agent.step_with_action(obs, delta, action)
            env.update_agent_state(0, agent.vitality.energy, agent.world_model.last_surprise)
            env.tick()

        assert len(agent.history) > 0


class TestScrambling2D:
    def test_scramble_social_nonzero(self):
        """Scrambled social dims are non-zero when a neighbor is present."""
        env = SocialGrid2DEnv(
            grid_size=10, perception_radius=5,
            include_self_emission=True, scramble_social=True,
        )
        env.register_agent(0, position=(5, 5))
        env.register_agent(1, position=(5, 7))
        env.update_agent_state(1, vitality=0.8, surprise=0.3)

        obs, _, _ = env.step_agent(0, 4)
        social_part = obs.data[12:30]
        assert not np.allclose(social_part, 0.0)

    def test_scramble_self_emission_nonzero(self):
        """Scrambled self-emission dims are non-zero and mean-centered."""
        env = SocialGrid2DEnv(
            grid_size=10, include_self_emission=True,
            scramble_self_emission=True,
        )
        env.register_agent(0, position=(5, 5))
        env.update_agent_state(0, vitality=0.7, surprise=0.2)

        obs, _, _ = env.step_agent(0, 4)
        own_vit = obs.data[30:34]
        own_surp = obs.data[34:38]
        # Non-zero
        assert not np.allclose(own_vit, 0.0)
        assert not np.allclose(own_surp, 0.0)
        # Mean-centered
        assert abs(np.sum(own_vit)) < 0.01
        assert abs(np.sum(own_surp)) < 0.01

    def test_scramble_preserves_dimensionality(self):
        """All scramble flag combos produce the same observation dim."""
        for scr_soc in (False, True):
            for scr_self in (False, True):
                env = SocialGrid2DEnv(
                    grid_size=10, include_self_emission=True,
                    scramble_social=scr_soc, scramble_self_emission=scr_self,
                )
                obs = env.register_agent(0, position=(5, 5))
                assert obs.dim == 44, f"Expected 44 dims for scr_soc={scr_soc}, scr_self={scr_self}"

    def test_scramble_defaults_false(self):
        """Scramble flags default to False (backward compatible)."""
        env = SocialGrid2DEnv(grid_size=10)
        assert env._scramble_social is False
        assert env._scramble_self_emission is False
