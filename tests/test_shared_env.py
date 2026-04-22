"""Tests for SharedGridEnv — multi-agent environment with finite resources."""

import numpy as np
import pytest

from fpi.env.shared import SharedGridEnv


class TestSharedGridEnv:
    def test_register_agent(self):
        env = SharedGridEnv(grid_size=10)
        obs = env.register_agent(0)
        assert obs.dim == SharedGridEnv.NUM_POSITION_BASES
        assert obs.modality == "env"
        assert 0 in env.agent_positions

    def test_register_agent_at_position(self):
        env = SharedGridEnv(grid_size=10)
        env.register_agent(0, position=3)
        assert env.agent_positions[0] == 3

    def test_step_agent_moves(self):
        env = SharedGridEnv(grid_size=10, num_resources=0)
        env.register_agent(0, position=5)

        obs, delta, done = env.step_agent(0, 2)  # move right
        assert env.agent_positions[0] == 6
        assert delta == pytest.approx(-0.015)
        assert obs.dim == SharedGridEnv.NUM_POSITION_BASES

    def test_staying_costs_less(self):
        env = SharedGridEnv(grid_size=10, num_resources=0, move_cost=0.02, stay_cost=0.005)
        env.register_agent(0, position=5)
        _, move_delta, _ = env.step_agent(0, 0)  # move left
        env._agent_positions[0] = 5  # reset position
        _, stay_delta, _ = env.step_agent(0, 1)  # stay
        assert abs(stay_delta) < abs(move_delta)

    def test_resource_depletion(self):
        env = SharedGridEnv(grid_size=10, num_resources=0, resource_value=0.5, resource_regen_rate=0.0)
        env._resources.add(5)
        env.register_agent(0, position=4)

        # Move right onto resource
        _, delta, _ = env.step_agent(0, 2)
        assert delta > 0  # Got the resource
        assert 5 not in env.resources  # Resource depleted

    def test_resource_regeneration(self):
        env = SharedGridEnv(grid_size=10, num_resources=0, resource_regen_rate=1.0, seed=42)
        assert len(env.resources) == 0
        env.tick()
        assert len(env.resources) > 0  # Resources regenerated

    def test_multiple_agents(self):
        env = SharedGridEnv(grid_size=10)
        obs0 = env.register_agent(0, position=2)
        obs1 = env.register_agent(1, position=8)
        assert len(env.agent_positions) == 2
        assert obs0.dim == obs1.dim

    def test_resource_competition(self):
        """First agent to step on a resource gets it."""
        env = SharedGridEnv(grid_size=10, num_resources=0, resource_value=0.5, resource_regen_rate=0.0)
        env._resources.add(5)
        env.register_agent(0, position=4)
        env.register_agent(1, position=6)

        # Agent 0 moves right onto resource first
        _, delta0, _ = env.step_agent(0, 2)
        # Agent 1 moves left onto same position — resource already gone
        _, delta1, _ = env.step_agent(1, 0)

        assert delta0 > 0  # Got the resource
        assert delta1 < 0  # Only paid move cost

    def test_bounds(self):
        env = SharedGridEnv(grid_size=5)
        env.register_agent(0, position=0)
        env.step_agent(0, 0)  # Try to move left at position 0
        assert env.agent_positions[0] == 0

        env._agent_positions[0] = 4
        env.step_agent(0, 2)  # Try to move right at last position
        assert env.agent_positions[0] == 4

    def test_regen_bias(self):
        """Society action shifts regeneration distribution."""
        env = SharedGridEnv(grid_size=20, resource_regen_rate=0.1, seed=42)

        # Bias left — count total regeneration events across many rounds
        env.set_regen_bias(0)
        left_total = 0
        right_total = 0
        for _ in range(100):
            env._resources.clear()
            env._regenerate_resources()
            left_total += sum(1 for r in env.resources if r < 10)
            right_total += sum(1 for r in env.resources if r >= 10)

        assert left_total > right_total, (
            f"Left bias should produce more left resources: left={left_total}, right={right_total}"
        )

    def test_done_after_max_steps(self):
        env = SharedGridEnv(grid_size=10, max_steps=3)
        env.register_agent(0)
        for _ in range(3):
            env.tick()
        _, _, done = env.step_agent(0, 1)
        assert done is True

    def test_reset(self):
        env = SharedGridEnv(grid_size=10)
        env.register_agent(0, position=3)
        env.tick()
        env.reset()
        assert len(env.agent_positions) == 0
        assert env.step_count == 0
        assert len(env.resources) > 0  # Resources re-initialized

    def test_observation_format_matches_survival_env(self):
        """SharedGridEnv observations have same format as SurvivalEnv."""
        env = SharedGridEnv(grid_size=10)
        obs = env.register_agent(0)
        assert obs.dim == 6  # NUM_POSITION_BASES
        assert obs.modality == "env"

    def test_unregistered_agent_raises(self):
        env = SharedGridEnv(grid_size=10)
        with pytest.raises(ValueError):
            env.step_agent(99, 1)

    def test_initial_resources_placed(self):
        env = SharedGridEnv(grid_size=20, num_resources=4)
        assert len(env.resources) == 4


class TestResourceClustering:
    def test_default_no_clustering(self):
        """Default cluster_prob is 0.0."""
        env = SharedGridEnv(grid_size=10)
        assert env._resource_cluster_prob == 0.0

    def test_clustering_produces_more_resources(self):
        """Clustering produces more resources than base rate alone."""
        total_clustered = 0
        total_plain = 0
        for trial in range(50):
            env_c = SharedGridEnv(
                grid_size=20, num_resources=0,
                resource_regen_rate=0.1, resource_cluster_prob=1.0, seed=trial,
            )
            env_c._regenerate_resources()
            total_clustered += len(env_c.resources)

            env_p = SharedGridEnv(
                grid_size=20, num_resources=0,
                resource_regen_rate=0.1, resource_cluster_prob=0.0, seed=trial,
            )
            env_p._regenerate_resources()
            total_plain += len(env_p.resources)
        assert total_clustered > total_plain

    def test_cluster_no_cascade(self):
        """Only seed resources expand, not their cluster neighbors."""
        env = SharedGridEnv(
            grid_size=20, num_resources=0,
            resource_regen_rate=0.0, resource_cluster_prob=1.0, seed=42,
        )
        env._regenerate_resources()
        assert len(env.resources) == 0  # no seeds → no clusters

    def test_cluster_respects_grid_bounds(self):
        """Cluster at boundary doesn't go out of range."""
        env = SharedGridEnv(
            grid_size=5, num_resources=0,
            resource_regen_rate=1.0, resource_cluster_prob=1.0, seed=42,
        )
        env._regenerate_resources()
        for pos in env.resources:
            assert 0 <= pos < 5
