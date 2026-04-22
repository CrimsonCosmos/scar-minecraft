"""Tests for PatchForagingEnv."""

import numpy as np
import pytest

from fpi.env.patch import PatchForagingEnv


class TestPatchForagingEnv:
    def test_register_agent(self):
        env = PatchForagingEnv(num_patches=8)
        obs = env.register_agent(0)
        assert obs.dim == 8  # num_patches bases (blind)
        assert obs.modality == "env"
        assert 0 in env.agent_patches

    def test_register_agent_at_patch(self):
        env = PatchForagingEnv(num_patches=8)
        env.register_agent(0, patch=3)
        assert env.agent_patches[0] == 3

    def test_action_space(self):
        env = PatchForagingEnv(num_patches=8)
        assert env.action_space == list(range(8))

    def test_step_changes_patch(self):
        env = PatchForagingEnv(num_patches=8)
        env.register_agent(0, patch=0)
        obs, delta, done = env.step_agent(0, 5)
        assert env.agent_patches[0] == 5
        assert obs.dim == 8

    def test_visit_cost(self):
        env = PatchForagingEnv(num_patches=8, visit_cost=0.01)
        env.register_agent(0, patch=0)
        env._resources.clear()  # No resources
        _, delta, _ = env.step_agent(0, 3)
        assert delta == pytest.approx(-0.01)

    def test_resource_collection(self):
        env = PatchForagingEnv(num_patches=8, resource_value=0.4, visit_cost=0.01)
        env.register_agent(0, patch=0)
        env._resources = {3}
        _, delta, _ = env.step_agent(0, 3)
        assert delta == pytest.approx(0.4 - 0.01)
        assert 3 not in env.resources

    def test_resource_regeneration(self):
        env = PatchForagingEnv(num_patches=8, rich_prob=1.0, poor_prob=1.0, seed=42)
        env._resources.clear()
        env.tick()
        assert len(env.resources) > 0

    def test_rich_patches_more_resources(self):
        """Rich patches produce more resources than poor ones over many ticks."""
        env = PatchForagingEnv(num_patches=8, num_rich=2, rich_prob=0.5, poor_prob=0.05, seed=42)
        rich_counts = np.zeros(8)
        for _ in range(500):
            env._resources.clear()
            env._regenerate_resources()
            for patch in env.resources:
                rich_counts[patch] += 1

        richness = env.richness
        rich_indices = np.where(richness > 0.1)[0]
        poor_indices = np.where(richness <= 0.1)[0]
        assert np.mean(rich_counts[rich_indices]) > np.mean(rich_counts[poor_indices])

    def test_done_after_max_steps(self):
        env = PatchForagingEnv(num_patches=8, max_steps=3)
        env.register_agent(0)
        for _ in range(3):
            env.tick()
        _, _, done = env.step_agent(0, 0)
        assert done is True

    def test_reset(self):
        env = PatchForagingEnv(num_patches=8)
        env.register_agent(0, patch=3)
        env.tick()
        env.reset()
        assert len(env.agent_patches) == 0
        assert env.step_count == 0

    def test_unregistered_agent_raises(self):
        env = PatchForagingEnv(num_patches=8)
        with pytest.raises(ValueError):
            env.step_agent(99, 0)

    def test_patch_bounds_clamped(self):
        env = PatchForagingEnv(num_patches=8)
        env.register_agent(0, patch=0)
        env.step_agent(0, -1)  # below 0
        assert env.agent_patches[0] == 0
        env.step_agent(0, 100)  # above max
        assert env.agent_patches[0] == 7

    def test_multiple_agents(self):
        env = PatchForagingEnv(num_patches=8)
        env.register_agent(0, patch=2)
        env.register_agent(1, patch=5)
        assert len(env.agent_patches) == 2


class TestPatchSocialObservation:
    def test_blind_dim(self):
        env = PatchForagingEnv(num_patches=8, include_social=False, include_self_emission=False)
        obs = env.register_agent(0)
        assert obs.dim == 8

    def test_social_dim(self):
        env = PatchForagingEnv(num_patches=8, include_social=True, include_self_emission=False)
        obs = env.register_agent(0)
        # 8 own patch + 8 other patch + 4 vit + 4 surp = 24
        assert obs.dim == 24

    def test_proprioceptive_dim(self):
        env = PatchForagingEnv(num_patches=8, include_social=True, include_self_emission=True)
        obs = env.register_agent(0)
        # 24 + 4 own vit + 4 own surp = 32
        assert obs.dim == 32

    def test_social_dims_zero_when_alone(self):
        env = PatchForagingEnv(num_patches=8, include_social=True)
        obs = env.register_agent(0, patch=3)
        # Social dims (indices 8:24) should be zero when alone
        social_part = obs.data[8:24]
        assert np.allclose(social_part, 0.0)

    def test_social_dims_nonzero_with_other(self):
        env = PatchForagingEnv(num_patches=8, include_social=True)
        env.register_agent(0, patch=3)
        env.register_agent(1, patch=5)
        env.update_agent_state(1, vitality=0.8, surprise=0.3)

        obs, _, _ = env.step_agent(0, 3)
        social_part = obs.data[8:24]
        assert not np.allclose(social_part, 0.0)

    def test_self_emission_mean_centered(self):
        env = PatchForagingEnv(num_patches=8, include_social=True, include_self_emission=True)
        env.register_agent(0, patch=3)
        env.update_agent_state(0, vitality=0.7, surprise=0.2)

        obs, _, _ = env.step_agent(0, 3)
        # Self-emission: [24:28] vit, [28:32] surp
        own_vit = obs.data[24:28]
        own_surp = obs.data[28:32]
        assert abs(np.sum(own_vit)) < 0.01
        assert abs(np.sum(own_surp)) < 0.01

    def test_update_agent_state(self):
        env = PatchForagingEnv(num_patches=8)
        env.register_agent(0, patch=3)
        env.update_agent_state(0, vitality=0.3, surprise=0.9)
        assert env._leaked_state[0][0] == 0.3
        assert env._leaked_state[0][1] == 0.9

    def test_works_with_agent(self):
        """PatchForagingEnv observations work with Agent."""
        from fpi.agent.core import Agent

        env = PatchForagingEnv(num_patches=8, include_social=True, include_self_emission=True)
        agent = Agent(similarity_threshold=0.7, seed=42)

        obs = env.register_agent(0, patch=0)
        agent.step_with_action(obs, 0.0, None)

        for _ in range(30):
            action = agent.select_action(env.action_space)
            obs, delta, done = env.step_agent(0, action)
            agent.step_with_action(obs, delta, action)
            env.update_agent_state(0, agent.vitality.energy, agent.world_model.last_surprise)
            env.tick()

        assert len(agent.history) > 0
