"""Tests for neural policy agents (PPO and DQN)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fpi.minecraft.neural_policy import (
    ActorCritic,
    DQNAgent,
    DQNNetwork,
    PPOAgent,
    ReplayBuffer,
    RolloutBuffer,
)

OBS_DIM = 92
N_ACTIONS = 20
N_ACTIONS_FACTORED = 168


# ---------------------------------------------------------------------------
# ActorCritic
# ---------------------------------------------------------------------------

class TestActorCritic:
    def test_forward_shapes(self):
        net = ActorCritic(obs_dim=OBS_DIM, hidden=128, n_actions=N_ACTIONS)
        obs = torch.randn(4, OBS_DIM)
        logits, value = net(obs)
        assert logits.shape == (4, N_ACTIONS)
        assert value.shape == (4, 1)

    def test_forward_single(self):
        net = ActorCritic(obs_dim=OBS_DIM, hidden=128, n_actions=N_ACTIONS)
        obs = torch.randn(OBS_DIM)
        logits, value = net(obs.unsqueeze(0))
        assert logits.shape == (1, N_ACTIONS)

    def test_act_returns_valid(self):
        net = ActorCritic(obs_dim=OBS_DIM, hidden=128, n_actions=N_ACTIONS)
        obs = torch.randn(OBS_DIM)
        action, log_prob, value = net.act(obs)
        assert 0 <= action < N_ACTIONS
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_evaluate_shapes(self):
        net = ActorCritic(obs_dim=OBS_DIM, hidden=128, n_actions=N_ACTIONS)
        obs = torch.randn(8, OBS_DIM)
        actions = torch.randint(0, N_ACTIONS, (8,))
        log_probs, values, entropy = net.evaluate(obs, actions)
        assert log_probs.shape == (8,)
        assert values.shape == (8,)
        assert entropy.shape == (8,)

    def test_gradient_flow(self):
        net = ActorCritic(obs_dim=OBS_DIM, hidden=128, n_actions=N_ACTIONS)
        obs = torch.randn(4, OBS_DIM)
        logits, value = net(obs)
        loss = logits.sum() + value.sum()
        loss.backward()
        for param in net.parameters():
            assert param.grad is not None

    def test_factored_action_space(self):
        net = ActorCritic(obs_dim=OBS_DIM, hidden=128, n_actions=N_ACTIONS_FACTORED)
        obs = torch.randn(2, OBS_DIM)
        logits, value = net(obs)
        assert logits.shape == (2, N_ACTIONS_FACTORED)


# ---------------------------------------------------------------------------
# DQNNetwork
# ---------------------------------------------------------------------------

class TestDQNNetwork:
    def test_forward_shapes(self):
        net = DQNNetwork(obs_dim=OBS_DIM, hidden=128, n_actions=N_ACTIONS)
        obs = torch.randn(4, OBS_DIM)
        q = net(obs)
        assert q.shape == (4, N_ACTIONS)

    def test_factored(self):
        net = DQNNetwork(obs_dim=OBS_DIM, hidden=128, n_actions=N_ACTIONS_FACTORED)
        obs = torch.randn(2, OBS_DIM)
        q = net(obs)
        assert q.shape == (2, N_ACTIONS_FACTORED)


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------

class TestRolloutBuffer:
    def test_store_and_len(self):
        buf = RolloutBuffer()
        for i in range(10):
            buf.store(np.zeros(OBS_DIM), i % N_ACTIONS, 0.1, False, 0.5, -0.1)
        assert len(buf) == 10

    def test_clear(self):
        buf = RolloutBuffer()
        buf.store(np.zeros(OBS_DIM), 0, 0.0, False, 0.0, 0.0)
        buf.clear()
        assert len(buf) == 0

    def test_compute_gae(self):
        buf = RolloutBuffer()
        for i in range(5):
            buf.store(np.zeros(OBS_DIM), 0, 1.0, False, 0.5, -0.1)
        advantages, returns = buf.compute_gae(last_value=0.0, gamma=0.99, gae_lambda=0.95)
        assert advantages.shape == (5,)
        assert returns.shape == (5,)
        # Advantages should decrease toward the end (less future reward)
        assert advantages[0] > advantages[-1]


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def test_store_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.store(np.zeros(OBS_DIM), i % N_ACTIONS, 0.1, np.ones(OBS_DIM), False)
        assert len(buf) == 50
        obs, actions, rewards, next_obs, dones = buf.sample(8)
        assert obs.shape == (8, OBS_DIM)
        assert actions.shape == (8,)
        assert rewards.shape == (8,)
        assert next_obs.shape == (8, OBS_DIM)
        assert dones.shape == (8,)

    def test_capacity_overflow(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(20):
            buf.store(np.full(OBS_DIM, i), 0, 0.0, np.zeros(OBS_DIM), False)
        assert len(buf) == 10
        # Oldest entries should be gone
        obs, *_ = buf.sample(10)
        mins = obs.min(axis=1)
        assert mins.min() >= 10  # entries 0-9 evicted


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

class TestPPOAgent:
    def test_select_action(self):
        agent = PPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, seed=42)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, log_prob, value = agent.select_action(obs)
        assert 0 <= action < N_ACTIONS
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_ready_to_update(self):
        agent = PPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, rollout_len=10, seed=42)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        assert not agent.ready_to_update
        for _ in range(10):
            agent.store_transition(obs, 0, 0.1, False, 0.5, -0.1)
        assert agent.ready_to_update

    def test_update_returns_metrics(self):
        agent = PPOAgent(
            obs_dim=OBS_DIM, n_actions=N_ACTIONS,
            rollout_len=32, batch_size=16, epochs=2, seed=42,
        )
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        for _ in range(32):
            action, log_prob, value = agent.select_action(obs)
            agent.store_transition(obs, action, 0.1, False, value, log_prob)
            obs = np.random.randn(OBS_DIM).astype(np.float32)

        losses = agent.update(last_obs=obs)
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "entropy" in losses
        assert "approx_kl" in losses
        assert len(agent.buffer) == 0  # buffer cleared after update

    def test_save_load_roundtrip(self):
        agent = PPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, seed=42)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action_before, _, _ = agent.select_action(obs)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save(path)

            agent2 = PPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, seed=42)
            agent2.load(path)

            # Same weights → same deterministic forward pass
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits1, val1 = agent.network(obs_t)
                logits2, val2 = agent2.network(obs_t)
            assert torch.allclose(logits1, logits2)
            assert torch.allclose(val1, val2)
        finally:
            os.unlink(path)

    def test_factored_action_space(self):
        agent = PPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS_FACTORED, seed=42)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, _, _ = agent.select_action(obs)
        assert 0 <= action < N_ACTIONS_FACTORED


# ---------------------------------------------------------------------------
# DQNAgent
# ---------------------------------------------------------------------------

class TestDQNAgent:
    def test_select_action_greedy(self):
        agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, epsilon=0.0, seed=42)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = agent.select_action(obs)
        assert 0 <= action < N_ACTIONS

    def test_select_action_explores(self):
        agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, epsilon=1.0, seed=42)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        actions = {agent.select_action(obs) for _ in range(100)}
        # With epsilon=1.0, should explore multiple actions
        assert len(actions) > 1

    def test_action_space_constraint(self):
        agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, epsilon=0.0, seed=42)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        subset = [0, 1, 2]
        action = agent.select_action(obs, action_space=subset)
        assert action in subset

    def test_update_returns_none_before_train_start(self):
        agent = DQNAgent(
            obs_dim=OBS_DIM, n_actions=N_ACTIONS,
            train_start=100, seed=42,
        )
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for i in range(10):
            agent.store_transition(obs, 0, 0.1, obs, False)
        result = agent.update()
        assert result is None

    def test_update_returns_metrics(self):
        agent = DQNAgent(
            obs_dim=OBS_DIM, n_actions=N_ACTIONS,
            train_start=50, batch_size=16, seed=42,
        )
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        for _ in range(60):
            next_obs = np.random.randn(OBS_DIM).astype(np.float32)
            agent.store_transition(obs, 0, 0.1, next_obs, False)
            obs = next_obs

        result = agent.update()
        assert result is not None
        assert "q_loss" in result
        assert "epsilon" in result

    def test_epsilon_decays(self):
        agent = DQNAgent(
            obs_dim=OBS_DIM, n_actions=N_ACTIONS,
            epsilon=1.0, epsilon_decay=0.99, seed=42,
        )
        initial_eps = agent.epsilon
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for _ in range(10):
            agent.store_transition(obs, 0, 0.0, obs, False)
        assert agent.epsilon < initial_eps

    def test_save_load_roundtrip(self):
        agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, epsilon=0.5, seed=42)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            agent2 = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, seed=42)
            agent2.load(path)

            assert agent2.epsilon == 0.5
            obs_t = torch.randn(1, OBS_DIM)
            with torch.no_grad():
                q1 = agent.q_network(obs_t)
                q2 = agent2.q_network(obs_t)
            assert torch.allclose(q1, q2)
        finally:
            os.unlink(path)

    def test_factored_action_space(self):
        agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS_FACTORED, epsilon=0.0, seed=42)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = agent.select_action(obs)
        assert 0 <= action < N_ACTIONS_FACTORED

