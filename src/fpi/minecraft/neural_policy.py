"""Neural policy agents for Minecraft — PPO and DQN.

Drop-in alternatives to the FPI compositional pattern-matching agent.
Same 428-dim signal input (396 base + 16 vision + 16 history), same
action space, same reward signal.

Requires PyTorch: pip install -e ".[neural]"

Usage:
    from fpi.minecraft.neural_policy import PPOAgent, DQNAgent

    agent = PPOAgent(obs_dim=428, n_actions=20)
    action = agent.select_action(obs_array, action_space)
    agent.store_transition(obs, action, reward, done, value, log_prob)
    losses = agent.update()
"""

from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
except ImportError as e:
    raise ImportError(
        "Neural policy requires PyTorch. Install with: pip install -e '.[neural]'"
    ) from e


class ActorCritic(nn.Module):
    """MLP with shared trunk, policy head (logits), and value head (scalar).

    Architecture:
        obs (428-dim) → Linear(128) → ReLU → Linear(128) → ReLU
                         ├→ Linear(n_actions) → policy logits
                         └→ Linear(1)         → state value
    """

    def __init__(self, obs_dim: int = 428, hidden: int = 128, n_actions: int = 20) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

        # Orthogonal init (standard for PPO)
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: Observation tensor, shape (batch, obs_dim) or (obs_dim,).

        Returns:
            (logits, value) — logits shape (batch, n_actions), value shape (batch, 1).
        """
        features = self.trunk(obs)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def act(self, obs: torch.Tensor) -> tuple[int, float, float]:
        """Sample an action from the policy.

        Args:
            obs: Single observation, shape (obs_dim,).

        Returns:
            (action, log_prob, value) — all as Python scalars.
        """
        with torch.no_grad():
            logits, value = self.forward(obs.unsqueeze(0))
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.squeeze().item()

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions under current policy (for PPO update).

        Args:
            obs: Batch of observations, shape (batch, obs_dim).
            actions: Batch of actions, shape (batch,).

        Returns:
            (log_probs, values, entropy) — for PPO loss computation.
        """
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


class RolloutBuffer:
    """Stores PPO rollout trajectories for batch updates."""

    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()

    def __len__(self) -> int:
        return len(self.observations)

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.

        Args:
            last_value: Bootstrap value for the final state.
            gamma: Discount factor.
            gae_lambda: GAE lambda.

        Returns:
            (advantages, returns) — both shape (N,).
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns


class PPOAgent:
    """Proximal Policy Optimization agent.

    Uses the same 428-dim signal as FPI but selects actions via a neural
    network trained with the clipped surrogate objective.
    """

    def __init__(
        self,
        obs_dim: int = 428,
        n_actions: int = 20,
        hidden: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_len: int = 2048,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_len = rollout_len

        self.network = ActorCritic(obs_dim, hidden, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()

        self._step_count = 0
        self._update_count = 0

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def select_action(
        self, obs: np.ndarray, action_space: list[int] | None = None
    ) -> tuple[int, float, float]:
        """Select an action given an observation.

        Args:
            obs: Observation array, shape (obs_dim,).
            action_space: Available actions (unused — network outputs all actions).

        Returns:
            (action, log_prob, value) — action index, log probability, state value.
        """
        obs_t = self._obs_to_tensor(obs)
        action, log_prob, value = self.network.act(obs_t)
        return action, log_prob, value

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """Store a transition in the rollout buffer."""
        self.buffer.store(obs, action, reward, done, value, log_prob)
        self._step_count += 1

    @property
    def ready_to_update(self) -> bool:
        """Whether enough transitions have been collected for an update."""
        return len(self.buffer) >= self.rollout_len

    def update(self, last_obs: np.ndarray | None = None) -> dict:
        """Run PPO update on collected rollout.

        Args:
            last_obs: Final observation for bootstrapping value. If None,
                      bootstrap value is 0.

        Returns:
            Dict with loss metrics (policy_loss, value_loss, entropy, approx_kl).
        """
        # Bootstrap value
        if last_obs is not None:
            with torch.no_grad():
                _, last_value = self.network.forward(
                    self._obs_to_tensor(last_obs).unsqueeze(0)
                )
                last_value = last_value.squeeze().item()
        else:
            last_value = 0.0

        # Compute GAE
        advantages, returns = self.buffer.compute_gae(
            last_value, self.gamma, self.gae_lambda
        )

        # Convert to tensors
        obs_t = torch.as_tensor(
            np.array(self.buffer.observations), dtype=torch.float32, device=self.device
        )
        actions_t = torch.as_tensor(
            self.buffer.actions, dtype=torch.long, device=self.device
        )
        old_log_probs_t = torch.as_tensor(
            self.buffer.log_probs, dtype=torch.float32, device=self.device
        )
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # PPO update epochs
        n = len(self.buffer)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        num_updates = 0

        for _ in range(self.epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = indices[start:end]

                batch_obs = obs_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]

                # Evaluate current policy
                new_log_probs, values, entropy = self.network.evaluate(
                    batch_obs, batch_actions
                )

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - new_log_probs).mean().item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                num_updates += 1

        self.buffer.clear()
        self._update_count += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "approx_kl": total_approx_kl / max(num_updates, 1),
        }

    def save(self, path: str) -> None:
        """Save model weights and optimizer state."""
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self._step_count,
                "update_count": self._update_count,
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model weights and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._step_count = checkpoint.get("step_count", 0)
        self._update_count = checkpoint.get("update_count", 0)


class ReplayBuffer:
    """Circular replay buffer for DQN."""

    def __init__(self, capacity: int = 100_000) -> None:
        self._buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append((obs, action, reward, next_obs, done))

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self._buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)


class DQNNetwork(nn.Module):
    """Q-network: MLP mapping observations to Q-values for each action."""

    def __init__(self, obs_dim: int = 428, hidden: int = 128, n_actions: int = 20) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DQNAgent:
    """Deep Q-Network agent with experience replay and target network.

    Uses the same 428-dim signal as FPI but selects actions via epsilon-greedy
    over Q-values estimated by a neural network.
    """

    def __init__(
        self,
        obs_dim: int = 428,
        n_actions: int = 20,
        hidden: int = 128,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        train_start: int = 1000,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_start = train_start

        self.q_network = DQNNetwork(obs_dim, hidden, n_actions).to(self.device)
        self.target_network = DQNNetwork(obs_dim, hidden, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self._step_count = 0
        self._update_count = 0
        self._rng = random.Random(seed)

    def select_action(
        self, obs: np.ndarray, action_space: list[int] | None = None
    ) -> int:
        """Epsilon-greedy action selection.

        Args:
            obs: Observation array, shape (obs_dim,).
            action_space: Available actions. If provided, constrains selection.

        Returns:
            Selected action index.
        """
        if self._rng.random() < self.epsilon:
            if action_space is not None:
                return self._rng.choice(action_space)
            return self._rng.randint(0, self.n_actions - 1)

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            q_values = self.q_network(obs_t.unsqueeze(0)).squeeze(0)

            if action_space is not None:
                mask = torch.full((self.n_actions,), float("-inf"), device=self.device)
                for a in action_space:
                    mask[a] = 0.0
                q_values = q_values + mask

            return q_values.argmax().item()

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.store(obs, action, reward, next_obs, done)
        self._step_count += 1

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(self) -> dict | None:
        """Sample a batch from replay buffer and minimize TD error.

        Returns:
            Dict with loss metric, or None if not enough data.
        """
        if len(self.buffer) < self.train_start:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs_t = torch.as_tensor(obs, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)

        # Current Q-values
        q_values = self.q_network(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values (no gradient)
        with torch.no_grad():
            next_q = self.target_network(next_obs_t).max(1).values
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        self._update_count += 1

        # Sync target network
        if self._update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {"q_loss": loss.item(), "epsilon": self.epsilon}

    def save(self, path: str) -> None:
        """Save model weights and optimizer state."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self._step_count,
                "update_count": self._update_count,
                "epsilon": self.epsilon,
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model weights and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._step_count = checkpoint.get("step_count", 0)
        self._update_count = checkpoint.get("update_count", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
