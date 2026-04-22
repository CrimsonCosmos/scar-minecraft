"""SocialGridEnv — multi-agent environment with leaky embodiment.

Extends SharedGridEnv so that each agent's observation includes involuntary
emissions from the nearest other agent. Internal state (vitality, surprise,
movement direction) leaks as physics — not designed communication.

The agent doesn't know which dimensions are "self" vs "other." It processes
the wider signal with the same WorldModel it already has. Social cognition
must emerge from the same machinery, not from a social module.

Observation layout (21 dims default, all Gaussian-basis-encoded):
  [0:6]   Self position           — 6 bases over [0, grid_size-1]
  [6:10]  Nearest other's vitality — 4 bases over [0.0, 1.0]
  [10:14] Nearest other's surprise — 4 bases over [0.0, 1.0]
  [14:17] Nearest other's direction— 3 bases over [0.0, 1.0]
  [17:21] Nearest other's distance — 4 bases over [0, perception_radius]

With include_self_emission=True (32 dims, mean-centered):
  [21:25] Own vitality             — 4 bases over [0.0, 1.0], centered
  [25:29] Own surprise             — 4 bases over [0.0, 1.0], centered
  [29:32] Own direction            — 3 bases over [0.0, 1.0], centered

Self-emission reads from _leaked_state[agent_id] with one-tick delay,
same as how others perceive you. This proprioceptive feedback enables
pattern-level differentiation of own state, the foundation for
instrumental control of emissions.

Mean-centering self-emission dims prevents cosine similarity inflation:
when self-state changes slowly, uncentered Gaussian activations dominate
the dot product and collapse distinct positions into one pattern.
Centering makes near-constant self-state contribute ~0 to similarity.

When no agent is within perception_radius, social dims are all zeros.
This creates a distinct "alone" pattern the WorldModel learns to recognize.

State is perceived with a one-tick delay — physically realistic perception
latency. You see where the other agent WAS, not where it IS.
"""

from __future__ import annotations

import numpy as np

from ..primitives.signal import Signal
from .shared import SharedGridEnv


class SocialGridEnv(SharedGridEnv):
    """A shared grid where agents involuntarily leak internal state.

    Args:
        perception_radius: How far an agent can perceive others.
        **kwargs: Passed to SharedGridEnv.
    """

    # Gaussian basis counts per social dimension
    VITALITY_BASES = 4
    SURPRISE_BASES = 4
    DIRECTION_BASES = 3
    DISTANCE_BASES = 4
    SELF_EMISSION_DIMS = 11  # 4 vitality + 4 surprise + 3 direction

    def __init__(
        self,
        perception_radius: int = 5,
        include_self_emission: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._perception_radius = perception_radius
        self._include_self_emission = include_self_emission

        # Pre-compute Gaussian basis centers and sigmas for social dims
        # Vitality: 4 bases over [0, 1]
        self._vitality_centers = np.linspace(0.0, 1.0, self.VITALITY_BASES)
        self._vitality_sigma = 0.25  # 1.0 / VITALITY_BASES

        # Surprise: 4 bases over [0, 1]
        self._surprise_centers = np.linspace(0.0, 1.0, self.SURPRISE_BASES)
        self._surprise_sigma = 0.25

        # Direction: 3 bases over [0, 1] (0=left, 0.5=stay, 1=right)
        self._direction_centers = np.linspace(0.0, 1.0, self.DIRECTION_BASES)
        self._direction_sigma = 0.33  # 1.0 / DIRECTION_BASES

        # Distance: 4 bases over [0, perception_radius]
        self._distance_centers = np.linspace(0.0, perception_radius, self.DISTANCE_BASES)
        self._distance_sigma = max(0.5, perception_radius / (self.DISTANCE_BASES * 3))

        # Leaked state: agent_id -> (vitality, surprise, last_action)
        # Defaults: middle vitality, middle surprise, stay
        self._leaked_state: dict[int, tuple[float, float, int]] = {}

    @property
    def observation_dim(self) -> int:
        """Total observation dimensionality."""
        base = (
            self.NUM_POSITION_BASES
            + self.VITALITY_BASES
            + self.SURPRISE_BASES
            + self.DIRECTION_BASES
            + self.DISTANCE_BASES
        )
        if self._include_self_emission:
            base += self.SELF_EMISSION_DIMS
        return base

    @property
    def perception_radius(self) -> int:
        return self._perception_radius

    @property
    def include_self_emission(self) -> bool:
        return self._include_self_emission

    def register_agent(self, agent_id: int, position: int | None = None) -> Signal:
        """Register an agent with default leaked state."""
        # Initialize leaked state: vitality=0.5, surprise=0.5, action=stay(1)
        self._leaked_state[agent_id] = (0.5, 0.5, 1)
        # Call parent to place agent and get position set
        super().register_agent(agent_id, position)
        # Return social observation instead of parent's position-only one
        return self._make_social_observation(agent_id)

    def step_agent(self, agent_id: int, action: int) -> tuple[Signal, float, bool]:
        """Step one agent and return social observation.

        Records the action for direction encoding. The actual vitality and
        surprise are updated separately via update_agent_state().
        """
        if agent_id not in self._agent_positions:
            raise ValueError(f"Agent {agent_id} not registered")

        # Record action in leaked state (keep existing vitality/surprise)
        v, s, _old_action = self._leaked_state.get(agent_id, (0.5, 0.5, 1))
        self._leaked_state[agent_id] = (v, s, action)

        # Do the actual movement and resource check via parent
        pos = self._agent_positions[agent_id]
        energy_delta = 0.0

        if action == 0:  # left
            pos = max(0, pos - 1)
            energy_delta -= self._move_cost
        elif action == 2:  # right
            pos = min(self._grid_size - 1, pos + 1)
            energy_delta -= self._move_cost
        else:  # stay
            energy_delta -= self._stay_cost

        self._agent_positions[agent_id] = pos

        # Check for resource
        if pos in self._resources:
            energy_delta += self._resource_value
            self._resources.discard(pos)

        done = self._step_count >= self._max_steps

        # Return social observation instead of position-only
        return self._make_social_observation(agent_id), energy_delta, done

    def update_agent_state(
        self, agent_id: int, vitality: float, surprise: float
    ) -> None:
        """Update an agent's leaked state for others to perceive next tick.

        Called by SocialSociety after the agent processes its step.
        The one-tick delay means others see this state next tick, not now.
        """
        _v, _s, action = self._leaked_state.get(agent_id, (0.5, 0.5, 1))
        self._leaked_state[agent_id] = (vitality, surprise, action)

    def reset(self) -> None:
        """Reset environment and clear leaked state."""
        super().reset()
        self._leaked_state.clear()

    def _find_nearest_other(self, agent_id: int) -> int | None:
        """Find the closest other agent within perception_radius.

        Returns agent_id of nearest, or None if alone.
        """
        my_pos = self._agent_positions[agent_id]
        nearest_id = None
        nearest_dist = float("inf")

        for other_id, other_pos in self._agent_positions.items():
            if other_id == agent_id:
                continue
            dist = abs(my_pos - other_pos)
            if dist <= self._perception_radius and dist < nearest_dist:
                nearest_dist = dist
                nearest_id = other_id

        return nearest_id

    def _encode_gaussian(
        self, value: float, centers: np.ndarray, sigma: float
    ) -> np.ndarray:
        """Gaussian basis encoding of a scalar value."""
        return np.exp(-((value - centers) ** 2) / (2 * sigma**2))

    def _make_social_observation(self, agent_id: int) -> Signal:
        """Build observation: self position + nearest other's leaked state.

        21 dims without self-emission, 32 dims with self-emission enabled.
        """
        position = self._agent_positions[agent_id]

        # Self position: same Gaussian basis as SharedGridEnv (6 dims)
        position_basis = np.exp(
            -((position - self._basis_centers) ** 2)
            / (2 * self._basis_sigma**2)
        )

        # Find nearest other agent
        nearest_id = self._find_nearest_other(agent_id)

        if nearest_id is None:
            # Alone: all social dims are zeros
            social = np.zeros(
                self.VITALITY_BASES
                + self.SURPRISE_BASES
                + self.DIRECTION_BASES
                + self.DISTANCE_BASES,
                dtype=np.float64,
            )
        else:
            vitality, surprise, action = self._leaked_state[nearest_id]

            # Encode vitality (4 dims)
            vit_basis = self._encode_gaussian(
                np.clip(vitality, 0.0, 1.0),
                self._vitality_centers,
                self._vitality_sigma,
            )

            # Encode surprise (4 dims)
            surp_basis = self._encode_gaussian(
                np.clip(surprise, 0.0, 1.0),
                self._surprise_centers,
                self._surprise_sigma,
            )

            # Encode direction: 0=left→0.0, 1=stay→0.5, 2=right→1.0 (3 dims)
            direction_val = action / 2.0  # maps {0,1,2} → {0.0, 0.5, 1.0}
            dir_basis = self._encode_gaussian(
                direction_val,
                self._direction_centers,
                self._direction_sigma,
            )

            # Encode distance (4 dims)
            my_pos = self._agent_positions[agent_id]
            other_pos = self._agent_positions[nearest_id]
            dist = float(abs(my_pos - other_pos))
            dist_basis = self._encode_gaussian(
                dist, self._distance_centers, self._distance_sigma
            )

            social = np.concatenate([vit_basis, surp_basis, dir_basis, dist_basis])

        parts = [position_basis, social]

        # Self-emission: own leaked state (proprioceptive feedback)
        if self._include_self_emission:
            own_v, own_s, own_action = self._leaked_state.get(
                agent_id, (0.5, 0.5, 1)
            )

            own_vit = self._encode_gaussian(
                np.clip(own_v, 0.0, 1.0),
                self._vitality_centers,
                self._vitality_sigma,
            )
            own_vit -= np.mean(own_vit)  # Center to prevent cosine inflation

            own_surp = self._encode_gaussian(
                np.clip(own_s, 0.0, 1.0),
                self._surprise_centers,
                self._surprise_sigma,
            )
            own_surp -= np.mean(own_surp)  # Center to prevent cosine inflation

            own_dir_val = own_action / 2.0
            own_dir = self._encode_gaussian(
                own_dir_val,
                self._direction_centers,
                self._direction_sigma,
            )
            own_dir -= np.mean(own_dir)  # Center to prevent cosine inflation

            parts.append(np.concatenate([own_vit, own_surp, own_dir]))

        data = np.concatenate(parts).astype(np.float64)
        return Signal(data=data, timestamp=self._step_count, modality="env")
