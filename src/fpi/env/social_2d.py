"""SocialGrid2DEnv — 2D multi-agent environment with leaky embodiment.

Extends SharedGrid2DEnv so that each agent's observation includes involuntary
emissions from the nearest other agent. Same architecture as 1D SocialGridEnv
but with 2D direction encoding (6 dims: 3 for dx + 3 for dy).

Observation layout (30 dims default, 44 with self-emission):
  [0:6]   Self row position         — 6 bases
  [6:12]  Self col position         — 6 bases
  [12:16] Nearest other's vitality  — 4 bases over [0, 1]
  [16:20] Nearest other's surprise  — 4 bases over [0, 1]
  [20:23] Nearest other's dx (row)  — 3 bases over [0, 1]
  [23:26] Nearest other's dy (col)  — 3 bases over [0, 1]
  [26:30] Nearest other's distance  — 4 bases over [0, perception_radius]

With include_self_emission=True:
  [30:34] Own vitality              — 4 bases, centered
  [34:38] Own surprise              — 4 bases, centered
  [38:41] Own dx                    — 3 bases, centered
  [41:44] Own dy                    — 3 bases, centered

Direction encoding maps action deltas {-1, 0, 1} → {0.0, 0.5, 1.0} per axis.
"""

from __future__ import annotations

import numpy as np

from ..primitives.signal import Signal
from .shared_2d import SharedGrid2DEnv, ACTION_DELTAS


class SocialGrid2DEnv(SharedGrid2DEnv):
    """A 2D shared grid where agents involuntarily leak internal state.

    Args:
        perception_radius: Manhattan distance within which agents perceive others.
        include_self_emission: Whether to include proprioceptive feedback.
        **kwargs: Passed to SharedGrid2DEnv.
    """

    VITALITY_BASES = 4
    SURPRISE_BASES = 4
    DIRECTION_BASES = 3  # Per axis (dx and dy each get 3)
    DISTANCE_BASES = 4
    SELF_EMISSION_DIMS = 14  # 4 vit + 4 surp + 3 dx + 3 dy

    def __init__(
        self,
        perception_radius: int = 5,
        include_self_emission: bool = False,
        scramble_social: bool = False,
        scramble_self_emission: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._perception_radius = perception_radius
        self._include_self_emission = include_self_emission
        self._scramble_social = scramble_social
        self._scramble_self_emission = scramble_self_emission

        # Gaussian basis centers and sigmas for social dims
        self._vitality_centers = np.linspace(0.0, 1.0, self.VITALITY_BASES)
        self._vitality_sigma = 0.25

        self._surprise_centers = np.linspace(0.0, 1.0, self.SURPRISE_BASES)
        self._surprise_sigma = 0.25

        # Direction: 3 bases per axis, maps {-1,0,1} → {0.0, 0.5, 1.0}
        self._direction_centers = np.linspace(0.0, 1.0, self.DIRECTION_BASES)
        self._direction_sigma = 0.33

        # Distance: Manhattan, 4 bases over [0, perception_radius]
        self._distance_centers = np.linspace(
            0.0, perception_radius, self.DISTANCE_BASES
        )
        self._distance_sigma = max(0.5, perception_radius / (self.DISTANCE_BASES * 3))

        # Leaked state: agent_id → (vitality, surprise, action)
        self._leaked_state: dict[int, tuple[float, float, int]] = {}

    @property
    def observation_dim(self) -> int:
        base = (
            self.NUM_POSITION_BASES * 2  # row + col
            + self.VITALITY_BASES
            + self.SURPRISE_BASES
            + self.DIRECTION_BASES * 2  # dx + dy
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

    def register_agent(
        self, agent_id: int, position: tuple[int, int] | None = None
    ) -> Signal:
        self._leaked_state[agent_id] = (0.5, 0.5, 4)  # default: stay
        super().register_agent(agent_id, position)
        return self._make_social_observation(agent_id)

    def step_agent(
        self, agent_id: int, action: int
    ) -> tuple[Signal, float, bool]:
        if agent_id not in self._agent_positions:
            raise ValueError(f"Agent {agent_id} not registered")

        # Record action in leaked state
        v, s, _old = self._leaked_state.get(agent_id, (0.5, 0.5, 4))
        self._leaked_state[agent_id] = (v, s, action)

        # Movement and resource check
        r, c = self._agent_positions[agent_id]
        dr, dc = ACTION_DELTAS[action]
        energy_delta = 0.0

        if action == 4:
            energy_delta -= self._stay_cost
        else:
            r = max(0, min(self._grid_size - 1, r + dr))
            c = max(0, min(self._grid_size - 1, c + dc))
            energy_delta -= self._move_cost

        self._agent_positions[agent_id] = (r, c)

        if (r, c) in self._resources:
            energy_delta += self._resource_value
            self._resources.discard((r, c))

        done = self._step_count >= self._max_steps
        return self._make_social_observation(agent_id), energy_delta, done

    def update_agent_state(
        self, agent_id: int, vitality: float, surprise: float
    ) -> None:
        """Update leaked state for others to perceive next tick."""
        _, _, action = self._leaked_state.get(agent_id, (0.5, 0.5, 4))
        self._leaked_state[agent_id] = (vitality, surprise, action)

    def reset(self) -> None:
        super().reset()
        self._leaked_state.clear()

    def _find_nearest_other(self, agent_id: int) -> int | None:
        """Find closest other agent within perception_radius (Manhattan)."""
        my_r, my_c = self._agent_positions[agent_id]
        nearest_id = None
        nearest_dist = float("inf")

        for other_id, (or_, oc) in self._agent_positions.items():
            if other_id == agent_id:
                continue
            dist = abs(my_r - or_) + abs(my_c - oc)
            if dist <= self._perception_radius and dist < nearest_dist:
                nearest_dist = dist
                nearest_id = other_id

        return nearest_id

    def _encode_gaussian(
        self, value: float, centers: np.ndarray, sigma: float
    ) -> np.ndarray:
        return np.exp(-((value - centers) ** 2) / (2 * sigma**2))

    def _action_to_direction(self, action: int) -> tuple[float, float]:
        """Map action to normalized direction values for encoding.

        delta {-1,0,1} → {0.0, 0.5, 1.0}
        """
        dr, dc = ACTION_DELTAS[action]
        return ((dr + 1) / 2.0, (dc + 1) / 2.0)

    def _make_social_observation(self, agent_id: int) -> Signal:
        """Build observation: position + nearest other's leaked state."""
        r, c = self._agent_positions[agent_id]

        # Position encoding (12 dims)
        r_basis = np.exp(
            -((r - self._basis_centers) ** 2) / (2 * self._basis_sigma**2)
        )
        c_basis = np.exp(
            -((c - self._basis_centers) ** 2) / (2 * self._basis_sigma**2)
        )

        # Social dims
        nearest_id = self._find_nearest_other(agent_id)

        if nearest_id is None:
            social_dim_count = (
                self.VITALITY_BASES
                + self.SURPRISE_BASES
                + self.DIRECTION_BASES * 2
                + self.DISTANCE_BASES
            )
            social = np.zeros(social_dim_count, dtype=np.float64)
        else:
            if self._scramble_social:
                # Random values from same distributions — destroys info, keeps structure
                vitality = float(self._rng.random())
                surprise = float(self._rng.random())
                dx_val = float(self._rng.choice([0.0, 0.5, 1.0]))
                dy_val = float(self._rng.choice([0.0, 0.5, 1.0]))
                dist = float(self._rng.random() * self._perception_radius)
            else:
                vitality, surprise, action = self._leaked_state[nearest_id]
                vitality = float(np.clip(vitality, 0.0, 1.0))
                surprise = float(np.clip(surprise, 0.0, 1.0))
                dx_val, dy_val = self._action_to_direction(action)
                or_, oc = self._agent_positions[nearest_id]
                dist = float(abs(r - or_) + abs(c - oc))

            vit_basis = self._encode_gaussian(
                vitality, self._vitality_centers, self._vitality_sigma,
            )
            surp_basis = self._encode_gaussian(
                surprise, self._surprise_centers, self._surprise_sigma,
            )
            dx_basis = self._encode_gaussian(
                dx_val, self._direction_centers, self._direction_sigma
            )
            dy_basis = self._encode_gaussian(
                dy_val, self._direction_centers, self._direction_sigma
            )
            dist_basis = self._encode_gaussian(
                dist, self._distance_centers, self._distance_sigma
            )

            social = np.concatenate(
                [vit_basis, surp_basis, dx_basis, dy_basis, dist_basis]
            )

        parts = [r_basis, c_basis, social]

        # Self-emission (proprioceptive feedback, one-tick delay)
        if self._include_self_emission:
            if self._scramble_self_emission:
                # Random values from same distributions
                own_v = float(self._rng.random())
                own_s = float(self._rng.random())
                own_dx_val = float(self._rng.choice([0.0, 0.5, 1.0]))
                own_dy_val = float(self._rng.choice([0.0, 0.5, 1.0]))
            else:
                own_v, own_s, own_action = self._leaked_state.get(
                    agent_id, (0.5, 0.5, 4)
                )
                own_v = float(np.clip(own_v, 0.0, 1.0))
                own_s = float(np.clip(own_s, 0.0, 1.0))
                own_dx_val, own_dy_val = self._action_to_direction(own_action)

            own_vit = self._encode_gaussian(
                own_v, self._vitality_centers, self._vitality_sigma,
            )
            own_vit -= np.mean(own_vit)

            own_surp = self._encode_gaussian(
                own_s, self._surprise_centers, self._surprise_sigma,
            )
            own_surp -= np.mean(own_surp)

            own_dx = self._encode_gaussian(
                own_dx_val, self._direction_centers, self._direction_sigma
            )
            own_dx -= np.mean(own_dx)
            own_dy = self._encode_gaussian(
                own_dy_val, self._direction_centers, self._direction_sigma
            )
            own_dy -= np.mean(own_dy)

            parts.append(np.concatenate([own_vit, own_surp, own_dx, own_dy]))

        data = np.concatenate(parts).astype(np.float64)
        return Signal(data=data, timestamp=self._step_count, modality="env")
