"""SharedGrid2DEnv — a multi-agent 2D grid environment with finite, regenerating resources.

Extends the 1D SharedGridEnv concept to a 2D square grid. Agents have 5 actions
(up, down, left, right, stay) and observe their (row, col) position encoded as
two Gaussian basis vectors (12 dims total).

2D direction carries real information for social facilitation: if Agent A moves
east and finds a resource, a cluster likely extends further east. Following A's
direction in 2D narrows the search space from 4 directions to 1 — unlike 1D
where symmetric clusters make following ~50% correct.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..primitives.signal import Signal


# Action → (delta_row, delta_col)
ACTION_DELTAS: dict[int, tuple[int, int]] = {
    0: (-1, 0),   # up (north)
    1: (1, 0),    # down (south)
    2: (0, -1),   # left (west)
    3: (0, 1),    # right (east)
    4: (0, 0),    # stay
}


class SharedGrid2DEnv:
    """A multi-agent 2D grid world with depletable, regenerating resources.

    Args:
        grid_size: Side length of the square grid (grid_size × grid_size).
        num_resources: Initial number of resource positions (random).
        resource_value: Energy gained from collecting a resource.
        resource_regen_rate: Probability per empty cell per tick of spawning.
        move_cost: Energy cost of moving.
        stay_cost: Energy cost of staying.
        max_steps: Maximum ticks per episode.
        seed: Random seed.
        resource_cluster_prob: Probability that a newly spawned resource also
            spawns on adjacent cardinal cells. Default 0.0 (no clustering).
    """

    NUM_POSITION_BASES = 6  # Per axis

    def __init__(
        self,
        grid_size: int = 15,
        num_resources: int = 4,
        resource_value: float = 0.4,
        resource_regen_rate: float = 0.002,
        move_cost: float = 0.015,
        stay_cost: float = 0.005,
        max_steps: int = 300,
        seed: int = 42,
        resource_cluster_prob: float = 0.0,
    ) -> None:
        self._grid_size = grid_size
        self._num_resources = num_resources
        self._resource_value = resource_value
        self._resource_regen_rate = resource_regen_rate
        self._resource_cluster_prob = resource_cluster_prob
        self._move_cost = move_cost
        self._stay_cost = stay_cost
        self._max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._seed = seed

        # Gaussian basis encoding (same params per axis as 1D)
        self._basis_centers = np.linspace(0, grid_size - 1, self.NUM_POSITION_BASES)
        self._basis_sigma = max(1.0, grid_size / (self.NUM_POSITION_BASES * 3))

        # State
        self._agent_positions: dict[int, tuple[int, int]] = {}
        self._resources: set[tuple[int, int]] = set()
        self._step_count = 0

        self._init_resources()

    def _init_resources(self) -> None:
        """Place initial resources at random positions."""
        self._resources = set()
        while len(self._resources) < self._num_resources:
            r = int(self._rng.integers(0, self._grid_size))
            c = int(self._rng.integers(0, self._grid_size))
            self._resources.add((r, c))

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def action_space(self) -> list[int]:
        return [0, 1, 2, 3, 4]  # up, down, left, right, stay

    @property
    def agent_positions(self) -> dict[int, tuple[int, int]]:
        return dict(self._agent_positions)

    @property
    def resources(self) -> set[tuple[int, int]]:
        return set(self._resources)

    @property
    def step_count(self) -> int:
        return self._step_count

    def register_agent(
        self, agent_id: int, position: tuple[int, int] | None = None
    ) -> Signal:
        """Register an agent and place it on the grid."""
        if position is None:
            position = (self._grid_size // 2, self._grid_size // 2)
        r = max(0, min(self._grid_size - 1, position[0]))
        c = max(0, min(self._grid_size - 1, position[1]))
        self._agent_positions[agent_id] = (r, c)
        return self._make_observation((r, c))

    def step_agent(
        self, agent_id: int, action: int
    ) -> tuple[Signal, float, bool]:
        """Step one agent. Returns (observation, energy_delta, done)."""
        if agent_id not in self._agent_positions:
            raise ValueError(f"Agent {agent_id} not registered")

        r, c = self._agent_positions[agent_id]
        dr, dc = ACTION_DELTAS[action]
        energy_delta = 0.0

        if action == 4:  # stay
            energy_delta -= self._stay_cost
        else:
            r = max(0, min(self._grid_size - 1, r + dr))
            c = max(0, min(self._grid_size - 1, c + dc))
            energy_delta -= self._move_cost

        self._agent_positions[agent_id] = (r, c)

        # Check for resource
        if (r, c) in self._resources:
            energy_delta += self._resource_value
            self._resources.discard((r, c))

        done = self._step_count >= self._max_steps
        return self._make_observation((r, c)), energy_delta, done

    def tick(self) -> None:
        """Advance global clock and regenerate resources."""
        self._step_count += 1
        self._regenerate_resources()

    def reset(self) -> None:
        """Reset environment state for a new episode."""
        self._agent_positions.clear()
        self._step_count = 0
        self._init_resources()

    def _regenerate_resources(self) -> None:
        """Stochastically regenerate resources on empty cells.

        Phase 1: natural spawning (collect seeds).
        Phase 2: cluster expansion to cardinal neighbors (seeds only).
        """
        new_positions = []
        for r in range(self._grid_size):
            for c in range(self._grid_size):
                if (r, c) not in self._resources:
                    if self._rng.random() < self._resource_regen_rate:
                        new_positions.append((r, c))

        for pos in new_positions:
            self._resources.add(pos)

        if self._resource_cluster_prob > 0:
            for r, c in new_positions:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < self._grid_size
                        and 0 <= nc < self._grid_size
                        and (nr, nc) not in self._resources
                    ):
                        if self._rng.random() < self._resource_cluster_prob:
                            self._resources.add((nr, nc))

    def _make_observation(self, position: tuple[int, int]) -> Signal:
        """Gaussian basis encoding of (row, col) — 12 dims total."""
        r, c = position
        r_basis = np.exp(
            -((r - self._basis_centers) ** 2) / (2 * self._basis_sigma**2)
        )
        c_basis = np.exp(
            -((c - self._basis_centers) ** 2) / (2 * self._basis_sigma**2)
        )
        data = np.concatenate([r_basis, c_basis]).astype(np.float64)
        return Signal(data=data, timestamp=self._step_count, modality="env")
