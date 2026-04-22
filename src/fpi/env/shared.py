"""SharedGridEnv — a multi-agent environment with finite, regenerating resources.

Multiple agents coexist on a 1D grid. Resources deplete when consumed and
regenerate stochastically. This creates natural competition and incentivizes
spreading out — the substrate from which coordination can emerge.

The Society can influence this environment by biasing where resources
regenerate (like a brain allocating blood flow). Individual agents don't
know the society exists — they just notice resources appearing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..primitives.signal import Signal


class SharedGridEnv:
    """A multi-agent 1D grid world with depletable, regenerating resources.

    This is the shared world where agents coexist. It extends the single-agent
    SurvivalEnv concept to multiple agents with resource competition.

    Args:
        grid_size: Length of the 1D grid.
        num_resources: Initial number of resource positions (evenly spaced).
        resource_value: Energy gained from collecting a resource.
        resource_regen_rate: Base probability per empty cell per tick of spawning a resource.
        move_cost: Energy cost of moving left/right.
        stay_cost: Energy cost of staying in place.
        max_steps: Maximum ticks per episode.
        seed: Random seed.
        resource_cluster_prob: Probability that a newly spawned resource also
            spawns on adjacent cells. Creates resource patches that reward
            social information. Default 0.0 (no clustering).
    """

    NUM_POSITION_BASES = 6

    def __init__(
        self,
        grid_size: int = 20,
        num_resources: int = 4,
        resource_value: float = 0.3,
        resource_regen_rate: float = 0.03,
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

        # Gaussian basis encoding (same as SurvivalEnv)
        self._basis_centers = np.linspace(0, grid_size - 1, self.NUM_POSITION_BASES)
        self._basis_sigma = max(1.0, grid_size / (self.NUM_POSITION_BASES * 3))

        # State
        self._agent_positions: dict[int, int] = {}
        self._resources: set[int] = set()
        self._step_count = 0

        # Regeneration bias: [0, 1] weight per cell (society's influence)
        # Default: uniform (all cells equally likely to regen)
        self._regen_bias: NDArray[np.float64] = np.ones(grid_size, dtype=np.float64)

        self._init_resources()

    def _init_resources(self) -> None:
        """Place initial resources evenly across the grid."""
        self._resources = set()
        if self._num_resources > 0:
            spacing = self._grid_size / (self._num_resources + 1)
            for i in range(1, self._num_resources + 1):
                pos = int(spacing * i)
                pos = min(pos, self._grid_size - 1)
                self._resources.add(pos)

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def action_space(self) -> list[int]:
        return [0, 1, 2]  # left, stay, right

    @property
    def agent_positions(self) -> dict[int, int]:
        """Current positions of all registered agents."""
        return dict(self._agent_positions)

    @property
    def resources(self) -> set[int]:
        """Current resource positions."""
        return set(self._resources)

    @property
    def step_count(self) -> int:
        return self._step_count

    def register_agent(self, agent_id: int, position: int | None = None) -> Signal:
        """Register an agent and place it on the grid.

        Args:
            agent_id: Unique identifier for this agent.
            position: Starting position (default: center of grid).

        Returns:
            Initial observation signal.
        """
        if position is None:
            position = self._grid_size // 2
        position = max(0, min(self._grid_size - 1, position))
        self._agent_positions[agent_id] = position
        return self._make_observation(position)

    def step_agent(self, agent_id: int, action: int) -> tuple[Signal, float, bool]:
        """Step one agent. Returns (observation, energy_delta, done).

        Same interface as SurvivalEnv.step() — individual agents don't know
        they're in a shared environment.
        """
        if agent_id not in self._agent_positions:
            raise ValueError(f"Agent {agent_id} not registered")

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

        # Check for resource (first-to-step gets it)
        if pos in self._resources:
            energy_delta += self._resource_value
            self._resources.discard(pos)

        done = self._step_count >= self._max_steps
        return self._make_observation(pos), energy_delta, done

    def tick(self) -> None:
        """Advance global clock and regenerate resources."""
        self._step_count += 1
        self._regenerate_resources()

    def reset(self) -> None:
        """Reset environment state for a new episode."""
        self._agent_positions.clear()
        self._step_count = 0
        self._regen_bias = np.ones(self._grid_size, dtype=np.float64)
        self._init_resources()

    def set_regen_bias(self, action: int) -> None:
        """Society's action: bias where resources regenerate.

        Action 0: bias left half of grid (more resources on the left)
        Action 1: even distribution (no bias)
        Action 2: bias right half of grid (more resources on the right)

        This is how the society influences agents without them knowing.
        Like a brain allocating blood flow to neural regions.
        """
        half = self._grid_size // 2
        if action == 0:  # bias left
            self._regen_bias[:half] = 2.0
            self._regen_bias[half:] = 0.5
        elif action == 2:  # bias right
            self._regen_bias[:half] = 0.5
            self._regen_bias[half:] = 2.0
        else:  # even
            self._regen_bias[:] = 1.0

    def _regenerate_resources(self) -> None:
        """Stochastically regenerate resources at empty cells.

        Phase 1: natural spawning (collect seed positions first).
        Phase 2: cluster expansion — only seeds expand, no cascading.
        """
        # Phase 1: natural spawning
        new_positions = []
        for pos in range(self._grid_size):
            if pos not in self._resources:
                rate = self._resource_regen_rate * self._regen_bias[pos]
                if self._rng.random() < rate:
                    new_positions.append(pos)

        for pos in new_positions:
            self._resources.add(pos)

        # Phase 2: cluster expansion (seeds only, no cascading)
        if self._resource_cluster_prob > 0:
            for pos in new_positions:
                for neighbor in [pos - 1, pos + 1]:
                    if 0 <= neighbor < self._grid_size and neighbor not in self._resources:
                        if self._rng.random() < self._resource_cluster_prob:
                            self._resources.add(neighbor)

    def _make_observation(self, position: int) -> Signal:
        """Same Gaussian basis encoding as SurvivalEnv — position only."""
        basis = np.exp(
            -((position - self._basis_centers) ** 2) / (2 * self._basis_sigma ** 2)
        )
        return Signal(
            data=basis.astype(np.float64),
            timestamp=self._step_count,
            modality="env",
        )
