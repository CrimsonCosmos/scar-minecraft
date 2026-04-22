"""Environment — the interface between agent and world.

Defines the abstract Environment protocol and concrete implementations:
- SequencePredictionEnv: passive signal prediction (Phase 1)
- SurvivalEnv: 1D grid with resources where the agent must persist (Phase 2)
- ContextualSurvivalEnv: resources shift positions over time (Phase 8)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..primitives.signal import Signal


class Environment(ABC):
    """Abstract environment interface.

    The agent interacts with the environment through:
    - reset(): start a new episode, get first observation
    - step(action): take an action, get (observation, energy_delta, done)
    - action_space: what actions are available
    """

    @abstractmethod
    def reset(self) -> Signal:
        """Reset the environment and return the initial observation."""
        ...

    @abstractmethod
    def step(self, action: int | None = None) -> tuple[Signal, float, bool]:
        """Advance one timestep.

        Args:
            action: The agent's chosen action (ignored in passive environments).

        Returns:
            (observation, energy_delta, done)
        """
        ...

    @property
    def action_space(self) -> list[int]:
        """Available discrete actions. Empty list = passive environment."""
        return []


class SequencePredictionEnv(Environment):
    """A repeating signal sequence that the agent must learn to predict.

    The environment cycles through a fixed sequence of signals. The agent
    receives one signal per timestep. This is the simplest possible environment
    for testing whether the sense-predict-learn loop works.

    Args:
        sequence: List of signal vectors that repeat cyclically.
        signal_dim: Dimensionality of signals (used if sequence not given).
        sequence_length: Number of distinct signals in the cycle.
        num_steps: Steps per episode before done=True.
    """

    def __init__(
        self,
        sequence: list[list[float]] | None = None,
        signal_dim: int = 4,
        sequence_length: int = 4,
        num_steps: int = 100,
    ) -> None:
        if sequence is not None:
            self._sequence = [np.array(s, dtype=np.float64) for s in sequence]
        else:
            rng = np.random.default_rng(42)
            self._sequence = [
                rng.standard_normal(signal_dim) for _ in range(sequence_length)
            ]
            # Normalize so cosine similarity is meaningful
            self._sequence = [
                s / np.linalg.norm(s) for s in self._sequence
            ]

        self._num_steps = num_steps
        self._step_count = 0
        self._index = 0

    @property
    def sequence_length(self) -> int:
        return len(self._sequence)

    def reset(self) -> Signal:
        self._step_count = 0
        self._index = 0
        return Signal(data=self._sequence[0].copy(), timestamp=0, modality="env")

    def step(self, action: int | None = None) -> tuple[Signal, float, bool]:
        self._step_count += 1
        self._index = self._step_count % len(self._sequence)
        done = self._step_count >= self._num_steps

        signal = Signal(
            data=self._sequence[self._index].copy(),
            timestamp=self._step_count,
            modality="env",
        )
        return signal, 0.0, done


class SurvivalEnv(Environment):
    """A 1D grid world where the agent must find resources to survive.

    The agent exists on a 1D grid. It can move left, right, or stay.
    Resources exist at fixed positions. Moving costs energy. Staying
    costs less. Finding a resource restores energy.

    This is the minimal environment that requires volition: the agent
    must CHOOSE to move toward resources or it dies.

    Actions: 0=move_left, 1=stay, 2=move_right

    The observation Signal uses Gaussian basis function encoding for
    position, creating distinct patterns for different locations. This
    lets the agent's pattern-matching machinery distinguish "here" from
    "there" via cosine similarity.

    Vitality is included in the observation as proprioception. The agent
    senses its own energy state as a signal. "Hunger" becomes a
    perceptible pattern, not a hardcoded drive.
    """

    NUM_POSITION_BASES = 6

    def __init__(
        self,
        grid_size: int = 20,
        resource_positions: list[int] | None = None,
        move_cost: float = 0.02,
        stay_cost: float = 0.005,
        resource_value: float = 0.4,
        max_steps: int = 200,
    ) -> None:
        self._grid_size = grid_size
        self._resource_positions = resource_positions if resource_positions is not None else [5, 15]

        # Gaussian basis centers for position encoding
        # Tight sigma so different positions produce clearly distinct patterns
        self._basis_centers = np.linspace(0, grid_size - 1, self.NUM_POSITION_BASES)
        self._basis_sigma = max(1.0, grid_size / (self.NUM_POSITION_BASES * 3))
        self._move_cost = move_cost
        self._stay_cost = stay_cost
        self._resource_value = resource_value
        self._max_steps = max_steps
        self._position = grid_size // 2
        self._step_count = 0

    @property
    def action_space(self) -> list[int]:
        return [0, 1, 2]  # left, stay, right

    def reset(self) -> Signal:
        self._position = self._grid_size // 2
        self._step_count = 0
        return self._make_observation()

    def step(self, action: int | None = None) -> tuple[Signal, float, bool]:
        """Execute action. Returns (observation, energy_delta, done).

        energy_delta is the vitality change caused by this step.
        The agent doesn't get "reward" — it gets resources or loses energy.
        """
        self._step_count += 1

        energy_delta = 0.0
        if action == 0:  # left
            self._position = max(0, self._position - 1)
            energy_delta -= self._move_cost
        elif action == 2:  # right
            self._position = min(self._grid_size - 1, self._position + 1)
            energy_delta -= self._move_cost
        else:  # stay (action == 1 or None)
            energy_delta -= self._stay_cost

        # Check for resource
        if self._position in self._resource_positions:
            energy_delta += self._resource_value

        done = self._step_count >= self._max_steps
        return self._make_observation(), energy_delta, done

    def _make_observation(self) -> Signal:
        """Create observation using Gaussian basis encoding for position.

        The observation is ONLY position — the agent must learn which
        positions are valuable through experience (valence), not through
        a built-in "food radar." Vitality is accessed directly as an
        internal state, not sensed as an external signal.
        """
        # Gaussian basis activations for position
        basis = np.exp(
            -((self._position - self._basis_centers) ** 2) / (2 * self._basis_sigma ** 2)
        )

        return Signal(
            data=basis.astype(np.float64),
            timestamp=self._step_count,
            modality="env",
        )


class ContextualSurvivalEnv(SurvivalEnv):
    """A survival environment where resource positions change over time.

    Resources alternate between different positions in phases. The agent
    must adapt when learned resource locations stop yielding energy.

    This is harder than SurvivalEnv because:
    - Learned knowledge becomes wrong at phase boundaries.
    - The agent must detect context switches and re-orient.
    - Temporal credit assignment helps: patterns preceding a phase
      transition get negative valence retroactively.

    Args:
        phase_length: Steps per phase before resources shift.
        resource_sets: List of resource position lists to alternate between.
        **kwargs: Passed to SurvivalEnv.
    """

    def __init__(
        self,
        phase_length: int = 50,
        resource_sets: list[list[int]] | None = None,
        **kwargs,
    ) -> None:
        # Default resource sets: two alternating configs
        if resource_sets is None:
            resource_sets = [[2, 4], [6, 8]]
        self._phase_length = phase_length
        self._resource_sets = resource_sets
        # Start with first resource set
        kwargs.setdefault("resource_positions", resource_sets[0])
        super().__init__(**kwargs)

    def reset(self) -> Signal:
        self._resource_positions = self._resource_sets[0]
        return super().reset()

    def step(self, action: int | None = None) -> tuple[Signal, float, bool]:
        # super().step() increments _step_count first, so use the upcoming count
        upcoming = self._step_count + 1
        phase = (upcoming // self._phase_length) % len(self._resource_sets)
        self._resource_positions = self._resource_sets[phase]
        return super().step(action)
