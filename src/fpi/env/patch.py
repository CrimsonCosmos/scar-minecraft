"""PatchForagingEnv — a non-spatial multi-agent foraging environment.

Agents choose from N discrete patches each tick. Some patches are "rich"
(high resource probability), others are "poor." Social observation lets
agents see which patch another agent chose and whether they succeeded.

This environment isolates social learning from spatial navigation. The
social facilitation prediction is clean: after Agent A gets rewarded at
patch P, does Agent B choose patch P more than random baseline (1/N)?

Observation layout:
  Blind:          [0:N]  own patch — N Gaussian bases
  Social adds:    [N:2N] other's patch — N bases
                  [2N:2N+4] other's vitality — 4 bases
                  [2N+4:2N+8] other's surprise — 4 bases
  Self-emission:  [+4] own vitality (centered)
                  [+4] own surprise (centered)
"""

from __future__ import annotations

import numpy as np

from ..primitives.signal import Signal


class PatchForagingEnv:
    """A multi-agent patch foraging environment.

    Args:
        num_patches: Number of discrete patches to choose from.
        num_rich: Number of patches with high resource probability.
        rich_prob: Resource spawn probability for rich patches.
        poor_prob: Resource spawn probability for poor patches.
        resource_value: Energy gained from finding a resource.
        visit_cost: Energy cost per tick (regardless of patch).
        max_steps: Maximum ticks per episode.
        seed: Random seed.
        include_social: Whether observations include other agent's state.
        include_self_emission: Whether observations include own state.
    """

    VITALITY_BASES = 4
    SURPRISE_BASES = 4

    def __init__(
        self,
        num_patches: int = 8,
        num_rich: int = 2,
        rich_prob: float = 0.5,
        poor_prob: float = 0.05,
        resource_value: float = 0.4,
        visit_cost: float = 0.01,
        max_steps: int = 300,
        seed: int = 42,
        include_social: bool = False,
        include_self_emission: bool = False,
    ) -> None:
        self._num_patches = num_patches
        self._num_rich = num_rich
        self._rich_prob = rich_prob
        self._poor_prob = poor_prob
        self._resource_value = resource_value
        self._visit_cost = visit_cost
        self._max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._include_social = include_social
        self._include_self_emission = include_self_emission

        # Gaussian basis for patch encoding (tight sigma — each patch distinct)
        self._patch_centers = np.linspace(0, num_patches - 1, num_patches)
        self._patch_sigma = 0.5

        # Social encoding bases
        self._vitality_centers = np.linspace(0.0, 1.0, self.VITALITY_BASES)
        self._vitality_sigma = 0.25
        self._surprise_centers = np.linspace(0.0, 1.0, self.SURPRISE_BASES)
        self._surprise_sigma = 0.25

        # State
        self._agent_patches: dict[int, int] = {}  # agent_id → patch index
        self._resources: set[int] = set()  # patches that currently have resources
        self._richness: np.ndarray = np.full(num_patches, poor_prob)
        self._step_count = 0

        # Leaked state: agent_id → (vitality, surprise, patch_choice)
        self._leaked_state: dict[int, tuple[float, float, int]] = {}

        self._init_resources()

    def _init_resources(self) -> None:
        """Designate rich patches and spawn initial resources."""
        self._richness = np.full(self._num_patches, self._poor_prob)
        rich_indices = self._rng.choice(
            self._num_patches, self._num_rich, replace=False
        )
        self._richness[rich_indices] = self._rich_prob
        # Initial resource spawn
        self._resources = set()
        self._regenerate_resources()

    @property
    def num_patches(self) -> int:
        return self._num_patches

    @property
    def action_space(self) -> list[int]:
        return list(range(self._num_patches))

    @property
    def agent_patches(self) -> dict[int, int]:
        return dict(self._agent_patches)

    @property
    def resources(self) -> set[int]:
        return set(self._resources)

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def richness(self) -> np.ndarray:
        return self._richness.copy()

    def register_agent(self, agent_id: int, patch: int | None = None) -> Signal:
        """Register an agent at a patch (default: random)."""
        if patch is None:
            patch = int(self._rng.integers(0, self._num_patches))
        patch = max(0, min(self._num_patches - 1, patch))
        self._agent_patches[agent_id] = patch
        self._leaked_state[agent_id] = (0.5, 0.5, patch)
        return self._make_observation(agent_id)

    def step_agent(
        self, agent_id: int, action: int
    ) -> tuple[Signal, float, bool]:
        """Agent chooses a patch. Returns (observation, energy_delta, done)."""
        if agent_id not in self._agent_patches:
            raise ValueError(f"Agent {agent_id} not registered")

        patch = max(0, min(self._num_patches - 1, action))
        self._agent_patches[agent_id] = patch

        # Update leaked state with new patch choice
        v, s, _ = self._leaked_state.get(agent_id, (0.5, 0.5, 0))
        self._leaked_state[agent_id] = (v, s, patch)

        energy_delta = -self._visit_cost

        # Check for resource at chosen patch
        if patch in self._resources:
            energy_delta += self._resource_value
            self._resources.discard(patch)

        done = self._step_count >= self._max_steps
        return self._make_observation(agent_id), energy_delta, done

    def update_agent_state(
        self, agent_id: int, vitality: float, surprise: float
    ) -> None:
        """Update leaked state for others to perceive next tick."""
        _, _, patch = self._leaked_state.get(agent_id, (0.5, 0.5, 0))
        self._leaked_state[agent_id] = (vitality, surprise, patch)

    def tick(self) -> None:
        """Advance clock and regenerate resources."""
        self._step_count += 1
        self._regenerate_resources()

    def reset(self) -> None:
        """Reset for a new episode. Re-randomizes which patches are rich."""
        self._agent_patches.clear()
        self._leaked_state.clear()
        self._step_count = 0
        self._init_resources()

    def _regenerate_resources(self) -> None:
        """Each empty patch spawns a resource with probability = richness."""
        for patch in range(self._num_patches):
            if patch not in self._resources:
                if self._rng.random() < self._richness[patch]:
                    self._resources.add(patch)

    def _encode_gaussian(
        self, value: float, centers: np.ndarray, sigma: float
    ) -> np.ndarray:
        return np.exp(-((value - centers) ** 2) / (2 * sigma**2))

    def _make_observation(self, agent_id: int) -> Signal:
        """Build observation based on current mode (blind/social/proprioceptive)."""
        patch = self._agent_patches[agent_id]

        # Own patch encoding
        patch_basis = self._encode_gaussian(
            float(patch), self._patch_centers, self._patch_sigma
        )

        parts = [patch_basis]

        if self._include_social:
            # Find a random other alive agent
            others = [
                aid for aid in self._agent_patches if aid != agent_id
            ]
            if not others:
                # Alone: social dims all zeros
                social = np.zeros(
                    self._num_patches + self.VITALITY_BASES + self.SURPRISE_BASES,
                    dtype=np.float64,
                )
            else:
                other_id = others[int(self._rng.integers(0, len(others)))]
                other_v, other_s, other_patch = self._leaked_state[other_id]

                other_patch_basis = self._encode_gaussian(
                    float(other_patch),
                    self._patch_centers,
                    self._patch_sigma,
                )
                other_vit = self._encode_gaussian(
                    np.clip(other_v, 0.0, 1.0),
                    self._vitality_centers,
                    self._vitality_sigma,
                )
                other_surp = self._encode_gaussian(
                    np.clip(other_s, 0.0, 1.0),
                    self._surprise_centers,
                    self._surprise_sigma,
                )
                social = np.concatenate([other_patch_basis, other_vit, other_surp])

            parts.append(social)

        if self._include_self_emission:
            own_v, own_s, _ = self._leaked_state.get(agent_id, (0.5, 0.5, 0))

            own_vit = self._encode_gaussian(
                np.clip(own_v, 0.0, 1.0),
                self._vitality_centers,
                self._vitality_sigma,
            )
            own_vit -= np.mean(own_vit)

            own_surp = self._encode_gaussian(
                np.clip(own_s, 0.0, 1.0),
                self._surprise_centers,
                self._surprise_sigma,
            )
            own_surp -= np.mean(own_surp)

            parts.append(np.concatenate([own_vit, own_surp]))

        data = np.concatenate(parts).astype(np.float64)
        return Signal(data=data, timestamp=self._step_count, modality="env")
