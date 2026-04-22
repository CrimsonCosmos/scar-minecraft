"""SignalBridge — converting aggregate agent state into society-level signals.

The society doesn't see individual neurons — it sees population-level
activity patterns. The SignalBridge encodes collective agent state as a
Signal that the society's WorldModel can process.

This is self-similar: the agent uses Gaussian basis encoding for its
position on the grid. The society uses regional density encoding for the
population's distribution across the grid. Same principle, different scale.
"""

from __future__ import annotations

import numpy as np

from ..primitives.signal import Signal


class SignalBridge:
    """Converts aggregate agent state into a Signal for the Society's WorldModel.

    Divides the grid into regions and computes per-region statistics:
    - Agent density: fraction of alive agents in each region
    - Mean vitality: average vitality of agents in each region

    The output Signal has dimensionality 2 * n_regions and modality "collective".

    Args:
        grid_size: Size of the grid agents live on.
        n_regions: Number of regions to divide the grid into.
    """

    def __init__(self, grid_size: int, n_regions: int = 4) -> None:
        self._grid_size = grid_size
        self._n_regions = n_regions
        self._region_size = grid_size / n_regions

    @property
    def signal_dim(self) -> int:
        return 2 * self._n_regions

    def encode(
        self,
        agent_positions: dict[int, int],
        agent_vitalities: dict[int, float],
        timestamp: int,
    ) -> Signal:
        """Encode collective agent state as a single Signal.

        Args:
            agent_positions: Mapping of agent_id to grid position.
            agent_vitalities: Mapping of agent_id to vitality energy.
            timestamp: Current tick.

        Returns:
            A Signal with modality "collective" encoding population statistics.
        """
        density = np.zeros(self._n_regions, dtype=np.float64)
        vitality_sum = np.zeros(self._n_regions, dtype=np.float64)
        counts = np.zeros(self._n_regions, dtype=np.float64)

        for agent_id, pos in agent_positions.items():
            region = min(int(pos / self._region_size), self._n_regions - 1)
            density[region] += 1
            vitality_sum[region] += agent_vitalities.get(agent_id, 0.0)
            counts[region] += 1

        # Normalize density to [0, 1]
        total = max(1.0, density.sum())
        density /= total

        # Average vitality per region (0 if no agents in region)
        mean_vitality = np.zeros(self._n_regions, dtype=np.float64)
        mask = counts > 0
        mean_vitality[mask] = vitality_sum[mask] / counts[mask]

        data = np.concatenate([density, mean_vitality])
        return Signal(data=data, timestamp=timestamp, modality="collective")
