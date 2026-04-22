"""Salience — learned per-dimension attention weights.

Salience modifies cosine similarity so the agent can learn which observation
dimensions are informative for predicting outcomes. Dimensions that frequently
deviate from pattern centroids when outcomes are unexpected get higher weight,
causing future observations to create finer-grained patterns along those
dimensions.

Learning rule (Hebbian):
    weights += lr * deviation * error
    weights *= (1 - decay)
    weights = clip(weights, min, max)
    weights /= mean(weights)  # only relative weights matter
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Salience:
    """Per-dimension attention weights learned from prediction errors.

    Attributes:
        _weights: Weight vector (same dim as signals).
        _learning_rate: How fast weights adapt.
        _decay: Per-tick decay rate (prevents unbounded growth).
        _min_weight: Floor on any dimension's weight.
        _max_weight: Ceiling on any dimension's weight.
    """

    _weights: NDArray[np.float64]
    _learning_rate: float = 0.2
    _decay: float = 0.001
    _min_weight: float = 0.1
    _max_weight: float = 5.0

    @staticmethod
    def uniform(
        dim: int,
        learning_rate: float = 0.2,
        decay: float = 0.001,
        min_weight: float = 0.1,
        max_weight: float = 5.0,
    ) -> Salience:
        """Create salience with uniform weights (all 1.0)."""
        return Salience(
            _weights=np.ones(dim, dtype=np.float64),
            _learning_rate=learning_rate,
            _decay=decay,
            _min_weight=min_weight,
            _max_weight=max_weight,
        )

    @property
    def weights(self) -> NDArray[np.float64]:
        return self._weights

    @property
    def dim(self) -> int:
        return len(self._weights)

    def update(self, deviation: NDArray[np.float64], error: float) -> None:
        """Update weights based on per-dimension deviation and outcome error.

        Dimensions that deviate from the pattern centroid when the outcome is
        unexpected get higher weight — they're the dimensions the pattern
        matching should have discriminated on.
        """
        self._weights += self._learning_rate * deviation * error
        self._weights *= (1.0 - self._decay)
        self._weights = np.clip(self._weights, self._min_weight, self._max_weight)
        # Normalize to mean 1.0 so only relative weights matter
        mean_w = float(np.mean(self._weights))
        if mean_w > 0:
            self._weights /= mean_w
