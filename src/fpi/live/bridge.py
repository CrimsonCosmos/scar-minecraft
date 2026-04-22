"""WatcherBridge — converting per-watcher state into society-level signals.

Analogous to SignalBridge but for watchers instead of grid agents.
Encodes per-watcher (surprise, vitality) using Gaussian basis encoding.

Self-similar at three scales:
- Agent: Gaussian basis encodes position
- Society (Phase 4): regional density encodes population distribution
- Society (Phase 5): per-watcher Gaussian basis encodes surprise/vitality
"""

from __future__ import annotations

import numpy as np

from ..primitives.signal import Signal


class WatcherBridge:
    """Converts per-watcher state into a Signal for the Society's WorldModel.

    For each watcher, encodes:
    - Surprise level (Gaussian basis, range [0, 1])
    - Vitality level (Gaussian basis, range [0, 1])

    These per-watcher encodings are concatenated into a single signal.

    Args:
        n_watchers: Number of watchers being monitored.
        bases_per_dim: Gaussian bases per encoded dimension.
    """

    def __init__(self, n_watchers: int, bases_per_dim: int = 6) -> None:
        self._n_watchers = n_watchers
        self._bases_per_dim = bases_per_dim
        self._centers = np.linspace(0.0, 1.0, bases_per_dim)
        self._sigma = max(0.01, 1.0 / bases_per_dim)

    @property
    def signal_dim(self) -> int:
        """Total dimensionality: 2 dims * bases_per_dim * n_watchers."""
        return 2 * self._bases_per_dim * self._n_watchers

    def encode(
        self,
        watcher_surprises: dict[str, float],
        watcher_vitalities: dict[str, float],
        timestamp: int,
    ) -> Signal:
        """Encode all watcher states as a single collective Signal.

        Args:
            watcher_surprises: {watcher_name: last_surprise}
            watcher_vitalities: {watcher_name: vitality_energy}
            timestamp: Current tick.
        """
        parts: list[np.ndarray] = []
        for name in sorted(watcher_surprises.keys()):
            surprise = np.clip(watcher_surprises.get(name, 0.5), 0.0, 1.0)
            vitality = np.clip(watcher_vitalities.get(name, 0.0), 0.0, 1.0)

            s_basis = np.exp(
                -((surprise - self._centers) ** 2) / (2 * self._sigma**2)
            )
            v_basis = np.exp(
                -((vitality - self._centers) ** 2) / (2 * self._sigma**2)
            )

            parts.append(s_basis)
            parts.append(v_basis)

        data = np.concatenate(parts).astype(np.float64)
        return Signal(data=data, timestamp=timestamp, modality="collective")

    def decode_summary(
        self,
        watcher_surprises: dict[str, float],
        watcher_vitalities: dict[str, float],
    ) -> dict[str, dict[str, float]]:
        """Return per-watcher state as a readable dict."""
        summary: dict[str, dict[str, float]] = {}
        for name in sorted(watcher_surprises.keys()):
            summary[name] = {
                "surprise": watcher_surprises.get(name, 0.0),
                "vitality": watcher_vitalities.get(name, 0.0),
            }
        return summary
