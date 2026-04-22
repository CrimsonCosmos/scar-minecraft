"""EpisodicMemory — remembering specific events.

Unlike associations (which track statistical averages), episodic memory
stores snapshots of specific high-surprise moments. This enables:
"I've seen something like this before — last time, it was bad."

Episodes are stored in a ring buffer with fixed capacity (FIFO eviction).
Recall is by cosine similarity of pattern centroids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Episode:
    """A snapshot of a specific moment worth remembering.

    Attributes:
        tick: When this happened.
        pattern_id: The pattern that was active.
        centroid: The pattern's centroid (for similarity-based recall).
        surprise: How surprising this moment was.
        vitality: Agent's vitality at this moment.
        context_ids: Neighboring pattern IDs (before/after).
        valence: Vitality delta at this moment.
    """

    tick: int
    pattern_id: int
    centroid: NDArray[np.float64]
    surprise: float
    vitality: float
    context_ids: tuple[int, ...]
    valence: float
    action_taken: int | None = None


class EpisodicMemory:
    """Ring buffer of high-surprise episodes with similarity-based recall.

    Args:
        capacity: Maximum number of episodes stored (FIFO eviction).
        recall_threshold: Minimum cosine similarity for recall matches.
    """

    def __init__(self, capacity: int = 50, recall_threshold: float = 0.7) -> None:
        self._capacity = capacity
        self._recall_threshold = recall_threshold
        self._episodes: list[Episode] = []

    def record(self, episode: Episode) -> None:
        """Store an episode. Evicts oldest if at capacity."""
        if len(self._episodes) >= self._capacity:
            self._episodes.pop(0)
        self._episodes.append(episode)

    def recall(self, centroid: NDArray[np.float64], k: int = 3) -> list[Episode]:
        """Return up to k episodes similar to the query centroid.

        Matches require cosine similarity >= recall_threshold.
        Results sorted by similarity descending.
        """
        if len(self._episodes) == 0:
            return []

        norm_q = np.linalg.norm(centroid)
        if norm_q == 0.0:
            return []

        scored: list[tuple[float, Episode]] = []
        for ep in self._episodes:
            norm_e = np.linalg.norm(ep.centroid)
            if norm_e == 0.0:
                continue
            sim = float(np.dot(centroid, ep.centroid) / (norm_q * norm_e))
            if sim >= self._recall_threshold:
                scored.append((sim, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    @property
    def count(self) -> int:
        """Number of episodes currently stored."""
        return len(self._episodes)

    def get_recent(self, n: int = 5) -> list[Episode]:
        """Return the n most recent episodes."""
        return list(self._episodes[-n:])
