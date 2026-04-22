"""Working Memory — capacity-limited active buffer.

Implements Baddeley's working memory model: a limited-capacity buffer
(~4 items) where patterns are ACTIVELY maintained. Items decay unless
refreshed by attention. Maintenance costs energy.

Key difference from _recent_pattern_ids (which is a passive trace):
- Capacity limited (Cowan's 4±1)
- Items decay unless actively refreshed
- Maintenance costs vitality (thinking is work)
- Items can be compared and queried
- Serves as bridge between perception and action planning

The attention↔WM loop: attended patterns enter WM, WM items bias
attention priorities. This is the central executive function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class WorkingMemoryItem:
    """An item held in working memory."""

    pattern_id: int
    centroid: NDArray[np.float64]
    valence: float
    age: int = 0       # Ticks since last refresh
    refreshed: bool = False


class WorkingMemory:
    """Capacity-limited active buffer with decay and maintenance cost.

    Args:
        capacity: Maximum simultaneous items (Cowan's 4±1).
        decay_ticks: Ticks before unrehearsed item is evicted.
        maintenance_cost: Energy cost per item per tick.
    """

    def __init__(
        self,
        capacity: int = 4,
        decay_ticks: int = 8,
        maintenance_cost: float = 0.002,
    ) -> None:
        self._capacity = capacity
        self._decay_ticks = decay_ticks
        self._maintenance_cost = maintenance_cost
        self._items: list[WorkingMemoryItem] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def count(self) -> int:
        return len(self._items)

    def hold(
        self,
        pattern_id: int,
        centroid: NDArray[np.float64],
        valence: float,
    ) -> None:
        """Add or refresh an item in working memory.

        If already present, refreshes it. If at capacity, evicts the
        lowest-|valence| item to make room.
        """
        # Refresh if already present
        for item in self._items:
            if item.pattern_id == pattern_id:
                item.age = 0
                item.refreshed = True
                item.valence = valence
                item.centroid = centroid.copy()
                return

        # Evict if at capacity
        if len(self._items) >= self._capacity:
            weakest = min(self._items, key=lambda it: abs(it.valence))
            self._items.remove(weakest)

        self._items.append(WorkingMemoryItem(
            pattern_id=pattern_id,
            centroid=centroid.copy(),
            valence=valence,
            age=0,
            refreshed=True,
        ))

    def refresh(self, pattern_id: int) -> bool:
        """Reset decay timer for an item. Returns True if found."""
        for item in self._items:
            if item.pattern_id == pattern_id:
                item.age = 0
                item.refreshed = True
                return True
        return False

    def tick(self) -> float:
        """Age all items, evict decayed ones. Returns maintenance cost."""
        for item in self._items:
            item.age += 1
            item.refreshed = False

        # Evict expired items
        self._items = [
            item for item in self._items
            if item.age <= self._decay_ticks
        ]

        return len(self._items) * self._maintenance_cost

    def contents(self) -> list[WorkingMemoryItem]:
        """Current WM contents, sorted by age (newest first)."""
        return sorted(self._items, key=lambda it: it.age)

    def best_item(self) -> WorkingMemoryItem | None:
        """Highest-valence item in WM (for action guidance)."""
        if not self._items:
            return None
        return max(self._items, key=lambda it: it.valence)

    def contains(self, pattern_id: int) -> bool:
        """Is this pattern currently in WM?"""
        return any(item.pattern_id == pattern_id for item in self._items)

    def clear(self) -> None:
        """Clear all items."""
        self._items.clear()
