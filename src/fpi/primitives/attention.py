"""Attention — selective gating via competitive broadcast.

Implements Global Workspace Theory (Baars, Dehaene): multiple information
channels compete for a limited-capacity "workspace." Winners get full
processing; losers are suppressed (attenuated, not deleted).

This is NOT salience (which weights dimensions continuously). Attention is
discrete channel selection — some modalities are fully ON, others are
scaled down. Combined with affect: high arousal narrows attention (fewer
channels win), implementing the Easterbrook hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .signal import Signal


@dataclass(slots=True)
class AttentionChannel:
    """One channel competing for the workspace."""

    name: str
    slice_start: int
    slice_end: int
    priority: float = 1.0
    suppressed: bool = False


class AttentionGate:
    """Competitive selection: channels compete, top-K win.

    Winners are "broadcast" (fully processed). Losers have their signal
    dims attenuated by suppress_factor — they still contribute, but weakly.

    Args:
        capacity: How many channels can be attended simultaneously.
        suppress_factor: Scaling factor for losing channels (0 = full
            suppression, 1 = no suppression).
    """

    def __init__(
        self,
        capacity: int = 2,
        suppress_factor: float = 0.1,
    ) -> None:
        self._capacity = capacity
        self._suppress_factor = suppress_factor

    @property
    def capacity(self) -> int:
        return self._capacity

    def effective_capacity(self, arousal: float = 0.0) -> int:
        """Capacity narrows under high arousal (Easterbrook hypothesis).

        At arousal=0, full capacity. At arousal=1, capacity reduces to 1
        (tunnel vision on highest-priority channel only).
        """
        if arousal <= 0.0 or self._capacity <= 1:
            return self._capacity
        reduction = int(arousal * (self._capacity - 1))
        return max(1, self._capacity - reduction)

    def compete(
        self, channels: list[AttentionChannel], arousal: float = 0.0,
    ) -> list[AttentionChannel]:
        """Sort channels by priority, mark losers as suppressed.

        Returns all channels with suppression flags set.
        Arousal narrows effective capacity (fewer winners).
        """
        effective = self.effective_capacity(arousal)
        sorted_channels = sorted(
            channels, key=lambda c: c.priority, reverse=True,
        )
        for i, ch in enumerate(sorted_channels):
            ch.suppressed = i >= effective
        return sorted_channels

    def gate(
        self,
        observation: Signal,
        modality_slices: list[tuple[int, int]],
        priorities: list[float],
        arousal: float = 0.0,
    ) -> Signal:
        """Gate a multi-modal observation by attenuating unattended modalities.

        Args:
            observation: The full multi-modal signal.
            modality_slices: [(start, end), ...] for each modality.
            priorities: Priority score for each modality (same order as slices).

        Returns:
            New Signal with suppressed modality dims scaled down.
        """
        if len(modality_slices) != len(priorities):
            return observation  # Safety fallback

        channels = [
            AttentionChannel(
                name=f"mod_{i}",
                slice_start=start,
                slice_end=end,
                priority=priorities[i],
            )
            for i, (start, end) in enumerate(modality_slices)
        ]

        self.compete(channels, arousal=arousal)

        gated = observation.data.copy()
        for ch in channels:
            if ch.suppressed:
                gated[ch.slice_start:ch.slice_end] *= self._suppress_factor

        return Signal(
            data=gated,
            timestamp=observation.timestamp,
            modality=observation.modality,
        )
