"""Monitor — orchestrates multiple watchers + optional Society.

When multiple streams are watched, a Society emerges that detects
cross-stream correlations invisible to any single watcher. Uses the
same primitives (WorldModel, Vitality, Valence) as the Phase 4 Society,
manually composed to avoid coupling to SharedGridEnv.
"""

from __future__ import annotations

import time
from typing import Callable

from ..primitives.vitality import Vitality
from ..primitives.valence import Valence
from ..world_model.model import WorldModel
from .bridge import WatcherBridge
from .insight import Insight, InsightLevel, SocietyInsight
from .watcher import Watcher


class Monitor:
    """Orchestrates multiple Watchers + optional Society.

    Args:
        watchers: All active watchers.
        enable_society: Enable cross-stream society layer.
        poll_interval: Seconds between poll cycles.
    """

    def __init__(
        self,
        watchers: list[Watcher],
        enable_society: bool = True,
        poll_interval: float = 1.0,
    ) -> None:
        self.watchers = watchers
        self._poll_interval = poll_interval
        self._tick = 0

        # Society layer — emerges with multiple watchers
        self._society_world_model: WorldModel | None = None
        self._society_vitality: Vitality | None = None
        self._society_valence: Valence | None = None
        self._bridge: WatcherBridge | None = None

        if len(watchers) > 1 and enable_society:
            self._init_society()

    def _init_society(self) -> None:
        """Initialize the society layer over multiple watchers.

        Uses the same primitives as Society in Phase 4, but without
        the SharedGridEnv dependency.
        """
        self._bridge = WatcherBridge(n_watchers=len(self.watchers))
        self._society_world_model = WorldModel(
            similarity_threshold=0.7,
            max_patterns=20,
            max_associations=60,
            association_decay_rate=0.005,
        )
        self._society_vitality = Vitality(entropy_rate=0.005)
        self._society_valence = Valence(learning_rate=0.3)

    def run(
        self,
        max_ticks: int | None = None,
        callback: Callable[[list[Insight | SocietyInsight]], None] | None = None,
    ) -> None:
        """Main monitoring loop.

        Args:
            max_ticks: Stop after this many ticks (None = run forever).
            callback: Called with insights per tick.
        """
        try:
            while max_ticks is None or self._tick < max_ticks:
                insights = self.tick_once()

                if callback and insights:
                    callback(insights)

                self._tick += 1
                if self._poll_interval > 0:
                    time.sleep(self._poll_interval)

        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def tick_once(self) -> list[Insight | SocietyInsight]:
        """Execute one monitoring tick.

        1. Poll all watchers
        2. If society exists, observe aggregate state
        """
        tick_insights: list[Insight | SocietyInsight] = []

        for watcher in self.watchers:
            watcher_insights = watcher.poll_and_observe()
            tick_insights.extend(watcher_insights)

        if self._society_world_model is not None and self._bridge is not None:
            society_insight = self._society_step()
            if society_insight is not None:
                tick_insights.append(society_insight)

        return tick_insights

    def _society_step(self) -> SocietyInsight | None:
        """One tick of the society layer.

        Structurally identical to Society.step() in Phase 4.
        """
        assert self._bridge is not None
        assert self._society_world_model is not None
        assert self._society_vitality is not None
        assert self._society_valence is not None

        # Sense: encode per-watcher state
        surprises: dict[str, float] = {}
        vitalities: dict[str, float] = {}
        for w in self.watchers:
            surprises[w.name] = w.agent.world_model.last_surprise
            vitalities[w.name] = w.agent.vitality.energy

        observation = self._bridge.encode(surprises, vitalities, self._tick)

        # Vitality delta: society thrives when watchers thrive
        vitality_before = self._society_vitality.energy
        mean_vitality = sum(vitalities.values()) / max(1, len(vitalities))
        mean_surprise = sum(surprises.values()) / max(1, len(surprises))

        collective_delta = (mean_vitality - 0.3) * 0.1 - mean_surprise * 0.05

        if collective_delta > 0:
            self._society_vitality.restore(collective_delta)
        elif collective_delta < 0:
            self._society_vitality.spend(abs(collective_delta))
        self._society_vitality.tick()

        maintenance = self._society_world_model.tick()
        if maintenance > 0:
            self._society_vitality.spend(maintenance)

        vitality_after = self._society_vitality.energy
        actual_delta = vitality_after - vitality_before

        # Observe
        surprise = self._society_world_model.observe(observation)

        # Update valence
        current = self._society_world_model.current_pattern
        if current is not None:
            self._society_valence.update(current.pattern_id, actual_delta)

        # Generate insight if society is surprised
        if surprise >= 0.5:
            level = (
                InsightLevel.ALERT if surprise > 0.8 else InsightLevel.ANOMALY
            )
            watcher_states = self._bridge.decode_summary(surprises, vitalities)
            message = self._format_society_message(surprise, watcher_states)
            return SocietyInsight(
                tick=self._tick,
                timestamp_seconds=time.time(),
                surprise=surprise,
                level=level,
                watcher_states=watcher_states,
                message=message,
                collective_vitality=self._society_vitality.energy,
            )
        return None

    def _format_society_message(
        self, surprise: float, states: dict[str, dict[str, float]]
    ) -> str:
        """Generate human-readable society insight."""
        high_surprise = [
            name for name, s in states.items() if s["surprise"] > 0.5
        ]
        if high_surprise:
            return (
                f"Cross-stream anomaly (surprise: {surprise:.2f}): "
                f"streams [{', '.join(high_surprise)}] are simultaneously unusual"
            )
        return f"Collective state change detected (surprise: {surprise:.2f})"

    def _cleanup(self) -> None:
        """Close all streams."""
        for w in self.watchers:
            w.stream.close()
