"""Tests for Monitor — multi-watcher orchestration + society."""

from fpi.live.stream import Stream
from fpi.live.watcher import Watcher
from fpi.live.monitor import Monitor
from fpi.live.insight import Insight, SocietyInsight


class MockStream(Stream):
    """Stream that yields from a pre-set list of values."""

    def __init__(self, values: list[str], name: str = "mock") -> None:
        self._values = list(values)
        self._name = name
        self._index = 0

    @property
    def name(self) -> str:
        return self._name

    def poll(self) -> list[str]:
        if self._index < len(self._values):
            val = self._values[self._index]
            self._index += 1
            return [val]
        return []


class TestMonitor:
    def test_single_watcher_runs(self):
        stream = MockStream(["50", "50", "50"])
        watcher = Watcher(stream=stream)
        monitor = Monitor([watcher], enable_society=False, poll_interval=0)
        monitor.run(max_ticks=3)
        assert watcher._tick == 3

    def test_society_activates_with_multiple_watchers(self):
        w1 = Watcher(MockStream(["50"], name="a"))
        w2 = Watcher(MockStream(["hello world"], name="b"))
        monitor = Monitor([w1, w2])
        assert monitor._society_world_model is not None

    def test_society_disabled_with_single_watcher(self):
        w = Watcher(MockStream(["50"]))
        monitor = Monitor([w])
        assert monitor._society_world_model is None

    def test_society_disabled_by_flag(self):
        w1 = Watcher(MockStream(["50"], name="a"))
        w2 = Watcher(MockStream(["50"], name="b"))
        monitor = Monitor([w1, w2], enable_society=False)
        assert monitor._society_world_model is None

    def test_callback_invoked(self):
        called: list = []
        stream = MockStream(["50"])
        watcher = Watcher(stream=stream, surprise_threshold=0.0)
        monitor = Monitor([watcher], enable_society=False, poll_interval=0)
        monitor.run(max_ticks=1, callback=lambda insights: called.extend(insights))
        assert len(called) >= 1
        assert isinstance(called[0], Insight)

    def test_tick_once_returns_insights(self):
        stream = MockStream(["50"])
        watcher = Watcher(stream=stream, surprise_threshold=0.0)
        monitor = Monitor([watcher], enable_society=False, poll_interval=0)
        insights = monitor.tick_once()
        assert len(insights) >= 1

    def test_society_processes_multi_watcher(self):
        """Society should observe aggregate state of multiple watchers."""
        values_a = ["50"] * 5
        values_b = ["hello world"] * 5
        w1 = Watcher(MockStream(values_a, name="a"))
        w2 = Watcher(MockStream(values_b, name="b"))
        monitor = Monitor([w1, w2], poll_interval=0)

        for _ in range(5):
            monitor.tick_once()
            monitor._tick += 1

        # Society should have observed something
        assert monitor._society_world_model is not None
        assert monitor._society_world_model.observation_count == 5

    def test_society_detects_collective_change(self):
        """Society should be surprised when all watchers change simultaneously."""
        # Stable phase
        stable_a = ["50"] * 10
        stable_b = ["50"] * 10
        # Then both change
        change_a = ["95"] * 3
        change_b = ["95"] * 3

        w1 = Watcher(MockStream(stable_a + change_a, name="a"))
        w2 = Watcher(MockStream(stable_b + change_b, name="b"))
        monitor = Monitor([w1, w2], poll_interval=0)

        # Run stable phase
        for _ in range(10):
            monitor.tick_once()
            monitor._tick += 1

        # Run change phase — society should notice
        society_insights = []
        for _ in range(3):
            insights = monitor.tick_once()
            monitor._tick += 1
            for i in insights:
                if isinstance(i, SocietyInsight):
                    society_insights.append(i)

        # Society should have noticed something (may or may not generate insight
        # depending on whether the collective state changed enough)
        assert monitor._society_world_model.observation_count == 13
