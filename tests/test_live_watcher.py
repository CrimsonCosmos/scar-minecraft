"""Tests for Watcher — agent watching a data stream."""

import pytest

from fpi.live.insight import InsightKind, InsightLevel
from fpi.live.stream import Stream
from fpi.live.watcher import Watcher


class MockStream(Stream):
    """Stream that yields from a pre-set list of values."""

    def __init__(self, values: list[str]) -> None:
        self._values = list(values)
        self._index = 0

    @property
    def name(self) -> str:
        return "mock"

    def poll(self) -> list[str]:
        if self._index < len(self._values):
            val = self._values[self._index]
            self._index += 1
            return [val]
        return []


class TestWatcher:
    def test_observe_processes_data(self):
        """Watcher should process data from stream."""
        stream = MockStream(["50", "50", "50"])
        watcher = Watcher(stream=stream, surprise_threshold=0.3)
        for _ in range(3):
            watcher.poll_and_observe()
        assert watcher._tick == 3

    def test_first_observation_has_surprise(self):
        """First observation is always maximally surprising."""
        stream = MockStream(["42"])
        watcher = Watcher(stream=stream, surprise_threshold=0.0)
        insights = watcher.poll_and_observe()
        assert len(insights) == 1
        assert insights[0].surprise == 1.0  # First observation = max surprise

    def test_repeated_values_reduce_surprise(self):
        """Same value repeated should eventually have low surprise."""
        values = ["50"] * 20
        stream = MockStream(values)
        watcher = Watcher(stream=stream, surprise_threshold=0.5)

        # Feed repeated values
        all_insights = []
        for _ in range(20):
            insights = watcher.poll_and_observe()
            all_insights.extend(insights)

        # After learning, surprise should be low (no insights near end)
        # First observation always surprises, later ones should not
        assert watcher.agent.world_model.last_surprise < 0.5

    def test_novel_value_after_stable_causes_surprise(self):
        """A novel value after a stable sequence should trigger an insight."""
        values = ["50"] * 10 + ["95"]
        stream = MockStream(values)
        watcher = Watcher(stream=stream, surprise_threshold=0.3)

        for _ in range(10):
            watcher.poll_and_observe()

        # "95" should be surprising after learning "50"
        insights = watcher.poll_and_observe()
        assert len(insights) >= 1
        assert insights[0].is_new_pattern

    def test_vitality_stays_healthy_with_predictable_stream(self):
        """Predictable stream keeps vitality high."""
        # Alternating numeric values — predictable pattern
        values = ["50", "51"] * 40
        stream = MockStream(values)
        watcher = Watcher(stream=stream)

        for _ in range(80):
            watcher.poll_and_observe()

        # Agent should still be alive (predictions succeed → energy restores)
        assert watcher.agent.vitality.alive

    def test_empty_stream_no_insights(self):
        """Empty poll returns no insights."""
        stream = MockStream([])
        watcher = Watcher(stream=stream)
        insights = watcher.poll_and_observe()
        assert insights == []

    def test_insight_fields(self):
        """Insight should have all required fields."""
        stream = MockStream(["hello"])
        watcher = Watcher(stream=stream, surprise_threshold=0.0)
        insights = watcher.poll_and_observe()
        assert len(insights) == 1
        insight = insights[0]
        assert insight.stream_name == "mock"
        assert insight.tick == 0
        assert insight.timestamp_seconds > 0
        assert insight.raw_value == "hello"
        assert insight.pattern_id >= 0
        assert insight.vitality > 0

    def test_custom_agent_kwargs(self):
        """Agent kwargs should be passed through."""
        stream = MockStream(["50"])
        watcher = Watcher(
            stream=stream,
            agent_kwargs={"similarity_threshold": 0.9, "max_patterns": 10},
        )
        assert watcher.agent.world_model.memory.distinction.similarity_threshold == 0.9


class TestWatcherStatus:
    def test_get_status_returns_dict(self):
        stream = MockStream(["50"] * 10)
        watcher = Watcher(stream=stream)
        for _ in range(10):
            watcher.poll_and_observe()
        status = watcher.get_status()
        assert isinstance(status, dict)
        assert status["tick"] == 10
        assert "vitality" in status
        assert "patterns_learned" in status
        assert "alive" in status

    def test_get_status_patterns_have_valence(self):
        stream = MockStream(["50"] * 5)
        watcher = Watcher(stream=stream)
        for _ in range(5):
            watcher.poll_and_observe()
        status = watcher.get_status()
        assert status["patterns_learned"] >= 1
        for p in status["patterns"]:
            assert "valence" in p
            assert "pattern_id" in p
            assert "exposure_count" in p
            assert "fitness" in p

    def test_get_status_associations(self):
        stream = MockStream(["50", "60"] * 10)
        watcher = Watcher(stream=stream)
        for _ in range(20):
            watcher.poll_and_observe()
        status = watcher.get_status()
        # After alternating values, should have associations
        assert len(status["associations"]) >= 1
        assert len(status["strongest_associations"]) >= 1
        top = status["strongest_associations"][0]
        assert "source_id" in top
        assert "strength" in top

    def test_get_status_prediction(self):
        stream = MockStream(["50", "60"] * 10)
        watcher = Watcher(stream=stream)
        for _ in range(20):
            watcher.poll_and_observe()
        status = watcher.get_status()
        # After learning alternating pattern, should have a prediction
        assert status["current_prediction"] is not None
        assert status["prediction_confidence"] is not None

    def test_get_status_valence_counts(self):
        stream = MockStream(["50"] * 10)
        watcher = Watcher(stream=stream)
        for _ in range(10):
            watcher.poll_and_observe()
        status = watcher.get_status()
        total = (
            status["positive_patterns"]
            + status["negative_patterns"]
            + status["neutral_patterns"]
        )
        assert total == status["patterns_learned"]


class TestStatusReports:
    def test_status_report_emitted_at_interval(self):
        stream = MockStream(["50"] * 55)
        watcher = Watcher(stream=stream, surprise_threshold=2.0, report_interval=50)
        all_insights = []
        for _ in range(55):
            all_insights.extend(watcher.poll_and_observe())
        status_reports = [i for i in all_insights if i.kind == InsightKind.STATUS_REPORT]
        assert len(status_reports) == 1  # at tick 50

    def test_no_status_report_when_disabled(self):
        stream = MockStream(["50"] * 55)
        watcher = Watcher(stream=stream, surprise_threshold=2.0, report_interval=0)
        all_insights = []
        for _ in range(55):
            all_insights.extend(watcher.poll_and_observe())
        status_reports = [i for i in all_insights if i.kind == InsightKind.STATUS_REPORT]
        assert len(status_reports) == 0

    def test_status_report_has_correct_kind(self):
        stream = MockStream(["50"] * 11)
        watcher = Watcher(stream=stream, report_interval=10)
        all_insights = []
        for _ in range(11):
            all_insights.extend(watcher.poll_and_observe())
        reports = [i for i in all_insights if i.kind == InsightKind.STATUS_REPORT]
        assert len(reports) == 1
        assert reports[0].level == InsightLevel.INFO
        assert reports[0].stream_name == "mock"
        assert "patterns" in reports[0].message

    def test_status_report_message_content(self):
        stream = MockStream(["50", "60"] * 30)
        watcher = Watcher(stream=stream, report_interval=20)
        all_insights = []
        for _ in range(60):
            all_insights.extend(watcher.poll_and_observe())
        reports = [i for i in all_insights if i.kind == InsightKind.STATUS_REPORT]
        assert len(reports) >= 1
        msg = reports[-1].message
        # Should contain pattern count and vitality
        assert "vitality" in msg
        assert "surprise" in msg

    def test_surprise_insight_enriched_with_valence(self):
        stream = MockStream(["50"] * 5 + ["95"])
        watcher = Watcher(stream=stream, surprise_threshold=0.3)
        all_insights = []
        for _ in range(6):
            all_insights.extend(watcher.poll_and_observe())
        surprise_insights = [i for i in all_insights if i.kind == InsightKind.SURPRISE]
        # At least the first observation should be surprising
        assert len(surprise_insights) >= 1
        # valence field should be populated (may be 0.0 or a learned value)
        for si in surprise_insights:
            if si.pattern_id >= 0:
                assert si.valence is not None


class TestWatcherSequence:
    def test_sequence_memory_disabled_by_default(self):
        stream = MockStream(["50"] * 5)
        watcher = Watcher(stream=stream)
        assert watcher._sequence_memory is None

    def test_sequence_memory_enabled(self):
        stream = MockStream(["50"] * 10)
        watcher = Watcher(stream=stream, sequence_window=3)
        assert watcher._sequence_memory is not None
        for _ in range(10):
            watcher.poll_and_observe()
        assert watcher._sequence_memory.observation_count > 0

    def test_status_includes_sequence_info_when_enabled(self):
        stream = MockStream(["50"] * 10)
        watcher = Watcher(stream=stream, sequence_window=3)
        for _ in range(10):
            watcher.poll_and_observe()
        status = watcher.get_status()
        assert status["sequences"] is not None
        assert "sequence_patterns_learned" in status["sequences"]
        assert status["sequences"]["sequence_patterns_learned"] >= 1

    def test_status_sequences_none_when_disabled(self):
        stream = MockStream(["50"] * 5)
        watcher = Watcher(stream=stream)
        for _ in range(5):
            watcher.poll_and_observe()
        status = watcher.get_status()
        assert status["sequences"] is None

    def test_sequence_learns_from_alternating_values(self):
        """Alternating values should create recognizable sequences."""
        values = ["50", "60", "70"] * 20
        stream = MockStream(values)
        watcher = Watcher(stream=stream, sequence_window=3)
        for _ in range(60):
            watcher.poll_and_observe()
        sm = watcher._sequence_memory
        assert sm is not None
        assert sm.pattern_count >= 1
        assert sm.observation_count >= 10

    def test_status_report_includes_sequence_info(self):
        values = ["50", "60", "70"] * 20
        stream = MockStream(values)
        watcher = Watcher(
            stream=stream, sequence_window=3, report_interval=30
        )
        all_insights = []
        for _ in range(60):
            all_insights.extend(watcher.poll_and_observe())
        reports = [i for i in all_insights if i.kind == InsightKind.STATUS_REPORT]
        assert len(reports) >= 1
        # Status message should mention sequences
        assert "sequences" in reports[-1].message


class TestWatcherTemporal:
    def test_temporal_hierarchy_disabled_by_default(self):
        stream = MockStream(["50"] * 5)
        watcher = Watcher(stream=stream)
        assert watcher._temporal_hierarchy is None

    def test_temporal_hierarchy_enabled(self):
        stream = MockStream(["50", "60", "70"] * 20)
        watcher = Watcher(stream=stream, temporal_scales=(3, 5))
        assert watcher._temporal_hierarchy is not None
        for _ in range(60):
            watcher.poll_and_observe()
        status = watcher.get_status()
        assert status["temporal"] is not None
        assert "scales" in status["temporal"]
        assert 3 in status["temporal"]["scales"]
        assert 5 in status["temporal"]["scales"]

    def test_temporal_scales_overrides_sequence_window(self):
        stream = MockStream(["50"] * 5)
        watcher = Watcher(stream=stream, sequence_window=3, temporal_scales=(2, 4))
        # temporal_scales takes precedence
        assert watcher._temporal_hierarchy is not None
        assert watcher._sequence_memory is None

    def test_status_report_includes_temporal_info(self):
        values = ["50", "60", "70"] * 20
        stream = MockStream(values)
        watcher = Watcher(
            stream=stream, temporal_scales=(3,), report_interval=30
        )
        all_insights = []
        for _ in range(60):
            all_insights.extend(watcher.poll_and_observe())
        reports = [i for i in all_insights if i.kind == InsightKind.STATUS_REPORT]
        assert len(reports) >= 1
        assert "temporal" in reports[-1].message


class TestWatcherEpisodic:
    def test_episodic_created_with_temporal(self):
        stream = MockStream(["50"] * 5)
        watcher = Watcher(stream=stream, temporal_scales=(3,))
        assert watcher._episodic is not None

    def test_episodic_created_with_sequence(self):
        stream = MockStream(["50"] * 5)
        watcher = Watcher(stream=stream, sequence_window=3)
        assert watcher._episodic is not None

    def test_episodic_not_created_without_hierarchy(self):
        stream = MockStream(["50"] * 5)
        watcher = Watcher(stream=stream)
        assert watcher._episodic is None

    def test_episodic_records_high_surprise(self):
        # First value = surprise 1.0, novel values cause high surprise
        values = ["50"] * 10 + ["99"]
        stream = MockStream(values)
        watcher = Watcher(stream=stream, sequence_window=3)
        for _ in range(11):
            watcher.poll_and_observe()
        assert watcher._episodic is not None
        # Should have recorded at least one episode (first observation always 1.0)
        assert watcher._episodic.count >= 1

    def test_status_includes_episodic_info(self):
        values = ["50"] * 10 + ["99"]
        stream = MockStream(values)
        watcher = Watcher(stream=stream, sequence_window=3)
        for _ in range(11):
            watcher.poll_and_observe()
        status = watcher.get_status()
        assert status["episodic"] is not None
        assert "episode_count" in status["episodic"]
        assert "recent_episodes" in status["episodic"]


class TestWatcherIntrospection:
    def test_introspection_disabled_by_default(self):
        stream = MockStream(["50"] * 5)
        watcher = Watcher(stream=stream)
        assert watcher._self_model is None

    def test_introspection_enabled(self):
        stream = MockStream(["50"] * 20)
        watcher = Watcher(stream=stream, introspection=True)
        assert watcher._self_model is not None
        for _ in range(20):
            watcher.poll_and_observe()
        status = watcher.get_status()
        assert status["self_model"] is not None
        assert "vitals" in status["self_model"]
        assert "cognitive_surprise" in status["self_model"]
        assert "cognitive_patterns" in status["self_model"]

    def test_status_report_includes_cognitive_info(self):
        values = ["50"] * 40
        stream = MockStream(values)
        watcher = Watcher(
            stream=stream, introspection=True, report_interval=20
        )
        all_insights = []
        for _ in range(40):
            all_insights.extend(watcher.poll_and_observe())
        reports = [i for i in all_insights if i.kind == InsightKind.STATUS_REPORT]
        assert len(reports) >= 1
        assert "cognitive" in reports[-1].message

    def test_self_model_learns_stable_state(self):
        stream = MockStream(["50"] * 30)
        watcher = Watcher(stream=stream, introspection=True)
        for _ in range(30):
            watcher.poll_and_observe()
        assert watcher._self_model is not None
        # After stable stream, cognitive surprise should reduce
        assert watcher._self_model.cognitive_surprise < 1.0
