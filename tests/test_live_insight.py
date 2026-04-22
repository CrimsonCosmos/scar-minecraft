"""Tests for enriched Insight dataclass."""

from fpi.live.insight import Insight, InsightKind, InsightLevel


class TestInsightEnrichment:
    def test_default_kind_is_surprise(self):
        insight = Insight(
            stream_name="test", tick=0, timestamp_seconds=0.0,
            surprise=1.0, level=InsightLevel.INFO, raw_value="x",
            pattern_id=0, is_new_pattern=True, message="test",
        )
        assert insight.kind == InsightKind.SURPRISE

    def test_default_valence_is_none(self):
        insight = Insight(
            stream_name="test", tick=0, timestamp_seconds=0.0,
            surprise=1.0, level=InsightLevel.INFO, raw_value="x",
            pattern_id=0, is_new_pattern=True, message="test",
        )
        assert insight.valence is None

    def test_default_predicted_next_is_none(self):
        insight = Insight(
            stream_name="test", tick=0, timestamp_seconds=0.0,
            surprise=1.0, level=InsightLevel.INFO, raw_value="x",
            pattern_id=0, is_new_pattern=True, message="test",
        )
        assert insight.predicted_next is None

    def test_insight_with_enrichment(self):
        insight = Insight(
            stream_name="test", tick=0, timestamp_seconds=0.0,
            surprise=0.5, level=InsightLevel.ANOMALY, raw_value="x",
            pattern_id=0, is_new_pattern=False, message="test",
            valence=0.3, predicted_next=5,
            kind=InsightKind.STATUS_REPORT,
        )
        assert insight.valence == 0.3
        assert insight.predicted_next == 5
        assert insight.kind == InsightKind.STATUS_REPORT

    def test_insight_kind_values(self):
        assert InsightKind.SURPRISE.value == "surprise"
        assert InsightKind.STATUS_REPORT.value == "status"
