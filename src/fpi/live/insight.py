"""Structured anomaly reports from the intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InsightLevel(Enum):
    """Severity of an insight."""

    INFO = "info"
    ANOMALY = "anomaly"
    ALERT = "alert"


class InsightKind(Enum):
    """What generated this insight."""

    SURPRISE = "surprise"
    STATUS_REPORT = "status"


@dataclass(frozen=True)
class Insight:
    """A structured report of something the intelligence noticed.

    Attributes:
        stream_name: Which data stream triggered this.
        tick: The watcher's internal tick count.
        timestamp_seconds: Wall-clock time (time.time()).
        surprise: How unexpected (0.0 = predicted, 1.0 = maximally novel).
        level: Categorized importance.
        raw_value: The raw input string that triggered this.
        pattern_id: The pattern this observation matched.
        is_new_pattern: Whether this created a never-before-seen pattern.
        message: Human-readable description.
        vitality: Current watcher vitality.
        valence: Learned value of the matched pattern (positive = good).
        predicted_next: Pattern ID the agent expects next.
        kind: What generated this insight.
    """

    stream_name: str
    tick: int
    timestamp_seconds: float
    surprise: float
    level: InsightLevel
    raw_value: str
    pattern_id: int
    is_new_pattern: bool
    message: str
    vitality: float = 1.0
    valence: float | None = None
    predicted_next: int | None = None
    kind: InsightKind = InsightKind.SURPRISE


@dataclass(frozen=True)
class SocietyInsight:
    """Cross-stream correlation insight from the Society layer.

    Generated when the collective state of multiple watchers changes
    in a way the society didn't predict.
    """

    tick: int
    timestamp_seconds: float
    surprise: float
    level: InsightLevel
    watcher_states: dict[str, dict]
    message: str
    collective_vitality: float
