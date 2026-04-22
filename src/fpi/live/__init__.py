"""Live monitoring — real-world interface for FP Intelligence.

Watch any data stream. Learn what's normal. Alert when something is unusual.
When watching multiple streams, a Society emerges that detects cross-stream
correlations invisible to any single watcher.

Usage:
    fpi watch --command "df -h /" --interval 60
    fpi watch --file /var/log/app.log
    fpi watch --command "uptime" --interval 30 --file access.log
    sensor_readings | fpi watch --stdin
"""

from .bridge import WatcherBridge
from .encoder import AutoEncoder, NumericEncoder, TextEncoder
from .insight import Insight, InsightKind, InsightLevel, SocietyInsight
from .monitor import Monitor
from .stream import CommandStream, FileStream, StdinStream
from .watcher import Watcher, WatcherStepResult

__all__ = [
    "AutoEncoder",
    "CommandStream",
    "FileStream",
    "Insight",
    "InsightKind",
    "InsightLevel",
    "Monitor",
    "NumericEncoder",
    "SocietyInsight",
    "StdinStream",
    "TextEncoder",
    "Watcher",
    "WatcherBridge",
    "WatcherStepResult",
]
