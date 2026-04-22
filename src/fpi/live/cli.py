"""CLI entry point for fpi watch.

Usage:
    fpi watch --command "df -h / | awk 'NR==2{print \\$5}'" --interval 60
    fpi watch --file /var/log/app.log
    fpi watch --command "uptime" --interval 30 --file access.log
    sensor_readings | fpi watch --stdin
"""

from __future__ import annotations

import argparse
import sys
import time

from .insight import Insight, InsightKind, InsightLevel, SocietyInsight
from .monitor import Monitor
from .stream import CommandStream, FileStream, StdinStream
from .watcher import Watcher


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for `fpi watch`."""
    parser = argparse.ArgumentParser(
        prog="fpi watch",
        description="Watch data streams with FP Intelligence",
    )
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="Shell command to run periodically (can specify multiple)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between command executions (default: 60)",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="File to tail (can specify multiple)",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read from stdin",
    )
    parser.add_argument(
        "--surprise-threshold",
        type=float,
        default=0.5,
        help="Minimum surprise to report (default: 0.5)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between poll cycles (default: 1.0)",
    )
    parser.add_argument(
        "--no-society",
        action="store_true",
        help="Disable society layer even with multiple streams",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print anomalies and alerts",
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=0,
        help="Emit status report every N ticks (0=disabled, default: 0)",
    )
    parser.add_argument(
        "--sequence-window",
        type=int,
        default=0,
        help="Enable sequence detection with N-pattern windows (0=disabled, default: 0)",
    )
    parser.add_argument(
        "--temporal-scales",
        type=str,
        default="",
        help="Comma-separated window sizes for multi-scale temporal hierarchy (e.g. '3,7,15')",
    )
    parser.add_argument(
        "--episodic-capacity",
        type=int,
        default=50,
        help="Maximum episodes in episodic memory (default: 50)",
    )
    parser.add_argument(
        "--introspection",
        action="store_true",
        help="Enable self-model (agent observes its own cognitive state)",
    )
    return parser


def print_insight(insight: Insight | SocietyInsight, quiet: bool = False) -> None:
    """Format and print an insight to the terminal."""
    if isinstance(insight, SocietyInsight):
        prefix = "[SOCIETY]"
        color = "\033[35m"  # Magenta
    elif not isinstance(insight, SocietyInsight) and insight.kind == InsightKind.STATUS_REPORT:
        if quiet:
            return
        prefix = "[STATUS]"
        color = "\033[32m"  # Green
    elif insight.level == InsightLevel.ALERT:
        prefix = "[ALERT]"
        color = "\033[31m"  # Red
    elif insight.level == InsightLevel.ANOMALY:
        prefix = "[ANOMALY]"
        color = "\033[33m"  # Yellow
    else:
        if quiet:
            return
        prefix = "[INFO]"
        color = "\033[36m"  # Cyan

    reset = "\033[0m"
    ts = time.strftime("%H:%M:%S")

    if isinstance(insight, SocietyInsight):
        print(f"{color}{ts} {prefix} {insight.message}{reset}")
    else:
        print(
            f"{color}{ts} {prefix} [{insight.stream_name}] "
            f"{insight.message}{reset}"
        )


def main(argv: list[str] | None = None) -> None:
    """Entry point for `fpi watch`."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Parse temporal scales
    temporal_scales: tuple[int, ...] | None = None
    if args.temporal_scales:
        temporal_scales = tuple(int(s.strip()) for s in args.temporal_scales.split(","))

    # Common watcher kwargs
    watcher_kwargs = {
        "surprise_threshold": args.surprise_threshold,
        "report_interval": args.report_interval,
        "sequence_window": args.sequence_window,
        "temporal_scales": temporal_scales,
        "episodic_capacity": args.episodic_capacity,
        "introspection": args.introspection,
    }

    # Build watchers
    watchers: list[Watcher] = []

    for cmd in args.command:
        stream = CommandStream(cmd, interval_seconds=args.interval)
        watcher = Watcher(stream=stream, **watcher_kwargs)
        watchers.append(watcher)

    for path in args.file:
        stream = FileStream(path)
        watcher = Watcher(stream=stream, **watcher_kwargs)
        watchers.append(watcher)

    if args.stdin:
        stream = StdinStream()
        watcher = Watcher(stream=stream, **watcher_kwargs)
        watchers.append(watcher)

    if not watchers:
        parser.error("At least one stream required: --command, --file, or --stdin")

    # Create monitor
    monitor = Monitor(
        watchers=watchers,
        enable_society=not args.no_society,
        poll_interval=args.poll_interval,
    )

    # Print header
    print(f"FP Intelligence watching {len(watchers)} stream(s)")
    for w in watchers:
        print(f"  - {w.name}")
    if len(watchers) > 1 and not args.no_society:
        print("  Society layer active (cross-stream correlation)")
    print(f"  Surprise threshold: {args.surprise_threshold}")
    print("  Press Ctrl+C to stop")
    print()

    # Run
    quiet = args.quiet
    monitor.run(callback=lambda insights: [print_insight(i, quiet) for i in insights])
