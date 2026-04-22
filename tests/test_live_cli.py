"""Tests for CLI argument parsing."""

from fpi.live.cli import build_parser


class TestCLIParser:
    def test_command_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--command", "echo hi", "--interval", "30"])
        assert args.command == ["echo hi"]
        assert args.interval == 30.0

    def test_multiple_commands(self):
        parser = build_parser()
        args = parser.parse_args(["--command", "uptime", "--command", "df -h"])
        assert len(args.command) == 2

    def test_file_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--file", "/var/log/app.log"])
        assert args.file == ["/var/log/app.log"]

    def test_multiple_files(self):
        parser = build_parser()
        args = parser.parse_args(["--file", "a.log", "--file", "b.log"])
        assert len(args.file) == 2

    def test_stdin_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--stdin"])
        assert args.stdin is True

    def test_surprise_threshold(self):
        parser = build_parser()
        args = parser.parse_args(["--command", "echo hi", "--surprise-threshold", "0.8"])
        assert args.surprise_threshold == 0.8

    def test_no_society_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--command", "echo hi", "--no-society"])
        assert args.no_society is True

    def test_quiet_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--command", "echo hi", "--quiet"])
        assert args.quiet is True

    def test_report_interval_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--command", "echo hi", "--report-interval", "50"])
        assert args.report_interval == 50

    def test_sequence_window_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--command", "echo hi", "--sequence-window", "3"])
        assert args.sequence_window == 3

    def test_temporal_scales_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--command", "echo hi", "--temporal-scales", "3,7,15"])
        assert args.temporal_scales == "3,7,15"

    def test_episodic_capacity_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--command", "echo hi", "--episodic-capacity", "100"])
        assert args.episodic_capacity == 100

    def test_introspection_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--command", "echo hi", "--introspection"])
        assert args.introspection is True

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--command", "echo hi"])
        assert args.interval == 60.0
        assert args.surprise_threshold == 0.5
        assert args.poll_interval == 1.0
        assert args.no_society is False
        assert args.quiet is False
        assert args.report_interval == 0
        assert args.sequence_window == 0
        assert args.temporal_scales == ""
        assert args.episodic_capacity == 50
        assert args.introspection is False
