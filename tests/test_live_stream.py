"""Tests for data stream sources."""

import time

from fpi.live.stream import CommandStream, FileStream, StdinStream


class TestCommandStream:
    def test_poll_returns_output(self):
        stream = CommandStream("echo hello", interval_seconds=0)
        result = stream.poll()
        assert result == ["hello"]

    def test_interval_respected(self):
        stream = CommandStream("echo hello", interval_seconds=100)
        stream.poll()  # First call runs immediately
        result = stream.poll()  # Second call too soon
        assert result == []

    def test_name(self):
        stream = CommandStream("df -h / | awk 'NR==2{print $5}'")
        assert stream.name.startswith("cmd:")

    def test_failed_command_returns_empty(self):
        stream = CommandStream("false", interval_seconds=0)
        result = stream.poll()
        assert result == []

    def test_multiline_output(self):
        """Command output is returned as a single string."""
        stream = CommandStream("echo 'line1\nline2'", interval_seconds=0)
        result = stream.poll()
        assert len(result) == 1  # Entire output as one observation


class TestFileStream:
    def test_reads_new_lines(self, tmp_path):
        """Should only read lines written after opening."""
        f = tmp_path / "test.log"
        f.write_text("old line\n")
        stream = FileStream(str(f))
        assert stream.poll() == []  # Seeks to end on open

        with open(f, "a") as fh:
            fh.write("new line\n")
        result = stream.poll()
        assert result == ["new line"]
        stream.close()

    def test_multiple_new_lines(self, tmp_path):
        f = tmp_path / "test.log"
        f.write_text("")
        stream = FileStream(str(f))

        with open(f, "a") as fh:
            fh.write("line1\nline2\nline3\n")
        result = stream.poll()
        assert result == ["line1", "line2", "line3"]
        stream.close()

    def test_name(self, tmp_path):
        f = tmp_path / "app.log"
        f.write_text("")
        stream = FileStream(str(f))
        assert stream.name == "file:app.log"
        stream.close()

    def test_nonexistent_file(self):
        stream = FileStream("/tmp/nonexistent_fpi_test_file.log")
        assert stream.poll() == []

    def test_empty_lines_skipped(self, tmp_path):
        f = tmp_path / "test.log"
        f.write_text("")
        stream = FileStream(str(f))

        with open(f, "a") as fh:
            fh.write("data\n\n\n")
        result = stream.poll()
        assert result == ["data"]
        stream.close()


class TestStdinStream:
    def test_name(self):
        stream = StdinStream()
        assert stream.name == "stdin"
