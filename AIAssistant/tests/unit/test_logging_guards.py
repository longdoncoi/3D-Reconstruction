"""Guard-path coverage for logging setup and rotation hygiene.

``setup_logging`` rewires global state (root handlers, ``sys.stdout``), so
each test snapshots and restores it. The arms worth pinning are the exotic
streams: one with neither ``reconfigure`` nor ``buffer`` (pass-through), one
whose ``reconfigure`` raises (tolerated twice — in the wrapper and in the
handler setup), and the log-rotation cleanup that must survive files it
cannot delete, whether by age or by total size.
"""
from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_assistant.config.logging import (
    cleanup_old_logs,
    force_utf8_stream,
    setup_logging,
)


class _RaisingStream:
    """A stream whose ``reconfigure`` always fails (locked console, Qt pipe)."""

    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, text: str) -> None:
        self.writes.append(text)

    def flush(self) -> None:
        pass

    def reconfigure(self, *args, **kwargs) -> None:
        msg = "stream is locked"
        raise RuntimeError(msg)


class ForceUtf8StreamTests(unittest.TestCase):
    def test_a_stream_with_no_hooks_passes_through_untouched(self):
        stream = object()
        self.assertIs(force_utf8_stream(stream), stream)

    def test_a_failing_reconfigure_falls_back_to_the_original_stream(self):
        stream = _RaisingStream()
        self.assertIs(force_utf8_stream(stream), stream)

    def test_a_buffered_binary_stream_is_wrapped_as_utf8_text(self):
        class _Buffered:
            def __init__(self) -> None:
                self.buffer = io.BytesIO()

        wrapped = force_utf8_stream(_Buffered())
        self.assertIsInstance(wrapped, io.TextIOWrapper)
        self.assertEqual(wrapped.encoding, "utf-8")
        wrapped.detach()


class SetupLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        root = logging.getLogger()
        self._handlers = root.handlers[:]
        self._level = root.level
        self._stdout, self._stderr = sys.stdout, sys.stderr

    def tearDown(self) -> None:
        root = logging.getLogger()
        for handler in root.handlers:
            if handler not in self._handlers:
                handler.close()
        root.handlers.clear()
        root.handlers.extend(self._handlers)
        root.setLevel(self._level)
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def _close_new_handlers(self) -> None:
        """Release log files before the temporary directory is removed."""
        root = logging.getLogger()
        for handler in root.handlers:
            if handler not in self._handlers:
                handler.close()

    def test_a_bare_stream_is_used_as_is(self):
        """No ``reconfigure`` hook means no reconfigure attempt."""
        with tempfile.TemporaryDirectory() as logs:
            sys.stdout = object()
            logger, log_file = setup_logging(logs)
            self.assertTrue(str(log_file).endswith(".log"))
            self.assertIsNotNone(logger)
            self._close_new_handlers()

    def test_a_locked_stream_does_not_break_setup(self):
        """``reconfigure`` may fail in the wrapper and again on the handler."""
        with tempfile.TemporaryDirectory() as logs:
            sys.stdout = _RaisingStream()
            logger, _ = setup_logging(logs)
            logger.info("setup survived a locked stream")
            self._close_new_handlers()


class CleanupOldLogsTests(unittest.TestCase):
    def _old_log(self, directory: str, name: str = "server_old.log") -> Path:
        path = Path(directory) / name
        path.write_text("old", encoding="utf-8")
        decade_ago = time.time() - 10 * 86_400
        os.utime(path, (decade_ago, decade_ago))
        return path

    def test_an_undeletable_old_log_is_kept_not_crashed_on(self):
        with tempfile.TemporaryDirectory() as logs:
            path = self._old_log(logs)
            with patch.object(Path, "unlink", side_effect=OSError("locked")):
                cleanup_old_logs(logs)
            self.assertTrue(path.exists())

    def test_an_undeletable_fresh_log_is_kept_when_over_budget(self):
        with tempfile.TemporaryDirectory() as logs:
            path = Path(logs) / "server_fresh.log"
            path.write_text("x" * 1024, encoding="utf-8")
            with patch.object(Path, "unlink", side_effect=OSError("locked")):
                cleanup_old_logs(logs, max_size_mb=0)
            self.assertTrue(path.exists())

    def test_old_logs_are_removed_and_fresh_logs_kept(self):
        with tempfile.TemporaryDirectory() as logs:
            old = self._old_log(logs)
            fresh = Path(logs) / "server_fresh.log"
            fresh.write_text("new", encoding="utf-8")
            cleanup_old_logs(logs)
            self.assertFalse(old.exists())
            self.assertTrue(fresh.exists())

    def test_a_missing_directory_is_a_noop(self):
        with tempfile.TemporaryDirectory() as parent:
            cleanup_old_logs(Path(parent) / "no-such-dir")


if __name__ == "__main__":
    unittest.main()
