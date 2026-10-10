"""Unit tests for logging setup, cleanup and agent loggers."""
from __future__ import annotations

import io
import logging
import logging.handlers
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_assistant.config.logging import cleanup_old_logs, force_utf8_stream, get_agent_logger, setup_logging


class _ReconfigurableStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.reconfigured = False

    def reconfigure(self, encoding=None, errors=None) -> None:
        self.reconfigured = True


class _BufferStream:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


class ForceUtf8StreamTests(unittest.TestCase):
    def test_reconfigure_used_when_available(self) -> None:
        stream = _ReconfigurableStream()
        result = force_utf8_stream(stream)
        self.assertIs(result, stream)
        self.assertTrue(stream.reconfigured)

    def test_wraps_buffer(self) -> None:
        stream = _BufferStream()
        result = force_utf8_stream(stream)
        self.assertIsInstance(result, io.TextIOWrapper)

    def test_unwrappable_returned_unchanged(self) -> None:
        stream = _ReconfigurableStream()
        with mock.patch.object(stream, "reconfigure", side_effect=RuntimeError):
            self.assertIs(force_utf8_stream(stream), stream)


class SetupLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = logging.getLogger()
        self._handlers = list(self.root.handlers)
        self._level = self.root.level
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def tearDown(self) -> None:
        self.root.handlers = self._handlers
        self.root.setLevel(self._level)
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def test_creates_log_file_and_handlers(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            logger, log_path = setup_logging(tmp)
            self.assertEqual(str(log_path.parent), tmp)
            self.assertTrue(Path(log_path).exists())
            handler_types = {type(h) for h in self.root.handlers}
            self.assertIn(logging.StreamHandler, handler_types)
            self.assertIn(logging.handlers.RotatingFileHandler, handler_types)
            self.assertEqual(logger.name, "chatbot_server")
            self.assertEqual(os.environ.get("PYTHONUTF8"), "1")
            # Release the log file before the temp dir is removed on Windows.
            for handler in list(self.root.handlers):
                handler.close() if hasattr(handler, "close") else None
                self.root.removeHandler(handler)
            self.root.handlers = self._handlers


class CleanupOldLogsTests(unittest.TestCase):
    def test_removes_age_expired_logs(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old.log"
            new = Path(tmp) / "new.log"
            old.write_text("x" * 100, encoding="utf-8")
            new.write_text("x" * 100, encoding="utf-8")
            old_time = time.time() - 8 * 86_400
            os.utime(old, (old_time, old_time))
            cleanup_old_logs(tmp, max_days=7)
            self.assertFalse(old.exists())
            self.assertTrue(new.exists())

    def test_non_existing_dir_is_noop(self) -> None:
        cleanup_old_logs("C:/definitely/not/here", max_days=7)  # must not raise

    def test_size_cap_removes_oldest(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.log", "b.log", "c.log"):
                (Path(tmp) / name).write_text("y" * 50_000, encoding="utf-8")
            cleanup_old_logs(tmp, max_days=7, max_size_mb=0.02)
            remaining = list(Path(tmp).glob("*.log"))
            self.assertLess(len(remaining), 3)
            # The oldest file must have been removed first.
            self.assertNotIn("a.log", {p.name for p in remaining})


class GetAgentLoggerTests(unittest.TestCase):
    @staticmethod
    def _close_agent_loggers() -> None:
        # Release OS handles before TemporaryDirectory cleanup on Windows.
        for logger_name in list(logging.Logger.manager.loggerDict):
            if not logger_name.startswith("agent."):
                continue
            logger = logging.getLogger(logger_name)
            for handler in list(logger.handlers):
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    try:
                        handler.close()
                    except Exception:
                        pass
                logger.removeHandler(handler)

    def test_creates_file_handler_once(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            try:
                with mock.patch("ai_assistant.config.paths.get_paths", return_value=SimpleNamespace(logs_dir=Path(tmp))):
                    logger = get_agent_logger("My Agent")
                    self.assertIn("agent.", logger.name)
                    self.assertTrue(logger.handlers)
                    cloned = get_agent_logger("My Agent")
                    self.assertEqual(cloned.handlers, logger.handlers)  # reused, not duplicated
                    self.assertTrue(any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers))
            finally:
                self._close_agent_loggers()

    def test_safe_name_normalisation(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            try:
                with mock.patch("ai_assistant.config.paths.get_paths", return_value=SimpleNamespace(logs_dir=Path(tmp))):
                    logger = get_agent_logger("Preprocess&Render")
                    self.assertIn("preprocess_render", logger.name)
            finally:
                self._close_agent_loggers()


if __name__ == "__main__":
    unittest.main()
