"""Process-entry logic is unit-tested, not merely excluded (ADR 0015).

Both composition roots used to hide their whole ``__main__`` body behind a
coverage exclusion: ``bootstrap/server.py`` re-exec'd uvicorn through two
untestable try/except paths, and ``bootstrap/server_manager.py`` set up the
Qt console contract inline. The bodies are now named functions
(``server.main``, ``manager.configure_stdio_for_qt``/``manager.qt_entry``);
the guard that remains excluded is the one-line interpreter dispatch itself.

``uvicorn`` is faked through ``sys.modules`` (the same pattern the tracing
tests use for the optional OTel exporter): the suite never starts a server,
and the test job installs ``requirements-test.txt``, which does not pin
uvicorn — the behaviour under test is *our* control flow, not uvicorn's.
"""
from __future__ import annotations

import ctypes
import io
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

from ai_assistant.bootstrap import server as server_module
from ai_assistant.bootstrap import server_manager as manager_module


class ServerMainEntryTests(unittest.TestCase):
    """``server.main()`` warms the runtime first and never swallows a serve failure."""

    def test_warms_up_then_serves_the_app(self) -> None:
        fake_server = MagicMock()
        fake_uvicorn = MagicMock()
        fake_uvicorn.Server.return_value = fake_server
        with (
            patch.dict(sys.modules, {"uvicorn": fake_uvicorn}),
            patch.object(server_module, "bootstrap_runtime") as warmup,
            patch.object(server_module, "_stop_server_when_requested") as watch,
        ):
            server_module.main()

        warmup.assert_called_once_with()
        fake_uvicorn.Config.assert_called_once()
        self.assertIs(fake_uvicorn.Config.call_args.args[0], server_module.app)
        self.assertIsNone(fake_uvicorn.Config.call_args.kwargs["log_config"])
        self.assertFalse(fake_uvicorn.Config.call_args.kwargs["access_log"])
        fake_uvicorn.Server.assert_called_once_with(fake_uvicorn.Config.return_value)
        fake_server.run.assert_called_once_with()
        # The graceful-stop watcher is armed with the process graph's event.
        watch.assert_called_once_with(fake_server, server_module.runtime_modules.shutdown_requested)

    def test_runtime_failure_exits_one(self) -> None:
        """A server that cannot warm up must exit 1, not serve a broken facade."""
        fake_uvicorn = MagicMock()
        with (
            patch.dict(sys.modules, {"uvicorn": fake_uvicorn}),
            patch.object(server_module, "bootstrap_runtime", side_effect=RuntimeError("no model")),
        ):
            with self.assertRaises(SystemExit) as raised:
                server_module.main()

        self.assertEqual(raised.exception.code, 1)
        fake_uvicorn.Config.assert_not_called()
        fake_uvicorn.Server.assert_not_called()

    def test_serve_termination_is_logged_and_reraised(self) -> None:
        """BaseException from uvicorn (Ctrl-C, sys.exit) passes through after logging."""
        fake_server = MagicMock()
        fake_server.run.side_effect = KeyboardInterrupt
        fake_uvicorn = MagicMock()
        fake_uvicorn.Server.return_value = fake_server
        with (
            patch.dict(sys.modules, {"uvicorn": fake_uvicorn}),
            patch.object(server_module, "bootstrap_runtime"),
            patch.object(server_module, "_stop_server_when_requested"),
        ):
            with self.assertRaises(KeyboardInterrupt):
                server_module.main()

    def test_the_shutdown_watcher_sets_should_exit(self) -> None:
        """The admin event is what stops uvicorn; no ``os._exit`` in the path."""
        server = MagicMock()
        server.should_exit = False
        event = threading.Event()

        worker = threading.Thread(
            target=server_module._stop_server_when_requested,
            args=(server, event),
            daemon=True,
        )
        worker.start()
        try:
            self.assertFalse(server.should_exit)
            event.set()
        finally:
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertTrue(server.should_exit)


class QtEntryTests(unittest.TestCase):
    """The Qt host contract: UTF-8 stdout first, supervisor loop second."""

    def setUp(self) -> None:
        self._saved_utf8 = os.environ.get("PYTHONUTF8")
        self.addCleanup(self._restore_utf8)

    def _restore_utf8(self) -> None:
        if self._saved_utf8 is None:
            os.environ.pop("PYTHONUTF8", None)
        else:
            os.environ["PYTHONUTF8"] = self._saved_utf8

    def test_win32_sets_console_cp_and_reconfigures_stdout(self) -> None:
        windll = MagicMock()
        stdout = MagicMock()
        with (
            patch.object(sys, "platform", "win32"),
            patch.object(sys, "stdout", stdout),
            patch.object(ctypes, "windll", windll, create=True),
        ):
            manager_module.configure_stdio_for_qt()

        self.assertEqual(os.environ.get("PYTHONUTF8"), "1")
        windll.kernel32.SetConsoleOutputCP.assert_called_once_with(65001)
        stdout.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")

    def test_foreign_or_missing_ctypes_is_not_fatal(self) -> None:
        """A console without a CP API (or a raised OSError) must not stop startup."""
        windll = MagicMock()
        windll.kernel32.SetConsoleOutputCP.side_effect = OSError("no console")
        with (
            patch.object(sys, "platform", "win32"),
            patch.object(sys, "stdout", MagicMock()),
            patch.object(ctypes, "windll", windll, create=True),
        ):
            manager_module.configure_stdio_for_qt()

        self.assertEqual(os.environ.get("PYTHONUTF8"), "1")

    def test_posix_skips_the_console_cp(self) -> None:
        windll = MagicMock()
        stdout = MagicMock()
        with (
            patch.object(sys, "platform", "linux"),
            patch.object(sys, "stdout", stdout),
            patch.object(ctypes, "windll", windll, create=True),
        ):
            manager_module.configure_stdio_for_qt()

        windll.kernel32.SetConsoleOutputCP.assert_not_called()
        stdout.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")

    def test_stdout_without_reconfigure_is_left_alone(self) -> None:
        """Streams without ``reconfigure`` (StringIO, redirected pipes) are legal."""
        with (
            patch.object(sys, "platform", "linux"),
            patch.object(sys, "stdout", io.StringIO()),
        ):
            manager_module.configure_stdio_for_qt()

        self.assertEqual(os.environ.get("PYTHONUTF8"), "1")

    def test_stdout_reconfigure_failure_is_not_fatal(self) -> None:
        stdout = MagicMock()
        stdout.reconfigure.side_effect = ValueError("stream closed")
        with (
            patch.object(sys, "platform", "linux"),
            patch.object(sys, "stdout", stdout),
        ):
            manager_module.configure_stdio_for_qt()

        stdout.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")

    def test_qt_entry_sets_up_stdio_then_runs_the_supervisor(self) -> None:
        order: list[str] = []
        with (
            patch.object(
                manager_module, "configure_stdio_for_qt", side_effect=lambda: order.append("stdio")
            ),
            patch.object(manager_module, "main", side_effect=lambda: order.append("main")),
        ):
            manager_module.qt_entry()

        self.assertEqual(order, ["stdio", "main"])
