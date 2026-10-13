"""Unit tests for the Qt-facing server-manager supervisor script."""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Imported as a package submodule (not via ``spec_from_file_location`` with a
# bare module name) so coverage attributes the traced lines to the source file.
from ai_assistant.bootstrap import server_manager


class _Utf8Stdout:
    """Reconfigure ``sys.stdout`` to UTF-8 so Vietnamese log lines survive.

    ``server_manager`` prints Vietnamese diagnostics; the ``__main__`` block
    reconfigures stdout, but importing it as a module (as these tests do) does
    not, so a cp1252 console would raise ``UnicodeEncodeError``.
    """

    def __enter__(self) -> None:
        self._previous = sys.stdout
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    def __exit__(self, *exc) -> None:
        sys.stdout = self._previous


class _FakeResponse:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status = status
        self._payload = payload or {}

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeLogFile:
    """File-like double for ``tail_log_forever`` with scripted reads."""

    def __init__(self, reads: list) -> None:
        self._reads = list(reads)
        self.seek_calls: list[tuple] = []
        self.closed = False

    def seek(self, *args) -> None:
        self.seek_calls.append(args)

    def readline(self) -> str:
        item = self._reads.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        self.closed = True
        return False


class ServerManagerTests(unittest.TestCase):
    def test_module_uses_the_canonical_bind_address(self) -> None:
        """Host/port come from settings, not scattered literals (ADR 0001)."""
        from ai_assistant.settings import DEFAULT_HOST, DEFAULT_PORT

        self.assertEqual(server_manager.HOST, DEFAULT_HOST)
        self.assertEqual(server_manager.PORT, DEFAULT_PORT)

    def test_is_server_running_true_on_health_200(self) -> None:
        with patch.object(server_manager.urllib.request, "urlopen", return_value=_FakeResponse(200)):
            self.assertTrue(server_manager.is_server_running())

    def test_is_server_running_false_on_error(self) -> None:
        with patch.object(server_manager.urllib.request, "urlopen", side_effect=OSError("down")):
            self.assertFalse(server_manager.is_server_running())

    def test_is_server_running_false_on_non_200_status(self) -> None:
        with patch.object(server_manager.urllib.request, "urlopen", return_value=_FakeResponse(503)):
            self.assertFalse(server_manager.is_server_running())

    def test_get_active_model_index_parses_health_payload(self) -> None:
        with patch.object(server_manager.urllib.request, "urlopen", return_value=_FakeResponse(200, {"model_idx": 3})):
            self.assertEqual(server_manager.get_active_model_index(), 3)

    def test_get_active_model_index_none_on_error(self) -> None:
        with patch.object(server_manager.urllib.request, "urlopen", side_effect=OSError("down")):
            self.assertIsNone(server_manager.get_active_model_index())

    def test_get_latest_log_file_none_when_dir_missing(self) -> None:
        with patch.object(server_manager.os.path, "isdir", return_value=False):
            self.assertIsNone(server_manager.get_latest_log_file())

    def test_get_latest_log_file_returns_newest_log(self) -> None:
        with tempfile.TemporaryDirectory() as logs_dir:
            older = os.path.join(logs_dir, "a.log")
            newer = os.path.join(logs_dir, "b.log")
            Path(older).write_text("old", encoding="utf-8")
            Path(newer).write_text("new", encoding="utf-8")
            os.utime(older, (1_000_000, 1_000_000))
            os.utime(newer, (2_000_000, 2_000_000))
            with patch.object(server_manager, "LOGS_DIR", logs_dir):
                self.assertEqual(server_manager.get_latest_log_file(), newer)

    def test_get_latest_log_file_none_when_no_logs(self) -> None:
        with tempfile.TemporaryDirectory() as logs_dir:
            (Path(logs_dir) / "notes.txt").write_text("x", encoding="utf-8")
            with patch.object(server_manager, "LOGS_DIR", logs_dir):
                self.assertIsNone(server_manager.get_latest_log_file())

    def test_shutdown_running_server_posts_to_admin_shutdown(self) -> None:
        with patch.object(server_manager.urllib.request, "urlopen", return_value=_FakeResponse(200, {"ok": True})):
            self.assertTrue(server_manager.shutdown_running_server())

    def test_shutdown_running_server_false_on_error(self) -> None:
        with _Utf8Stdout(), patch.object(server_manager.urllib.request, "urlopen", side_effect=OSError("refused")):
            self.assertFalse(server_manager.shutdown_running_server())

    def test_wait_until_ready_returns_true_when_healthy(self) -> None:
        with patch.object(server_manager, "is_server_running", return_value=True):
            self.assertTrue(server_manager.wait_until_ready())

    def test_wait_until_ready_false_when_process_exits(self) -> None:
        process = MagicMock()
        process.poll.return_value = 1
        with _Utf8Stdout(), patch.object(server_manager, "is_server_running", return_value=False):
            self.assertFalse(server_manager.wait_until_ready(process))

    def test_wait_until_ready_false_on_timeout(self) -> None:
        with patch.object(server_manager, "is_server_running", return_value=False), \
                patch.object(server_manager.time, "monotonic", side_effect=[0.0, 0.0, 1e9]), \
                patch.object(server_manager.time, "sleep"):
            self.assertFalse(server_manager.wait_until_ready())

    def test_wait_until_ready_prints_progress_every_ten_seconds(self) -> None:
        process = MagicMock()
        process.poll.return_value = None
        with patch.object(server_manager, "is_server_running", return_value=False), \
                patch.object(server_manager.time, "monotonic", side_effect=[0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]), \
                patch.object(server_manager.time, "sleep"), \
                patch("builtins.print") as mock_print:
            self.assertFalse(server_manager.wait_until_ready(process))
        texts = [str(c.args[0]) for c in mock_print.call_args_list]
        self.assertTrue(any("(10/600s)" in t for t in texts))

    def test_detached_kwargs_use_creationflags_only_on_windows(self) -> None:
        """``creationflags`` is a Windows-only ``Popen`` kwarg.

        Passing a non-zero value on POSIX raises ``ValueError``, so the flag
        set must be chosen per platform rather than passed unconditionally.
        """
        with patch.object(server_manager.sys, "platform", "win32"):
            kwargs = server_manager._detached_popen_kwargs()
        self.assertEqual(
            kwargs,
            {"creationflags": 0x00000008 | 0x00000200},
        )

    def test_detached_kwargs_use_a_new_session_on_posix(self) -> None:
        with patch.object(server_manager.sys, "platform", "linux"):
            kwargs = server_manager._detached_popen_kwargs()
        self.assertEqual(kwargs, {"start_new_session": True})
        self.assertNotIn("creationflags", kwargs)

    def test_tail_log_forever_streams_lines_then_waits(self) -> None:
        """Lines are echoed verbatim; an empty read falls back to polling."""
        fake = _FakeLogFile(["", "boom\n", KeyboardInterrupt()])
        with patch("builtins.print") as mock_print, \
                patch("builtins.open", return_value=fake), \
                patch.object(server_manager.time, "sleep"):
            with self.assertRaises(KeyboardInterrupt):
                server_manager.tail_log_forever("logs/ai.log", from_start=True)
        texts = [str(c.args[0]) for c in mock_print.call_args_list]
        self.assertIn("Tailing log:", texts[0])
        self.assertTrue(any("boom" in t for t in texts))
        self.assertEqual(fake.seek_calls, [])

    def test_tail_log_forever_seeks_to_end_when_not_from_start(self) -> None:
        """``from_start=False`` replays no history (Qt shows live output only)."""
        fake = _FakeLogFile([KeyboardInterrupt()])
        with patch("builtins.print") as mock_print, \
                patch("builtins.open", return_value=fake), \
                patch.object(server_manager.time, "sleep"):
            with self.assertRaises(KeyboardInterrupt):
                server_manager.tail_log_forever("logs/ai.log", from_start=False)
        texts = [str(c.args[0]) for c in mock_print.call_args_list]
        self.assertIn("Tailing log:", texts[0])
        self.assertEqual(fake.seek_calls, [(0, os.SEEK_END)])

    # ── main() flow ──────────────────────────────────────────────────────────
    def test_main_uses_running_server_and_tails_log(self) -> None:
        """Same model already active: no restart; the log tail is attached."""
        with patch("builtins.print") as mock_print, \
                patch.object(server_manager.sys, "argv", ["server_manager", "not-a-number"]), \
                patch.object(server_manager, "is_server_running", return_value=True), \
                patch.object(server_manager, "get_active_model_index", return_value=0), \
                patch.object(server_manager, "get_latest_log_file", return_value="logs/ai.log"), \
                patch.object(server_manager, "tail_log_forever", side_effect=KeyboardInterrupt):
            server_manager.main()  # must not raise: interrupt is swallowed
        texts = [str(c.args[0]) for c in mock_print.call_args_list]
        self.assertTrue(any("SUCCESS] AI Server started" in t for t in texts))
        self.assertTrue(any("Ngừng theo dõi log" in t for t in texts))

    def test_main_restarts_when_active_model_is_unknown(self) -> None:
        """Server running but health has no model: shut down and start again."""
        with patch("builtins.print"), \
                patch.object(server_manager.sys, "argv", ["server_manager", "2"]), \
                patch.object(server_manager, "is_server_running", side_effect=[True, False, False]), \
                patch.object(server_manager, "get_active_model_index", return_value=None), \
                patch.object(server_manager, "shutdown_running_server") as mock_shutdown, \
                patch.object(server_manager.time, "monotonic", side_effect=[0.0, 30.0]), \
                patch.object(server_manager.subprocess, "Popen", return_value=MagicMock()), \
                patch.object(server_manager, "wait_until_ready", return_value=True), \
                patch.object(server_manager, "get_latest_log_file", return_value="logs/ai.log"), \
                patch.object(server_manager, "tail_log_forever", side_effect=KeyboardInterrupt):
            server_manager.main()
        mock_shutdown.assert_called_once()

    def test_main_warns_when_server_survives_shutdown(self) -> None:
        """The old server ignores shutdown, but the new one is started anyway."""
        with patch("builtins.print") as mock_print, \
                patch.object(server_manager.sys, "argv", ["server_manager", "2"]), \
                patch.object(server_manager, "is_server_running", side_effect=[True, True, False, True]), \
                patch.object(server_manager, "get_active_model_index", return_value=0), \
                patch.object(server_manager, "shutdown_running_server"), \
                patch.object(server_manager.time, "monotonic", return_value=0.0), \
                patch.object(server_manager.time, "sleep"), \
                patch.object(server_manager.subprocess, "Popen", return_value=MagicMock()), \
                patch.object(server_manager, "wait_until_ready", return_value=True), \
                patch.object(server_manager, "get_latest_log_file", return_value="logs/ai.log"), \
                patch.object(server_manager, "tail_log_forever", side_effect=KeyboardInterrupt):
            server_manager.main()
        texts = [str(c.args[0]) for c in mock_print.call_args_list]
        self.assertTrue(any("CẢNH BÁO" in t for t in texts))

    def test_main_starts_server_and_reports_missing_log(self) -> None:
        """No server yet: spawn detached, wait, then keep the pipe open."""
        with patch("builtins.print") as mock_print, \
                patch.object(server_manager.sys, "argv", ["server_manager"]), \
                patch.object(server_manager, "is_server_running", return_value=False), \
                patch.object(server_manager.subprocess, "Popen") as mock_popen, \
                patch.object(server_manager, "wait_until_ready", return_value=True), \
                patch.object(server_manager, "get_latest_log_file", return_value=None):
            calls = {"n": 0}

            def sleep(*_args):
                calls["n"] += 1
                if calls["n"] > 15:
                    raise KeyboardInterrupt

            with patch.object(server_manager.time, "sleep", side_effect=sleep):
                server_manager.main()
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        self.assertEqual(args[0][0], sys.executable)
        self.assertEqual(args[0][-1], "0")
        self.assertEqual(kwargs["cwd"], server_manager.BASE_DIR)
        texts = [str(c.args[0]) for c in mock_print.call_args_list]
        self.assertTrue(any("Không tìm thấy file log" in t for t in texts))

    def test_main_reports_timeout_and_returns(self) -> None:
        """Server never becomes ready: no log tailing is attempted."""
        with patch("builtins.print") as mock_print, \
                patch.object(server_manager.sys, "argv", ["server_manager", "5"]), \
                patch.object(server_manager, "is_server_running", return_value=False), \
                patch.object(server_manager.subprocess, "Popen", return_value=MagicMock()), \
                patch.object(server_manager, "wait_until_ready", return_value=False), \
                patch.object(server_manager, "get_latest_log_file") as mock_log_file:
            server_manager.main()
        mock_log_file.assert_not_called()
        texts = [str(c.args[0]) for c in mock_print.call_args_list]
        self.assertTrue(any("Lỗi: Server không phản hồi" in t for t in texts))

    def test_main_reads_app_data_dir_env_for_logs(self) -> None:
        """The Qt host passes ``APP_DATA_DIR``; it must win over the default."""
        with patch.dict(os.environ, {"APP_DATA_DIR": "fake-appdata"}):
            importlib.reload(server_manager)
        self.assertEqual(
            server_manager.LOGS_DIR,
            os.path.join("fake-appdata", "AIAssistant", "logs"),
        )
        # Restore the defaults for the rest of the session.
        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(server_manager)
        self.assertEqual(
            server_manager.LOGS_DIR,
            os.path.join(server_manager.BASE_DIR, "logs"),
        )


if __name__ == "__main__":
    unittest.main()
