"""Unit tests for the Qt-facing server-manager supervisor script."""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_spec = importlib.util.spec_from_file_location(
    "server_manager", _SRC / "ai_assistant" / "bootstrap" / "server_manager.py"
)
server_manager = importlib.util.module_from_spec(_spec)
if _spec.loader is None:  # pragma: no cover — spec_from_file_location always sets a loader for a real file
    msg = "could not load the server-manager entry point for testing"
    raise RuntimeError(msg)
_spec.loader.exec_module(server_manager)


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

    def test_get_active_model_index_parses_health_payload(self) -> None:
        with patch.object(server_manager.urllib.request, "urlopen", return_value=_FakeResponse(200, {"model_idx": 3})):
            self.assertEqual(server_manager.get_active_model_index(), 3)

    def test_get_active_model_index_none_on_error(self) -> None:
        with patch.object(server_manager.urllib.request, "urlopen", side_effect=OSError("down")):
            self.assertIsNone(server_manager.get_active_model_index())

    def test_get_latest_log_file_none_when_dir_missing(self) -> None:
        with patch.object(server_manager.os.path, "isdir", return_value=False):
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
                patch.object(server_manager.time, "monotonic", side_effect=[0.0, 1e9]):
            self.assertFalse(server_manager.wait_until_ready())

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


if __name__ == "__main__":
    unittest.main()
