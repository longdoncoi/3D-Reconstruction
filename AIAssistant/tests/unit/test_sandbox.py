"""Unit tests for the policy-first command/write sandbox."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_assistant.tools.sandbox import create_directory, run, write_file


def _settings(**overrides) -> SimpleNamespace:
    base = {
        "sandbox_enabled": True,
        "sandbox_runtime": "local",
        "sandbox_image": "agent:latest",
        "max_write_bytes": 1024 * 1024,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class SandboxRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._settings = mock.patch("ai_assistant.tools.sandbox.load_agent_runtime_settings", return_value=_settings())
        self._settings.start()
        self.addCleanup(self._settings.stop)

    def test_disabled_policy(self) -> None:
        with mock.patch(
            "ai_assistant.tools.sandbox.load_agent_runtime_settings", return_value=_settings(sandbox_enabled=False)
        ):
            result = run("python --version", "C:/x", 10)
        self.assertIn("disabled", result["error"])

    def test_forbidden_metacharacters(self) -> None:
        for command in ("ls && rm -rf /", "echo hi | cat", "cmd > out.txt"):
            result = run(command, "C:/x", 10)
            self.assertIn("rejected", result["error"])

    @unittest.skipIf(os.name == "nt", "shlex does not enforce quotes with posix=False on Windows")
    def test_invalid_command_quoting(self) -> None:
        result = run('python "unterminated', "C:/x", 10)
        self.assertIn("Invalid command", result["error"])

    def test_unknown_executable(self) -> None:
        result = run("hacktool --all", "C:/x", 10)
        self.assertIn("allow-list", result["error"])

    def test_allowed_command_runs_local(self) -> None:
        with mock.patch(
            "ai_assistant.tools.sandbox.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ) as sub:
            result = run("python --version", "C:/workspace", 30)
        self.assertEqual(result["return_code"], 0)
        self.assertEqual(result["stdout"], "ok")
        self.assertEqual(sub.call_args.kwargs["cwd"], "C:/workspace")
        self.assertFalse(sub.call_args.kwargs["shell"])
        self.assertEqual(sub.call_args.kwargs["timeout"], 30)

    def test_timeout_returns_error(self) -> None:
        with mock.patch(
            "ai_assistant.tools.sandbox.subprocess.run", side_effect=__import__("subprocess").TimeoutExpired("cmd", 30)
        ):
            result = run("ctest --help", "C:/workspace", 30)
        self.assertIn("timed out", result["error"])

    def test_docker_runtime_builds_docker_argv(self) -> None:
        with mock.patch(
            "ai_assistant.tools.sandbox.load_agent_runtime_settings",
            return_value=_settings(sandbox_runtime="docker"),
        ):
            with mock.patch(
                "ai_assistant.tools.sandbox.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="x", stderr=""),
            ) as sub:
                result = run("cmake --version", "C:/workspace", 30)
        argv = sub.call_args.args[0]
        self.assertEqual(argv[0], "docker")
        self.assertIn("--network", argv)
        self.assertIn("none", argv)
        self.assertIn("agent:latest", argv)
        self.assertEqual(result["sandbox"], "docker")
        self.assertEqual(result["return_code"], 0)

    def test_inline_interpreter_code_is_rejected(self) -> None:
        """``python -c`` / ``-m`` must not turn the allow-list into an RCE."""
        for command in (
            "python -c print(1337)",
            "python -cprint(1337)",
            "python -m http.server 9000",
            "python -mhttp.server",
            "python -",
            "python --",
            "python -i",
            "python -O script.py",
        ):
            with self.subTest(command=command):
                with mock.patch("ai_assistant.tools.sandbox.subprocess.run") as sub:
                    result = run(command, "C:/workspace", 30)
                self.assertIn("Inline interpreter", result["error"])
                sub.assert_not_called()

    def test_script_and_version_invocations_still_allowed(self) -> None:
        for command in ("python script.py", "python --version", "python -V"):
            with self.subTest(command=command):
                with mock.patch(
                    "ai_assistant.tools.sandbox.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
                ) as sub:
                    result = run(command, "C:/workspace", 30)
                self.assertEqual(result["return_code"], 0)
                sub.assert_called_once()

    def test_shell_substitution_is_rejected(self) -> None:
        for command in ("git log $(whoami)", "git log `whoami`"):
            with self.subTest(command=command):
                result = run(command, "C:/workspace", 30)
                self.assertIn("rejected", result["error"])


class SandboxWriteFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self._settings = mock.patch("ai_assistant.tools.sandbox.load_agent_runtime_settings", return_value=_settings())
        self._settings.start()
        self.addCleanup(self._settings.stop)
        self._roots = mock.patch("ai_assistant.tools.sandbox.agent_write_roots", return_value=frozenset({"src"}))
        self._roots.start()
        self.addCleanup(self._roots.stop)
        (self.root / "src").mkdir(parents=True, exist_ok=True)

    def test_disabled_policy(self) -> None:
        with mock.patch(
            "ai_assistant.tools.sandbox.load_agent_runtime_settings", return_value=_settings(sandbox_enabled=False)
        ):
            result = write_file(str(self.root / "src" / "a.txt"), "x", str(self.root))
        self.assertIn("disabled", result["error"])

    def test_payload_too_large(self) -> None:
        with mock.patch(
            "ai_assistant.tools.sandbox.load_agent_runtime_settings", return_value=_settings(max_write_bytes=4)
        ):
            result = write_file(str(self.root / "src" / "a.txt"), "way too long", str(self.root))
        self.assertIn("exceeds", result["error"])

    def test_path_escaping_project(self) -> None:
        result = write_file(str(self.root.parent / "outside.txt"), "x", str(self.root))
        self.assertIn("escapes", result["error"])

    def test_path_not_in_write_roots(self) -> None:
        result = write_file(str(self.root / "notes" / "a.txt"), "x", str(self.root))
        self.assertIn("ALLOWLIST", result["error"])

    def test_success_writes_file(self) -> None:
        result = write_file(str(self.root / "src" / "out.txt"), "hello", str(self.root))
        self.assertTrue(result["success"])
        self.assertEqual(result["bytes_written"], 5)
        self.assertEqual((self.root / "src" / "out.txt").read_text(encoding="utf-8"), "hello")
        self.assertEqual(result["path"], "src/out.txt")

    def test_os_error_reported(self) -> None:
        with mock.patch("ai_assistant.tools.sandbox.os.replace", side_effect=OSError("disk full")):
            result = write_file(str(self.root / "src" / "out.txt"), "hello", str(self.root))
        self.assertIn("Unable to write", result["error"])


class SandboxCreateDirectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self._settings = mock.patch("ai_assistant.tools.sandbox.load_agent_runtime_settings", return_value=_settings())
        self._settings.start()
        self.addCleanup(self._settings.stop)
        self._roots = mock.patch("ai_assistant.tools.sandbox.agent_write_roots", return_value=frozenset({"src"}))
        self._roots.start()
        self.addCleanup(self._roots.stop)

    def test_disabled_policy(self) -> None:
        with mock.patch(
            "ai_assistant.tools.sandbox.load_agent_runtime_settings", return_value=_settings(sandbox_enabled=False)
        ):
            result = create_directory(str(self.root / "src" / "x"), str(self.root))
        self.assertIn("disabled", result["error"])

    def test_escape_rejected(self) -> None:
        result = create_directory(str(self.root.parent / "ev"), str(self.root))
        self.assertIn("escapes", result["error"])

    def test_not_allowlisted(self) -> None:
        result = create_directory(str(self.root / "tmp" / "x"), str(self.root))
        self.assertIn("ALLOWLIST", result["error"])

    def test_create_and_reuse(self) -> None:
        first = create_directory(str(self.root / "src" / "newdir"), str(self.root))
        self.assertTrue(first["success"])
        self.assertTrue(first["created"])
        self.assertTrue((self.root / "src" / "newdir").is_dir())
        second = create_directory(str(self.root / "src" / "newdir"), str(self.root))
        self.assertFalse(second["created"])

    def test_os_error_reported(self) -> None:
        with mock.patch("ai_assistant.tools.sandbox.Path.mkdir", side_effect=OSError("perm")):
            result = create_directory(str(self.root / "src" / "x"), str(self.root))
        self.assertIn("Unable to create", result["error"])


if __name__ == "__main__":
    unittest.main()
