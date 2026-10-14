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

    def test_a_blank_command_is_refused(self) -> None:
        for command in ("", "   "):
            with self.subTest(command=command):
                self.assertIn("allow-list", run(command, "C:/x", 10)["error"])

    def test_a_path_shaped_executable_is_refused_even_when_the_name_matches(self) -> None:
        """The allow-list constrains *which* program runs, not what a file is
        called: matching on the basename let ``/opt/attacker/git`` pass for
        ``git``, so only a bare name resolved from PATH is accepted."""
        with mock.patch("ai_assistant.tools.sandbox.subprocess.run") as sub:
            result = run("/opt/attacker/git status", "C:/workspace", 30)
        # The path gate answers — not the name comparison that sits behind it.
        self.assertIn("bare name", result["error"])
        sub.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "backslashes survive shlex only with posix=False")
    def test_a_windows_path_shaped_executable_is_refused(self) -> None:
        with mock.patch("ai_assistant.tools.sandbox.subprocess.run") as sub:
            result = run("C:\\attacker\\git status", "C:/workspace", 30)
        self.assertIn("bare name", result["error"])
        sub.assert_not_called()

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

    def test_docker_timeout_force_removes_the_container(self) -> None:
        """Killing the ``docker run`` client does NOT stop the container.

        ADR 0011: the Docker runtime must be as bounded as the local one, so a
        timeout triggers a follow-up ``docker rm -f`` of the named container.
        """
        import subprocess as sp

        with mock.patch(
            "ai_assistant.tools.sandbox.load_agent_runtime_settings",
            return_value=_settings(sandbox_runtime="docker"),
        ):
            responses = [
                sp.TimeoutExpired("docker run", 30),
                SimpleNamespace(returncode=0, stdout="", stderr=""),
            ]
            with mock.patch("ai_assistant.tools.sandbox.subprocess.run", side_effect=responses) as sub:
                result = run("cmake --version", "C:/workspace", 30)

        self.assertIn("timed out", result["error"])
        self.assertEqual(sub.call_count, 2)
        run_argv = sub.call_args_list[0].args[0]
        name = run_argv[run_argv.index("--name") + 1]
        self.assertTrue(name.startswith("agent-sbx-"))
        rm_argv = sub.call_args_list[1].args[0]
        self.assertEqual(rm_argv[:3], ["docker", "rm", "-f"])
        self.assertEqual(rm_argv[3], name)

    def test_container_removal_failure_stays_silent(self) -> None:
        """No docker daemon (or a vanished container) must not mask the timeout."""
        from ai_assistant.tools.sandbox import _force_remove_container

        with mock.patch(
            "ai_assistant.tools.sandbox.subprocess.run", side_effect=OSError("no docker"),
        ) as sub:
            self.assertIsNone(_force_remove_container("agent-sbx-gone"))  # must not raise
        # The removal really was attempted — the swallow covers its failure,
        # not a missing call.
        sub.assert_called_once_with(
            ["docker", "rm", "-f", "agent-sbx-gone"],
            capture_output=True, timeout=15, shell=False, check=False,
        )

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

    def test_cmake_escape_subcommands_are_blocked(self) -> None:
        """ADR 0011: ``-E`` itself is allow-listed, so it needs its own gate.

        ``cmake -E env <cmd>`` executed arbitrary commands: naming the
        executable is not an allow-list if a flag can re-open it.
        """
        for command in (
            "cmake -E env python -c print(1)",
            "cmake -E chdir .. git push origin main",
            "cmake -E remove_directory build",
        ):
            with self.subTest(command=command):
                with mock.patch("ai_assistant.tools.sandbox.subprocess.run") as sub:
                    result = run(command, "C:/workspace", 30)
                self.assertIn("cmake -E", result["error"])
                sub.assert_not_called()

    def test_cmake_benign_e_subcommand_still_runs(self) -> None:
        with mock.patch(
            "ai_assistant.tools.sandbox.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ) as sub:
            result = run("cmake -E cmake_echo_color red", "C:/workspace", 30)
        self.assertEqual(result["return_code"], 0)
        sub.assert_called_once()

    def test_git_config_injection_is_blocked(self) -> None:
        result = run("git -c core.sshCommand=touch /tmp/x log", "C:/workspace", 30)
        self.assertIn("git -c", result["error"])

    def test_git_subcommand_outside_the_allow_list_is_blocked(self) -> None:
        """The declared subcommand list is enforced, not merely documented."""
        for command in ("git config user.name", "git daemon"):
            with self.subTest(command=command):
                result = run(command, "C:/workspace", 30)
                self.assertIn("not in the git allow-list", result["error"])

    def test_interpreter_subcommand_outside_the_allow_list_is_blocked(self) -> None:
        result = run("python randomword", "C:/workspace", 30)
        self.assertIn("not in the python allow-list", result["error"])

    def test_standalone_dangerous_token_is_blocked_before_parsing(self) -> None:
        """ADR 0011: ``del``/``rm``/``format``/``shutdown`` die as tokens."""
        with mock.patch("ai_assistant.tools.sandbox.subprocess.run") as sub:
            result = run("git commit -m del old files", "C:/workspace", 30)
        self.assertIn("dangerous command token", result["error"])
        sub.assert_not_called()

    def test_bare_allow_listed_executable_is_accepted(self) -> None:
        """A command with no subcommand has nothing left to police."""
        with mock.patch(
            "ai_assistant.tools.sandbox.subprocess.run",
            return_value=SimpleNamespace(returncode=1, stdout="", stderr="usage"),
        ) as sub:
            result = run("git", "C:/workspace", 30)
        self.assertEqual(result["return_code"], 1)
        sub.assert_called_once()

    def test_allowed_subcommands_are_accepted(self) -> None:
        for command in ("git status", "git log --oneline", "ruff check src", "ctest -R focused"):
            with self.subTest(command=command):
                with mock.patch(
                    "ai_assistant.tools.sandbox.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
                ) as sub:
                    result = run(command, "C:/workspace", 30)
                self.assertEqual(result["return_code"], 0)
                sub.assert_called_once()

    def test_test_suite_path_subcommand_is_accepted_by_prefix(self) -> None:
        """``pytest tests/...`` matches the declared ``tests/`` prefix."""
        with mock.patch(
            "ai_assistant.tools.sandbox.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ) as sub:
            result = run("pytest tests/unit/test_sandbox.py -q", "C:/workspace", 30)
        self.assertEqual(result["return_code"], 0)
        sub.assert_called_once()


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

    def test_policy_and_secret_files_are_denied(self) -> None:
        """ADR 0011: the agent may not rewrite the rules it runs under."""
        for name in ("pyproject.toml", ".env", "security.py", "sandbox.py"):
            with self.subTest(name=name):
                result = write_file(str(self.root / "src" / name), "x", str(self.root))
                self.assertIn("denied by sandbox policy", result["error"])

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
