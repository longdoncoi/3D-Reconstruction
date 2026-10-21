"""Unit tests for the policy-first command/write sandbox."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_assistant.exec_policy import (
    executable_candidates,
    resolve_executable,
)
from ai_assistant.tools.sandbox import (
    create_directory,
    run,
    write_file,
)


def _settings(**overrides) -> SimpleNamespace:
    base = {
        "sandbox_enabled": True,
        "sandbox_runtime": "local",
        "sandbox_image": "agent:latest",
        "max_write_bytes": 1024 * 1024,
        "sandbox_require_isolation": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class SandboxRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._settings = mock.patch("ai_assistant.tools.sandbox.load_agent_runtime_settings", return_value=_settings())
        self._settings.start()
        self.addCleanup(self._settings.stop)
        # Hermetic PATH resolution: the suite must not depend on which of the
        # allow-listed tools happen to be installed on the machine, and every
        # local run now executes the resolved absolute hit, not the bare name.
        # The seam is the *resolver* itself, not the platform's lookup: the
        # resolver is the gate, and it is exercised for real by
        # ``ExecutableResolutionTests`` below.
        self._which = mock.patch(
            "ai_assistant.tools.sandbox._resolve_executable",
            side_effect=lambda name, path=None: f"C:/tools/{name}.exe",
        )
        self._which.start()
        self.addCleanup(self._which.stop)

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

    def test_isolation_required_refuses_the_local_runtime(self) -> None:
        """ADR 0023: an allow-list is not an OS boundary, so a deployment
        that declares untrusted tool execution refuses the local runtime."""
        with mock.patch(
            "ai_assistant.tools.sandbox.load_agent_runtime_settings",
            return_value=_settings(sandbox_require_isolation=True),
        ):
            with mock.patch("ai_assistant.tools.sandbox.subprocess.run") as sub:
                result = run("python --version", "C:/workspace", 30)
        self.assertIn("isolation is required", result["error"])
        sub.assert_not_called()

    def test_isolation_required_still_runs_inside_docker(self) -> None:
        with mock.patch(
            "ai_assistant.tools.sandbox.load_agent_runtime_settings",
            return_value=_settings(sandbox_runtime="docker", sandbox_require_isolation=True),
        ):
            with mock.patch(
                "ai_assistant.tools.sandbox.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
            ) as sub:
                result = run("python --version", "C:/workspace", 30)
        self.assertEqual(result["return_code"], 0)
        sub.assert_called_once()

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
        # Egress deny by policy + no capability path back out of the sandbox.
        self.assertEqual(argv[argv.index("--cap-drop") + 1], "ALL")
        self.assertEqual(argv[argv.index("--security-opt") + 1], "no-new-privileges")
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

    def test_the_executable_is_resolved_to_the_path_hit(self) -> None:
        """The command runs as the operator's absolute PATH hit.

        Windows ``CreateProcess`` searches the current directory *before*
        PATH, so a bare ``git`` would run a ``git.exe`` planted in the
        workspace; executing the resolved path removes that ordering.
        """
        with mock.patch(
            "ai_assistant.tools.sandbox.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ) as sub:
            result = run("git status", "C:/workspace", 30)
        self.assertEqual(result["return_code"], 0)
        self.assertEqual(sub.call_args.args[0], ["C:/tools/git.exe", "status"])

    def test_a_name_missing_from_path_is_refused_outright(self) -> None:
        """Letting the OS hunt for the binary is what hands the workspace
        priority, so a name PATH does not carry is refused, not delegated."""
        with mock.patch("ai_assistant.tools.sandbox._resolve_executable", return_value=None):
            with mock.patch("ai_assistant.tools.sandbox.subprocess.run") as sub:
                result = run("git status", "C:/workspace", 30)
        self.assertIn("not found on PATH", result["error"])
        sub.assert_not_called()

    def test_cmake_script_and_build_dirs_must_stay_in_the_workspace(self) -> None:
        """``-P`` runs a script and ``-S``/``-B``/``--build`` run a build
        system: pointed outside the workspace either is arbitrary execution."""
        with tempfile.TemporaryDirectory() as workspace:
            outside = Path(workspace).parent
            for command in (
                "cmake -P ../evil.cmake",
                f"cmake -P {outside / 'evil.cmake'}",
                "cmake -S ../elsewhere",
                "cmake --build ../outside",
            ):
                with self.subTest(command=command):
                    with mock.patch("ai_assistant.tools.sandbox.subprocess.run") as sub:
                        result = run(command, workspace, 30)
                    self.assertIn("inside the workspace", result["error"])
                    sub.assert_not_called()

    def test_cmake_workspace_local_targets_still_run(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with mock.patch(
                "ai_assistant.tools.sandbox.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
            ) as sub:
                result = run("cmake -P tools/gen.cmake", workspace, 30)
            self.assertEqual(result["return_code"], 0)
            sub.assert_called_once()

    def test_git_escape_flags_are_blocked(self) -> None:
        """Each flag re-opens the allow-list: a program to launch, a binary
        set to swap, or another repository to retarget."""
        for command in (
            "git fetch --upload-pack=evil",
            "git push --receive-pack=evil",
            "git clone --config core.sshCommand=evil https://example.invalid/r",
            "git --exec-path=/tmp/status status",
            "git --git-dir=/tmp/status status",
        ):
            with self.subTest(command=command):
                with mock.patch("ai_assistant.tools.sandbox.subprocess.run") as sub:
                    result = run(command, "C:/workspace", 30)
                self.assertIn("blocked", result["error"])
                sub.assert_not_called()

    def test_git_remote_helper_and_bisect_run_are_blocked(self) -> None:
        """``ext::<cmd>`` shells out on fetch; ``bisect run`` executes its
        verdict hook — both are arbitrary commands behind a valid subcommand."""
        for command in ("git remote add evil ext::sh", "git bisect run verdict.sh"):
            with self.subTest(command=command):
                with mock.patch("ai_assistant.tools.sandbox.subprocess.run") as sub:
                    result = run(command, "C:/workspace", 30)
                self.assertIn("blocked", result["error"])
                sub.assert_not_called()

    def test_git_bisect_state_verbs_still_run(self) -> None:
        with mock.patch(
            "ai_assistant.tools.sandbox.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ) as sub:
            result = run("git bisect start", "C:/workspace", 30)
        self.assertEqual(result["return_code"], 0)
        sub.assert_called_once()

    def test_dollar_brace_expansion_is_rejected(self) -> None:
        """``${VAR}`` is shell expansion like ``$(...)``/backticks (ADR 0011)."""
        for command in ("git log ${HOME}", "cmake --version ${PATH}"):
            with self.subTest(command=command):
                result = run(command, "C:/workspace", 30)
                self.assertIn("rejected", result["error"])

    def test_interpreter_script_outside_workspace_is_blocked(self) -> None:
        """``python /tmp/evil.py`` passed on shape alone; now confined."""
        with tempfile.TemporaryDirectory() as workspace:
            outside = str(Path(workspace).parent / "evil.py")
            for command in (f"python {outside}", f"pytest {outside}"):
                with self.subTest(command=command):
                    with mock.patch("ai_assistant.tools.sandbox.subprocess.run") as sub:
                        result = run(command, workspace, 30)
                    self.assertIn("inside the workspace", result["error"])
                    sub.assert_not_called()

    def test_interpreter_workspace_local_script_still_runs(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with mock.patch(
                "ai_assistant.tools.sandbox.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
            ) as sub:
                result = run("python tools/gen.py", workspace, 30)
            self.assertEqual(result["return_code"], 0)
            sub.assert_called_once()

    def test_non_interpreter_path_positional_ignores_confinement(self) -> None:
        """Confinement applies only to python/pytest; git paths use their own gate."""
        with mock.patch(
            "ai_assistant.tools.sandbox.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ) as sub:
            result = run("git status", "C:/workspace", 30)
        self.assertEqual(result["return_code"], 0)
        sub.assert_called_once()


class ExecutableResolutionTests(unittest.TestCase):
    """The PATH resolution the sandbox gate depends on, exercised for real.

    The gate used to be pinned by a mock: ``shutil.which`` was patched to
    return ``None``, so the test proved the refusal fires *when the platform
    lookup fails* — a precondition that never holds on Windows, where the
    lookup finds a binary planted in the current directory and answers with a
    CWD-relative path (``.\\git.BAT``). A mock of the platform's lookup
    cannot reproduce the attack, so the resolver itself is tested here,
    against real files on a real filesystem: this is the test that was
    missing when the control did not work.
    """

    def test_a_binary_in_the_current_directory_is_not_resolved(self) -> None:
        """The current directory is never consulted.

        Windows ``CreateProcess`` searches it *before* PATH, so a binary
        planted next to the process wins over the operator's own tool — and
        the server's working directory is a directory the agent may write
        to. The resolver walks PATH itself and never looks here.
        """
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "git.bat").write_text("@echo PWNED\r\n", encoding="utf-8")
            elsewhere = tempfile.mkdtemp(prefix="empty_path_")
            original = os.getcwd()
            os.chdir(workspace)
            try:
                resolved = resolve_executable("git", elsewhere)
            finally:
                os.chdir(original)
                shutil.rmtree(elsewhere, ignore_errors=True)
        self.assertIsNone(
            resolved,
            "a binary that is not on PATH must not be resolved, even from the current directory",
        )

    def test_a_path_hit_is_resolved_to_an_absolute_path(self) -> None:
        """The hit is absolute, so the OS cannot re-order the search."""
        with tempfile.TemporaryDirectory() as tools:
            binary = Path(tools) / ("git.exe" if os.name == "nt" else "git")
            binary.write_text("", encoding="utf-8")
            resolved = resolve_executable("git", tools)
        self.assertIsNotNone(resolved)
        self.assertTrue(os.path.isabs(str(resolved)), "the resolved hit must be absolute")
        self.assertEqual(Path(str(resolved)).name, binary.name)

    def test_a_name_that_carries_a_path_is_refused(self) -> None:
        """Only a bare name may be resolved; the allow-list needs a name."""
        self.assertIsNone(resolve_executable(os.path.join("x", "git"), os.environ.get("PATH", "")))
        self.assertIsNone(resolve_executable(os.path.abspath("git"), os.environ.get("PATH", "")))
        self.assertIsNone(resolve_executable("", os.environ.get("PATH", "")))

    def test_a_path_shaped_name_is_refused_even_when_the_file_exists(self) -> None:
        """The allow-list constrains the *name*, not the path.

        The refusal has to hold even when the scan would find something: a
        ``PATH`` entry that happens to contain ``x/git`` must not satisfy
        the lookup, or the allow-list would stop constraining *which*
        program runs.
        """
        with tempfile.TemporaryDirectory() as tools:
            nested = Path(tools) / "x"
            nested.mkdir()
            binary = nested / "git"
            binary.write_text("", encoding="utf-8")
            # Executable on POSIX; on Windows every existing file is one.
            os.chmod(binary, 0o700)
            self.assertIsNone(resolve_executable(os.path.join("x", "git"), tools))
            self.assertIsNone(resolve_executable(str(binary), tools))

    def test_an_empty_path_resolves_nothing(self) -> None:
        self.assertIsNone(resolve_executable("git", ""))

    def test_the_sandbox_aliases_delegate_to_the_kernel(self) -> None:
        """The sandbox keeps the historical names; the rule has one owner.

        The suite mocks the resolver to stay hermetic (it must not depend on
        which tools happen to be installed), so the aliases themselves would
        otherwise never run.
        """
        from ai_assistant.tools import sandbox

        with tempfile.TemporaryDirectory() as tools_dir:
            binary = Path(tools_dir) / ("git.exe" if os.name == "nt" else "git")
            binary.write_text("", encoding="utf-8")
            self.assertEqual(
                sandbox._resolve_executable("git", tools_dir),
                resolve_executable("git", tools_dir),
            )
            self.assertEqual(
                sandbox._executable_candidates("git"),
                executable_candidates("git"),
            )

    def test_the_platform_decides_the_program_spellings(self) -> None:
        """Windows resolves through ``PATHEXT``; POSIX has one spelling.

        The branch is a module constant so both sides can be exercised on
        either platform — the coverage gate runs on Linux *and* Windows, and
        a branch only one of them can reach would sit at 99 % forever on the
        other.
        """
        with mock.patch("ai_assistant.exec_policy.IS_WINDOWS", False):
            self.assertEqual(executable_candidates("git"), ["git"])
        with mock.patch("ai_assistant.exec_policy.IS_WINDOWS", True):
            candidates = executable_candidates("git")
        self.assertIn("git.exe", candidates)

    def test_a_missing_pathext_falls_back_to_the_standard_extensions(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("ai_assistant.exec_policy.IS_WINDOWS", True):
                candidates = executable_candidates("git")
        self.assertIn("git.exe", candidates)
        self.assertIn("git.bat", candidates)


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
        for name in ("pyproject.toml", ".env", "security.py", "sandbox.py",
                     "tasks.sqlite", "pending_agent_actions.json", "agent_audit.jsonl"):
            with self.subTest(name=name):
                result = write_file(str(self.root / "src" / name), "x", str(self.root))
                self.assertIn("denied by sandbox policy", result["error"])

    def test_the_policy_deny_list_ignores_the_case_of_the_file(self) -> None:
        """Windows filesystems are case-insensitive.

        ``PYPROJECT.TOML`` *is* ``pyproject.toml``, so a case-sensitive
        comparison let the variant through and the write replaced the real
        file — the dependency and build policy the agent is judged by. The
        substring fallback cannot cover this on its own: ``pyproject.toml`` is
        the one entry that is not a path fragment.
        """
        for name in ("PYPROJECT.TOML", "PyProject.Toml", "PyProject.toml"):
            with self.subTest(name=name):
                result = write_file(str(self.root / "src" / name), "x", str(self.root))
                self.assertIn("denied by sandbox policy", result["error"])

    def test_the_composition_root_is_not_writable(self) -> None:
        """The layer that assembles the platform *is* the policy.

        ADR 0032 named the policy files; the composition root decides
        whether the OpenAPI surface is served, which dotenv file the
        secrets are loaded from and which key the at-rest cipher is built
        from. An agent that can rewrite those can turn the gates off
        without touching a single policy file.
        """
        for rel in (
            "AIAssistant/src/ai_assistant/bootstrap/app_factory.py",
            "AIAssistant/src/ai_assistant/bootstrap/server.py",
            "AIAssistant/src/ai_assistant/bootstrap/modules.py",
            "AIAssistant/src/ai_assistant/llm/model_loader.py",
            "AIAssistant/src/ai_assistant/orchestration/graph.py",
            "AIAssistant/src/ai_assistant/plugins/registry.py",
            "AIAssistant/src/ai_assistant/rag/index.py",
            "AIAssistant/src/ai_assistant/observability/metrics.py",
            "AIAssistant/src/ai_assistant/tools/sandbox.py",
        ):
            with self.subTest(rel=rel):
                result = write_file(str(self.root / rel), "x", str(self.root))
                self.assertIn("denied by sandbox policy", result["error"])

    def test_vcs_and_task_state_paths_are_denied_by_substring(self) -> None:
        """``.git/config`` and task-store paths are denied even under allow-listed roots."""
        for rel in ("src/.git/config", "src/tasks.sqlite"):
            with self.subTest(rel=rel):
                target = self.root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                result = write_file(str(target), "x", str(self.root))
                self.assertIn("denied by sandbox policy", result["error"])

    def test_the_policy_trees_are_denied(self) -> None:
        """ADR 0032: the basename deny-list named two policy files and left
        the rest of the platform's own policy writable — an approved write
        could rewrite the origin allow-list, the A2A scope map, the tool
        catalogue or the approval logic the agent is judged by."""
        for rel in (
            "AIAssistant/config/base.toml",
            "AIAssistant/src/ai_assistant/domain/approvals.py",
            "AIAssistant/src/ai_assistant/application/tools.py",
            "AIAssistant/src/ai_assistant/adapters/http/security.py",
            "AIAssistant/src/ai_assistant/ports.py",
            "Config/agent_tools.json",
            "src/core/AppConstants.h",
        ):
            with self.subTest(rel=rel):
                target = self.root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                result = write_file(str(target), "x", str(self.root))
                self.assertIn("denied by sandbox policy", result["error"])
                self.assertFalse(target.exists() and target.read_text(encoding="utf-8") == "x")

    def test_a_path_that_only_looks_like_a_policy_tree_is_allowed(self) -> None:
        """The tree match is on resolved parts, so a sibling directory with a
        similar name is not caught by accident."""
        target = self.root / "src" / "ai_assistant" / "domain_notes" / "a.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        result = write_file(str(target), "x", str(self.root))
        self.assertTrue(result["success"])

    def test_a_symlink_into_a_policy_tree_is_denied(self) -> None:
        """The match is on the *resolved* path, so a symlink cannot walk
        around the deny-list."""
        denied = self.root / "AIAssistant" / "config" / "base.toml"
        denied.parent.mkdir(parents=True, exist_ok=True)
        link = self.root / "src" / "innocent.toml"
        try:
            link.symlink_to(denied)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not available on this filesystem")
        result = write_file(str(link), "x", str(self.root))
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
