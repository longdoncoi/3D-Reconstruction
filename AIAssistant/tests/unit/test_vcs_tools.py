"""Unit tests for the Git project-status and diff builtin tools.

``subprocess`` and ``get_paths`` are faked so no real Git invocation or repo
layout is required.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_assistant.tools.builtin import vcs_tools


def _fake_completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return SimpleNamespace(stdout=stdout, returncode=returncode, stderr=stderr)


class ProjectStatusTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "main.py").write_text("x", encoding="utf-8")
        (self.root / "lib.h").write_text("x", encoding="utf-8")
        (self.root / "data.txt").write_text("x", encoding="utf-8")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text("x", encoding="utf-8")

    def test_counts_source_files_and_skips_blocked_dirs(self):
        pending = {
            ("branch", "--show-current"): _fake_completed("main\n"),
            ("status", "--short"): _fake_completed("M main.py\n"),
        }

        def fake_run(command, **kwargs):
            # _run_git resolves "git" via PATH (absolute hit); the policy
            # gate is what matters here, not the exact binary prefix.
            key = tuple(command[1:3]) if len(command) >= 3 else tuple(command[1:])
            return pending[key]

        with mock.patch.object(vcs_tools, "get_paths", return_value=SimpleNamespace(project_root=self.root)):
            with mock.patch.object(vcs_tools.subprocess, "run", side_effect=fake_run):
                with mock.patch.object(vcs_tools, "resolve_executable", return_value="/usr/bin/git"):
                    result = vcs_tools.tool_get_project_status({})
        self.assertEqual(result["source_file_counts"], {".py": 1, ".h": 1})
        self.assertNotIn(".txt", result["source_file_counts"])
        self.assertEqual(result["git_branch"], "main")
        self.assertEqual(result["changed_files"], ["M main.py"])

    def test_git_failure_falls_back_to_empty(self):
        def fake_run(command, **kwargs):
            msg = "git missing"
            raise OSError(msg)

        with mock.patch.object(vcs_tools, "get_paths", return_value=SimpleNamespace(project_root=self.root)):
            with mock.patch.object(vcs_tools.subprocess, "run", side_effect=fake_run):
                with mock.patch.object(vcs_tools, "resolve_executable", return_value="/usr/bin/git"):
                    result = vcs_tools.tool_get_project_status({})
        self.assertEqual(result["git_branch"], "")
        self.assertEqual(result["changed_files"], [])
        self.assertIn(".py", result["source_file_counts"])


class GitDiffTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "file.py").write_text("content", encoding="utf-8")

    def _patch(self, completed):
        get_paths_patcher = mock.patch.object(
            vcs_tools, "get_paths",
            return_value=SimpleNamespace(project_root=self.root),
        )
        run_patcher = mock.patch.object(vcs_tools.subprocess, "run", return_value=completed)
        resolve_patcher = mock.patch.object(vcs_tools, "resolve_executable", return_value="/usr/bin/git")
        get_paths_patcher.start()
        run = run_patcher.start()
        resolve_patcher.start()
        self.addCleanup(get_paths_patcher.stop)
        self.addCleanup(run_patcher.stop)
        self.addCleanup(resolve_patcher.stop)
        return run

    def test_default_diff_path(self):
        run = self._patch(_fake_completed("+def f(): pass\n"))
        result = vcs_tools.tool_git_diff({})
        self.assertTrue(result["content"].startswith("+def f(): pass"))
        self.assertFalse(result["truncated"])
        self.assertNotIn("--cached", run.call_args.args[0])

    def test_staged_diff(self):
        run = self._patch(_fake_completed("diff"))
        vcs_tools.tool_git_diff({"path": ".", "staged": True})
        self.assertIn("--cached", run.call_args.args[0])

    def test_traversal_rejected(self):
        with mock.patch.object(vcs_tools, "get_paths", return_value=SimpleNamespace(project_root=self.root)):
            result = vcs_tools.tool_git_diff({"path": "../secret"})
        self.assertIn("error", result)

    def test_missing_path_returns_error(self):
        with mock.patch.object(vcs_tools, "get_paths", return_value=SimpleNamespace(project_root=self.root)):
            result = vcs_tools.tool_git_diff({"path": "nope.py"})
        self.assertIn("Invalid or missing diff path", result["error"])

    def test_nonzero_returncode_returns_error(self):
        self._patch(_fake_completed("", returncode=1, stderr="boom"))
        result = vcs_tools.tool_git_diff({})
        self.assertEqual(result["error"], "boom")

    def test_subprocess_error_returns_error(self):
        with mock.patch.object(vcs_tools, "get_paths", return_value=SimpleNamespace(project_root=self.root)):
            with mock.patch.object(vcs_tools.subprocess, "run", side_effect=OSError("no git")):
                with mock.patch.object(vcs_tools, "resolve_executable", return_value="/usr/bin/git"):
                    result = vcs_tools.tool_git_diff({})
        self.assertIn("Unable to read Git diff", result["error"])

    def test_policy_gate_refuses_a_blocked_subcommand(self):
        """The sandbox gate runs before any subprocess: a blocked git flag
        surfaces as a diff error without ever reaching ``subprocess.run``."""
        with mock.patch.object(vcs_tools, "get_paths", return_value=SimpleNamespace(project_root=self.root)):
            with mock.patch.object(vcs_tools.subprocess, "run") as sub:
                with mock.patch.object(
                    vcs_tools, "_subcommand_rejection", return_value="Sandbox: blocked"
                ):
                    result = vcs_tools.tool_git_diff({})
        self.assertIn("Unable to read Git diff", result["error"])
        sub.assert_not_called()

    def test_executable_gate_refusal_surfaces_as_diff_error(self):
        with mock.patch.object(vcs_tools, "get_paths", return_value=SimpleNamespace(project_root=self.root)):
            with mock.patch.object(vcs_tools.subprocess, "run") as sub:
                with mock.patch.object(
                    vcs_tools, "_executable_rejection", return_value="not allowed"
                ):
                    result = vcs_tools.tool_git_diff({})
        self.assertIn("Unable to read Git diff", result["error"])
        sub.assert_not_called()

    def test_an_unresolved_binary_is_refused_rather_than_degraded(self):
        """Fail closed: a bare name hands the search back to the OS.

        On Windows the operating system searches the *current directory*
        first, so a fallback to the bare name "git" re-opened exactly the
        hijack the resolver exists to close — silently, and with no way for
        the caller to tell. The shipped container image has no ``git`` at
        all, so there the fallback was the *default* path.
        """
        run = self._patch(_fake_completed("diff"))
        with mock.patch.object(vcs_tools, "resolve_executable", return_value=None):
            result = vcs_tools.tool_git_diff({})
        self.assertIn("Unable to read Git diff", result["error"])
        run.assert_not_called()

    def test_large_diff_is_truncated(self):
        self._patch(_fake_completed("x" * 20000))
        result = vcs_tools.tool_git_diff({})
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["content"]), 16000)


if __name__ == "__main__":
    unittest.main()
