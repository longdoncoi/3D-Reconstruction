"""Unit tests for agent path-safety utilities."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_assistant.tools.builtin.path_utils import (
    _AGENT_BLOCKED_DIRS,
    _AGENT_BLOCKED_EXTS,
    _AGENT_MAX_FILE_READ_CHARS,
    _AGENT_MAX_UNSCOPED_SOURCE_LINES,
    agent_safe_path,
)


class AgentSafePathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        patcher = mock.patch(
            "ai_assistant.tools.builtin.path_utils.get_paths",
            return_value=SimpleNamespace(project_root=self.root),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_safe_relative_path_resolves(self) -> None:
        resolved = agent_safe_path("src/app/main.py")
        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.startswith(self.root))

    def test_backslashes_normalised(self) -> None:
        resolved = agent_safe_path("src\\app\\main.py")
        self.assertIsNotNone(resolved)
        self.assertIn("src", resolved)

    def test_empty_path_is_rejected(self) -> None:
        self.assertIsNone(agent_safe_path(""))
        self.assertIsNone(agent_safe_path(None))

    def test_parent_traversal_is_rejected(self) -> None:
        self.assertIsNone(agent_safe_path("../../etc/passwd"))
        self.assertIsNone(agent_safe_path("a/../../b"))

    def test_blocked_directories_are_rejected(self) -> None:
        self.assertIsNone(agent_safe_path(".git/config"))
        self.assertIsNone(agent_safe_path("build/x/y.sln"))
        self.assertIsNone(agent_safe_path("node_modules/pkg/index.js"))

    @unittest.skipUnless(sys.platform == "win32", "a foreign drive letter is only spellable on Windows")
    def test_absolute_escape_is_rejected(self) -> None:
        # On Windows os.path.join discards the sandbox root for a path that
        # already carries a drive letter, and commonpath then raises; POSIX
        # re-roots absolute inputs under the sandbox instead (leading
        # slashes are stripped), so there is no equivalent escape vector.
        self.assertIsNone(agent_safe_path("C:/Windows/system32"))

    def test_a_containment_check_that_cannot_be_made_is_a_rejection(self) -> None:
        # The guard behind that test: comparing two paths that share no root
        # raises ValueError (mixed drives on Windows). Forced via mock so the
        # rejection branch is exercised on every platform. The comparison
        # itself now lives in the kernel (path_policy.resolve_within).
        with mock.patch(
            "ai_assistant.path_policy.Path.relative_to",
            side_effect=ValueError("Paths don't have the same drive"),
        ):
            self.assertIsNone(agent_safe_path("src/app/main.py"))

    def test_a_path_outside_the_sandbox_is_rejected(self) -> None:
        # A real path outside the root — not a mock of the comparison. The
        # decision belongs to the kernel; this pins that the agent gate uses
        # it and that an absolute input cannot re-root itself inside.
        outside = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(outside, ignore_errors=True))
        self.assertIsNone(agent_safe_path(os.path.join(outside, "secret.txt")))

    def test_a_symlink_that_leaves_the_sandbox_is_rejected(self) -> None:
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        link = os.path.join(self.root, "escape")
        try:
            os.symlink(outside.name, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not available for this user")
        self.assertIsNone(agent_safe_path("escape/secret.txt"))

    def test_the_resolved_target_decides_not_the_spelling(self) -> None:
        # The *real* target is what gets checked: a spelling that looks
        # inside the sandbox but resolves outside is an escape. Forced via
        # mock so the branch is exercised on platforms where creating a
        # symlink needs elevated privileges.
        def _fake_canonical(path: Path) -> Path:
            # The root keeps its own spelling; the *target* resolves outside.
            if Path(path).is_dir():
                return Path(path)
            return Path(os.path.dirname(self.root)) / "elsewhere" / Path(path).name

        with mock.patch("ai_assistant.path_policy.canonical_path", side_effect=_fake_canonical):
            self.assertIsNone(agent_safe_path("src/app/main.py"))

    def test_a_link_into_a_blocked_directory_is_refused(self) -> None:
        """The *resolved* path is checked too, so a link is not a way in.

        The link is created for real where the platform allows it; where it
        does not, the resolved target is forced — the *decision* is the same
        one, and the branch must be measured on every platform.
        """
        blocked = Path(self.root) / ".git"
        blocked.mkdir()
        (blocked / "config").write_text("[remote]\n", encoding="utf-8")
        link = Path(self.root) / "notes.txt"
        try:
            link.symlink_to(blocked / "config")
        except (OSError, NotImplementedError):
            def _fake_canonical(path: Path) -> Path:
                if Path(path).is_dir():
                    return Path(path)
                return blocked / "config"

            with mock.patch("ai_assistant.path_policy.canonical_path", side_effect=_fake_canonical):
                self.assertIsNone(agent_safe_path("notes.txt"))
            return
        self.assertIsNone(agent_safe_path("notes.txt"))

    def test_a_blocked_directory_in_any_case_is_refused(self) -> None:
        """A Windows filesystem is case-insensitive: ``.GIT`` is ``.git``.

        The case-sensitive form of this rule is why ``read_file(".GIT/
        config")`` returned the repository's git configuration — the remote
        URL and its credential helper included — while ``.git/config`` was
        refused.
        """
        for name in _AGENT_BLOCKED_DIRS:
            with self.subTest(name=name):
                self.assertIsNone(agent_safe_path(f"{name.upper()}/x"))
                self.assertIsNone(agent_safe_path(f"{name.capitalize()}/x"))

    def test_a_relative_path_that_cannot_be_compared_is_refused(self) -> None:
        """A comparison across drives cannot be made: that is a refusal."""
        with mock.patch(
            "ai_assistant.tools.builtin.path_utils.os.path.relpath",
            side_effect=ValueError("different drives"),
        ):
            self.assertIsNone(agent_safe_path("src/app/main.py"))

    def test_constant_defaults_present(self) -> None:
        self.assertIn(".exe", _AGENT_BLOCKED_EXTS)
        self.assertIn(".git", _AGENT_BLOCKED_DIRS)
        self.assertGreater(_AGENT_MAX_FILE_READ_CHARS, 0)
        self.assertGreater(_AGENT_MAX_UNSCOPED_SOURCE_LINES, 0)


if __name__ == "__main__":
    unittest.main()
