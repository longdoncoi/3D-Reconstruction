"""Unit tests for agent path-safety utilities."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
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

    def test_a_commonpath_valueerror_is_rejection_not_a_crash(self) -> None:
        # The guard behind that test: os.path.commonpath raises ValueError
        # for mixed drives (Windows) and absolute-vs-relative mixes. Forced
        # via mock so the rejection branch is exercised on every platform.
        with mock.patch(
            "ai_assistant.tools.builtin.path_utils.os.path.commonpath",
            side_effect=ValueError("Paths don't have the same drive"),
        ):
            self.assertIsNone(agent_safe_path("src/app/main.py"))

    def test_a_path_whose_common_root_is_not_the_sandbox_is_rejected(self) -> None:
        # The other guard outcome: commonpath succeeds but resolves to
        # something other than the sandbox root. On Windows only a same-drive
        # absolute input reaches it, so the mismatch is forced via mock to
        # keep the branch exercised on every platform.
        with mock.patch(
            "ai_assistant.tools.builtin.path_utils.os.path.commonpath",
            return_value=os.path.dirname(self.root),
        ):
            self.assertIsNone(agent_safe_path("src/app/main.py"))

    def test_a_symlink_that_leaves_the_sandbox_is_rejected(self) -> None:
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        link = os.path.join(self.root, "escape")
        try:
            os.symlink(outside.name, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not available for this user")
        self.assertIsNone(agent_safe_path("escape/secret.txt"))

    def test_a_realpath_mismatch_is_rejected_even_without_a_symlink(self) -> None:
        # Force the second containment check so its false branch is exercised
        # on platforms where creating symlinks needs elevated privileges.
        with mock.patch(
            "ai_assistant.tools.builtin.path_utils.os.path.commonpath",
            side_effect=[self.root, os.path.dirname(self.root)],
        ):
            self.assertIsNone(agent_safe_path("src/app/main.py"))

    def test_constant_defaults_present(self) -> None:
        self.assertIn(".exe", _AGENT_BLOCKED_EXTS)
        self.assertIn(".git", _AGENT_BLOCKED_DIRS)
        self.assertGreater(_AGENT_MAX_FILE_READ_CHARS, 0)
        self.assertGreater(_AGENT_MAX_UNSCOPED_SOURCE_LINES, 0)


if __name__ == "__main__":
    unittest.main()
