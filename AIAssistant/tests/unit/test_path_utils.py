"""Unit tests for agent path-safety utilities."""
from __future__ import annotations

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

    def test_absolute_escape_is_rejected(self) -> None:
        self.assertIsNone(agent_safe_path("C:/Windows/system32"))

    def test_constant_defaults_present(self) -> None:
        self.assertIn(".exe", _AGENT_BLOCKED_EXTS)
        self.assertIn(".git", _AGENT_BLOCKED_DIRS)
        self.assertGreater(_AGENT_MAX_FILE_READ_CHARS, 0)
        self.assertGreater(_AGENT_MAX_UNSCOPED_SOURCE_LINES, 0)


if __name__ == "__main__":
    unittest.main()
