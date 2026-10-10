"""Unit tests for system/sandbox execution tools."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_assistant.tools.builtin.system_tools import tool_run_command, tool_validate_file


class SystemToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        path_patcher = mock.patch(
            "ai_assistant.tools.builtin.path_utils.get_paths",
            return_value=SimpleNamespace(project_root=str(self.root)),
        )
        path_patcher.start()
        self.addCleanup(path_patcher.stop)
        self.sys_get_paths = mock.patch(
            "ai_assistant.tools.builtin.system_tools.get_paths",
            return_value=SimpleNamespace(project_root=str(self.root)),
        )
        self.sys_get_paths.start()
        self.addCleanup(self.sys_get_paths.stop)

    def _write(self, name: str, content: str) -> str:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return name

    def test_valid_python(self) -> None:
        name = self._write("ok.py", "import os\nVALUE = 1\n")
        result = tool_validate_file({"path": name})
        self.assertTrue(result["success"])
        self.assertEqual(result["validation"], "syntax_valid")

    def test_invalid_python(self) -> None:
        name = self._write("bad.py", "def f(:\n")
        result = tool_validate_file({"path": name})
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_valid_json(self) -> None:
        name = self._write("cfg.json", json.dumps({"a": 1}))
        result = tool_validate_file({"path": name})
        self.assertTrue(result["success"])

    def test_invalid_json(self) -> None:
        name = self._write("bad.json", "{oops")
        result = tool_validate_file({"path": name})
        self.assertFalse(result["success"])

    def test_unsupported_extension(self) -> None:
        name = self._write("x.txt", "hello")
        result = tool_validate_file({"path": name})
        self.assertIn("Only .py and .json", result["error"])

    def test_missing_file(self) -> None:
        result = tool_validate_file({"path": "nope.py"})
        self.assertIn("error", result)

    def test_run_command_default_cwd(self) -> None:
        with mock.patch(
            "ai_assistant.tools.builtin.system_tools.run_sandboxed_command",
            return_value={"exit_code": 0, "output": ""},
        ) as run:
            result = tool_run_command({"command": "ls"})
        run.assert_called_once_with("ls", str(self.root), 30)
        self.assertEqual(result, {"exit_code": 0, "output": ""})

    def test_run_command_invalid_cwd(self) -> None:
        result = tool_run_command({"command": "ls", "cwd": "../.."})
        self.assertIn("error", result)

    def test_run_command_timeout_capped(self) -> None:
        with mock.patch(
            "ai_assistant.tools.builtin.system_tools.run_sandboxed_command",
            return_value={"exit_code": 0, "output": ""},
        ) as run:
            tool_run_command({"command": "ls", "timeout": 5000})
        self.assertLessEqual(run.call_args.args[2], 120)


if __name__ == "__main__":
    unittest.main()
