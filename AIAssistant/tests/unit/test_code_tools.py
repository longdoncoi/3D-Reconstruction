"""Unit tests for code analysis tools."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_assistant.tools.builtin.code_tools import _analyze_cpp, _analyze_python, tool_analyze_code


class CodeToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        patcher = mock.patch(
            "ai_assistant.tools.builtin.path_utils.get_paths",
            return_value=SimpleNamespace(project_root=str(self.root)),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write(self, name: str, content: str) -> str:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return name

    def test_invalid_path_error(self) -> None:
        result = tool_analyze_code({"path": "../outside.py"})
        self.assertIn("error", result)

    def test_missing_file_error(self) -> None:
        result = tool_analyze_code({"path": "ghost.py"})
        self.assertIn("error", result)

    def test_python_analysis(self) -> None:
        name = self._write(
            "sample.py",
            "import os\nfrom pathlib import Path\n\nclass MyClass:\n    def run(self):\n        pass\n\ndef top(flag):\n    \"\"\"Docstring here.\"\"\"\n    return flag\n",
        )
        result = tool_analyze_code({"path": name})
        self.assertEqual(result["extension"], ".py")
        self.assertGreater(result["total_lines"], 0)
        self.assertEqual(result["classes"][0]["name"], "MyClass")
        self.assertEqual(result["classes"][0]["methods"], ["run"])
        self.assertEqual(result["functions"][0]["name"], "top")
        self.assertIn("os", result["imports"])
        self.assertIn("pathlib.Path", result["imports"])

    def test_python_syntax_error_reported(self) -> None:
        name = self._write("broken.py", "def f(:\n    pass")
        result = tool_analyze_code({"path": name})
        self.assertEqual(result["extension"], ".py")
        self.assertIn("syntax_error", result)

    def test_cpp_analysis(self) -> None:
        name = self._write(
            "widget.cpp",
            "#include <QWidget>\n\nclass Widget : public QWidget {\npublic:\n    void paint();\n};\n\nvoid Widget::paint() {\n}\n",
        )
        result = tool_analyze_code({"path": name})
        self.assertEqual(result["extension"], ".cpp")
        self.assertEqual(result["includes"], ["QWidget"])
        self.assertEqual(result["classes"][0]["name"], "Widget")
        self.assertEqual(result["classes"][0]["base"], "QWidget")

    def test_unsupported_extension_message(self) -> None:
        name = self._write("notes.txt", "hello")
        result = tool_analyze_code({"path": name})
        self.assertIn("không hỗ trợ", result["analysis"])

    def test_analyze_python_direct(self) -> None:
        result = _analyze_python("async def f():\n    pass\n", {"path": "x.py"})
        self.assertTrue(result["functions"][0]["is_async"])
        self.assertEqual(result["functions"][0]["name"], "f")

    def test_analyze_cpp_direct_skips_control_keywords(self) -> None:
        result = _analyze_cpp("int main() { return 0; }\n", {"path": "x.cpp"})
        names = [f["name"] for f in result["functions"]]
        self.assertNotIn("return", names)
        self.assertNotIn("if", names)


if __name__ == "__main__":
    unittest.main()
