"""Unit tests for search/discovery tools."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_assistant.tools.builtin.search_tools import tool_find_files, tool_list_directory, tool_search_text


class SearchToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
        (self.root / "src" / "notes.txt").write_text("hello world\nsecond line\n", encoding="utf-8")
        (self.root / "docs").mkdir()
        (self.root / "docs" / "readme.md").write_text("Xin chào dự án\n", encoding="utf-8")
        (self.root / "docs" / "binary.bin").write_bytes(b"\x00\x01\x02binary")
        (self.root / "src" / ".git").mkdir()  # blocked dir (constructed inside src)
        (self.root / "src" / ".git" / "config").write_text("[core]\n", encoding="utf-8")
        for target in (
            "ai_assistant.tools.builtin.search_tools.get_paths",
            "ai_assistant.tools.builtin.path_utils.get_paths",
        ):
            patcher = mock.patch(target, return_value=SimpleNamespace(project_root=str(self.root)))
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _fwd(path: str) -> str:
        return path.replace("\\", "/")

    # ---- tool_list_directory

    def test_list_root(self) -> None:
        result = tool_list_directory({"path": "."})
        self.assertEqual(result["path"], ".")
        names = [e["name"] for e in result["entries"]]
        self.assertIn("src", names)
        self.assertIn("docs", names)
        self.assertTrue(all(e["type"] in {"file", "directory"} for e in result["entries"]))

    def test_list_missing_dir(self) -> None:
        result = tool_list_directory({"path": "ghost"})
        self.assertIn("error", result)

    def test_list_unsafe_path(self) -> None:
        result = tool_list_directory({"path": "../"})
        self.assertIn("error", result)

    def test_recursive_listing_skips_blocked_dirs(self) -> None:
        result = tool_list_directory({"path": ".", "recursive": True, "max_depth": 2})
        names = [self._fwd(e["name"]) for e in result["entries"]]
        self.assertIn("src/main.py", names)
        self.assertTrue(all(".git" not in n for n in names))

    def test_list_empty_dir(self) -> None:
        (self.root / "empty").mkdir()
        result = tool_list_directory({"path": "empty"})
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["entries"], [])

    # ---- tool_find_files

    def test_find_by_glob(self) -> None:
        result = tool_find_files({"pattern": "*.py"})
        self.assertEqual(result["count"], 1)
        self.assertEqual(self._fwd(result["matches"][0]), "src/main.py")
        self.assertFalse(result["truncated"])

    def test_find_wildcard_relative(self) -> None:
        result = tool_find_files({"pattern": "**/main.py"})
        self.assertEqual(self._fwd(result["matches"][0]), "src/main.py")

    def test_find_empty_pattern(self) -> None:
        result = tool_find_files({"pattern": ""})
        self.assertIn("error", result)

    def test_find_missing_dir(self) -> None:
        result = tool_find_files({"pattern": "*.py", "path": "nowhere"})
        self.assertIn("error", result)

    def test_find_max_results_truncates(self) -> None:
        for i in range(5):
            (self.root / "src" / f"file_{i}.py").write_text("x\n", encoding="utf-8")
        result = tool_find_files({"pattern": "*.py", "max_results": 3})
        self.assertEqual(result["count"], 3)
        self.assertTrue(result["truncated"])

    # ---- tool_search_text

    def test_text_search_finds_line(self) -> None:
        result = tool_search_text({"query": "world"})
        self.assertEqual(result["count"], 1)
        self.assertEqual(self._fwd(result["results"][0]["file"]), "src/notes.txt")
        self.assertEqual(result["results"][0]["line"], 1)

    def test_search_without_query(self) -> None:
        result = tool_search_text({"query": ""})
        self.assertIn("error", result)

    def test_case_sensitive_mode(self) -> None:
        result = tool_search_text({"query": "WORLD", "case_sensitive": True})
        self.assertEqual(result["count"], 0)
        result_ci = tool_search_text({"query": "WORLD", "case_sensitive": False})
        self.assertEqual(result_ci["count"], 1)

    def test_file_pattern_filter(self) -> None:
        result = tool_search_text({"query": "dự án", "file_pattern": "*.md"})
        self.assertEqual(result["count"], 1)
        self.assertEqual(self._fwd(result["results"][0]["file"]), "docs/readme.md")

    def test_single_file_search(self) -> None:
        result = tool_search_text({"query": "world", "path": "src/notes.txt"})
        self.assertEqual(result["count"], 1)
        self.assertFalse(result["truncated"])

    def test_binary_extension_excluded(self) -> None:
        result = tool_search_text({"query": "binary"})
        self.assertEqual(result["count"], 0)


if __name__ == "__main__":
    unittest.main()
