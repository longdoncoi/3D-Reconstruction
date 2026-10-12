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


    # ---- guard paths the happy-path listings never reach

    def test_list_non_recursive_skips_a_blocked_child(self) -> None:
        result = tool_list_directory({"path": "src"})
        names = [e["name"] for e in result["entries"]]
        self.assertNotIn(self._fwd("src/.git"), [self._fwd(n) for n in names])
        self.assertIn(self._fwd("src/main.py"), [self._fwd(n) for n in names])

    def test_list_recursive_stops_at_max_depth(self) -> None:
        deep = self.root / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "leaf.txt").write_text("x", encoding="utf-8")
        result = tool_list_directory({"path": ".", "recursive": True, "max_depth": 1})
        names = [e["name"] for e in result["entries"]]
        # Depth 1 is visited, everything under it is pruned before descending.
        self.assertIn("./a", [self._fwd(n) for n in names])
        self.assertNotIn("./a/b", [self._fwd(n) for n in names])

    def test_recursive_listing_stops_at_the_entry_cap_in_the_directory_loop(self) -> None:
        for index in range(160):
            (self.root / "wide" / f"dir{index:03d}").mkdir(parents=True, exist_ok=True)
        result = tool_list_directory({"path": "wide", "recursive": True})
        self.assertEqual(result["count"], 150)

    def test_recursive_listing_stops_at_the_entry_cap_in_the_file_loop(self) -> None:
        folder = self.root / "flat"
        folder.mkdir()
        for index in range(160):
            (folder / f"file{index:03d}.txt").write_text("x", encoding="utf-8")
        result = tool_list_directory({"path": "flat", "recursive": True})
        self.assertEqual(result["count"], 150)

    def test_a_listing_entry_whose_size_cannot_be_read_is_still_listed(self) -> None:
        with mock.patch(
            "ai_assistant.tools.builtin.search_tools.os.path.getsize",
            side_effect=OSError("vanished mid-walk"),
        ):
            flat = tool_list_directory({"path": "src"})
            recursive = tool_list_directory({"path": ".", "recursive": True})
        files = [e for e in flat["entries"] if e["type"] == "file"]
        self.assertTrue(files)
        self.assertTrue(all(entry["size_bytes"] == 0 for entry in files))
        self.assertTrue(all(
            entry["size_bytes"] == 0
            for entry in recursive["entries"] if entry["type"] == "file"
        ))

    def test_a_shallow_directory_stops_listing_at_the_entry_cap(self) -> None:
        folder = self.root / "many"
        folder.mkdir()
        for index in range(160):
            (folder / f"item{index:03d}.txt").write_text("x", encoding="utf-8")
        result = tool_list_directory({"path": "many"})
        self.assertEqual(result["count"], 150)

    def test_an_unreadable_directory_is_reported_not_raised(self) -> None:
        with mock.patch(
            "ai_assistant.tools.builtin.search_tools.os.listdir",
            side_effect=PermissionError("denied"),
        ):
            result = tool_list_directory({"path": "."})
        self.assertIn("Lỗi liệt kê thư mục", result["error"])

    def test_an_unreadable_walk_is_reported_not_raised(self) -> None:
        with mock.patch(
            "ai_assistant.tools.builtin.search_tools.os.walk",
            side_effect=PermissionError("denied"),
        ):
            result = tool_list_directory({"path": ".", "recursive": True})
        self.assertIn("Lỗi liệt kê thư mục", result["error"])

    def test_find_files_reports_a_walk_that_fails(self) -> None:
        with mock.patch(
            "ai_assistant.tools.builtin.search_tools.os.walk",
            side_effect=PermissionError("denied"),
        ):
            result = tool_find_files({"pattern": "*.py"})
        self.assertIn("Unable to find files", result["error"])

    def test_search_rejects_a_path_outside_the_project(self) -> None:
        result = tool_search_text({"query": "hello", "path": "../outside"})
        self.assertIn("Đường dẫn không hợp lệ", result["error"])

    def test_search_rejects_a_path_that_does_not_exist(self) -> None:
        result = tool_search_text({"query": "hello", "path": "ghost_dir"})
        self.assertIn("Đường dẫn không tồn tại", result["error"])

    def test_a_separator_only_file_pattern_falls_back_to_everything(self) -> None:
        """``","`` splits to nothing, which must not filter every file away."""
        result = tool_search_text({"query": "hello", "path": "src", "file_pattern": ","})
        self.assertEqual(result["count"], 1)
        self.assertTrue(result["results"][0]["file"].endswith("notes.txt"))

    def test_search_stops_at_the_result_cap(self) -> None:
        (self.root / "src" / "many_hits.py").write_text(
            "".join(f"target line {index}\n" for index in range(10)), encoding="utf-8",
        )
        result = tool_search_text({"query": "target", "path": "src", "max_results": 3})
        self.assertTrue(result["truncated"])
        self.assertEqual(result["count"], 3)

    def test_an_unreadable_file_is_skipped_and_the_rest_still_searched(self) -> None:
        import builtins
        real_open = builtins.open

        def _guarded_open(name, *args, **kwargs):
            if str(name).endswith("notes.txt"):
                msg = "locked by another process"
                raise OSError(msg)
            return real_open(name, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=_guarded_open):
            result = tool_search_text({"query": "def", "path": "src"})
        self.assertFalse(result.get("error"))
        self.assertTrue(result["results"])
        self.assertTrue(all(r["file"].endswith("main.py") for r in result["results"]))

    def test_search_reports_a_walk_that_fails(self) -> None:
        with mock.patch(
            "ai_assistant.tools.builtin.search_tools.os.walk",
            side_effect=PermissionError("denied"),
        ):
            result = tool_search_text({"query": "hello"})
        self.assertIn("Lỗi tìm kiếm", result["error"])


if __name__ == "__main__":
    unittest.main()
