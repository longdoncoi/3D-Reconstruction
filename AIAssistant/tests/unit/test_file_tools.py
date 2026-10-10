"""Unit tests for file read/write/patch tools and code-citation helpers."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_assistant.tools.builtin.file_tools import (
    _build_code_citation_result,
    _citation_is_python,
    _citation_path_hint,
    _code_language,
    _expand_function_end,
    _find_python_symbol_range,
    _find_symbol_range,
    _requested_code_symbol,
    tool_create_directory,
    tool_multi_replace_file_content,
    tool_patch_file,
    tool_read_file,
    tool_replace_file_content,
    tool_write_file,
)


class FileToolsBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.project_root = str(self.root)
        for target in (
            "ai_assistant.tools.builtin.path_utils.get_paths",
            "ai_assistant.tools.builtin.file_tools.get_paths",
        ):
            patcher = mock.patch(target, return_value=SimpleNamespace(project_root=self.project_root))
            patcher.start()
            self.addCleanup(patcher.stop)

    def _write(self, name: str, content: str | bytes) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if isinstance(content, bytes) else "w"
        kwargs = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
        path.open(mode, **kwargs).write(content)
        return path


class ToolReadFileTests(FileToolsBase):
    def test_missing_file(self) -> None:
        self.assertIn("error", tool_read_file({"path": "ghost.py"}))

    def test_unsafe_path(self) -> None:
        self.assertIn("error", tool_read_file({"path": "../x.py"}))

    def test_binary_extension_blocked(self) -> None:
        self._write("a.bin", b"\x00\x01")
        result = tool_read_file({"path": "a.bin"})
        self.assertIn("binary", result["error"])

    def test_read_python_symbol_range(self) -> None:
        content = (
            "@staticmethod\n"
            "def handy(a, b):\n"
            "    return a + b\n\n"
            "def other():\n"
            "    pass\n"
        )
        self._write("mod.py", content)
        result = tool_read_file({"path": "mod.py", "symbol": "handy"})
        self.assertEqual(result["total_lines"], 6)
        self.assertIn("handy", result["content"])
        self.assertNotIn("other", result["content"])

    def test_read_cpp_scoped_from_large_file(self) -> None:
        lines = "int alpha() {\n    return 1;\n}\n"
        big = ("// filler\n" * 250) + lines
        self._write("big.cpp", big)
        # Unscoped read is refused for large C/C++ sources.
        unscoped = tool_read_file({"path": "big.cpp"})
        self.assertIn("error", unscoped)
        self.assertIn("total_lines", unscoped)
        # Scoped read succeeds.
        scoped = tool_read_file({"path": "big.cpp", "start_line": 251, "end_line": 253})
        self.assertNotIn("error", scoped)
        self.assertIn("alpha", scoped["content"])

    def test_line_range_bounds(self) -> None:
        self._write("data.txt", "1\n2\n3\n4\n5\n")
        result = tool_read_file({"path": "data.txt", "start_line": 2, "end_line": 4})
        self.assertEqual(result["showing"], "lines 2-4")
        self.assertEqual(result["content"], "2\n3\n4\n")

    def test_missing_symbol(self) -> None:
        self._write("mod.py", "x = 1\n")
        result = tool_read_file({"path": "mod.py", "symbol": "nope"})
        self.assertIn("error", result)

    def test_encoding_fallback(self) -> None:
        self._write("latin.txt", "caf\xe9 au lait\n".encode("latin-1"))
        result = tool_read_file({"path": "latin.txt"})
        self.assertIn("café", result["content"])

    def test_truncation_at_char_limit(self) -> None:
        self._write("long.txt", "z" * 30_000)
        result = tool_read_file({"path": "long.txt"})
        self.assertIn("truncated", result["content"])


class FindSymbolRangeTests(unittest.TestCase):
    def test_python_ast_exact_match(self) -> None:
        lines = ["class K:", "    def run(self):", "        pass", "", "def top():\n"]
        span = _find_python_symbol_range(lines, "K.run")
        self.assertIsNotNone(span)
        self.assertEqual(span, (2, 4))

    def test_python_decorator_start(self) -> None:
        lines = ["@deco", "def f():", "    pass"]
        span = _find_python_symbol_range(lines, "f")
        self.assertEqual(span, (2, 3))

    def test_python_regex_fallback(self) -> None:
        lines = ["x = 1", "def bare(a):", "    return a", "y = 2"]
        span = _find_python_symbol_range(lines, "bare")
        self.assertIsNotNone(span)
        self.assertEqual(span[0], 2)

    def test_python_missing(self) -> None:
        self.assertIsNone(_find_python_symbol_range(["a = 1"], "missing"))

    def test_cpp_symbol_range(self) -> None:
        lines = ["void do_it() {", "    work();", "}", "void other() {}"]
        span = _find_symbol_range(lines, "do_it")
        self.assertEqual(span, (1, 3))

    def test_cpp_symbol_missing(self) -> None:
        self.assertIsNone(_find_symbol_range(["int a;"], "frobnicate"))

    def test_expand_function_end(self) -> None:
        lines = ["void f() {", "    x();", "}"]
        self.assertGreaterEqual(_expand_function_end(lines, 0, 1), 3)

    def test_expand_non_function_noop(self) -> None:
        lines = ["hello", "world"]
        self.assertEqual(_expand_function_end(lines, 0, 1), 1)


class ToolWriteTests(FileToolsBase):
    def test_invalid_path(self) -> None:
        result = tool_write_file({"path": "../x.py", "content": "y"})
        self.assertIn("error", result)

    def test_valid_path_delegates_to_sandbox(self) -> None:
        with mock.patch(
            "ai_assistant.tools.builtin.file_tools.write_sandboxed_file",
            return_value={"success": True, "bytes_written": 3, "sandbox": "local-allowlist"},
        ) as write:
            result = tool_write_file({"path": "src/out.py", "content": "abc"})
        self.assertTrue(result["success"])
        write.assert_called_once_with(mock.ANY, "abc", self.project_root)

    def test_patch_file_exact_match(self) -> None:
        self._write("a.txt", "old content here")
        with mock.patch(
            "ai_assistant.tools.builtin.file_tools.write_sandboxed_file",
            return_value={"success": True, "bytes_written": 5, "sandbox": "local-allowlist"},
        ):
            result = tool_patch_file({"path": "a.txt", "find": "old", "replace": "new"})
        self.assertTrue(result["success"])
        self.assertEqual(result["replacements"], 1)

    def test_patch_file_ambiguous_match(self) -> None:
        self._write("a.txt", "dup dup")
        result = tool_patch_file({"path": "a.txt", "find": "dup", "replace": "x"})
        self.assertIn("exactly one", result["error"])

    def test_patch_file_empty_find(self) -> None:
        self._write("a.txt", "text")
        result = tool_patch_file({"path": "a.txt", "find": "", "replace": "x"})
        self.assertIn("must not be empty", result["error"])

    def test_replace_file_content(self) -> None:
        self._write("a.txt", "target here")
        with mock.patch(
            "ai_assistant.tools.builtin.file_tools.write_sandboxed_file",
            return_value={"success": True, "bytes_written": 4, "sandbox": "local-allowlist"},
        ):
            result = tool_replace_file_content(
                {"path": "a.txt", "targetContent": "target", "replacementContent": "done"}
            )
        self.assertTrue(result["success"])
        self.assertEqual(result["replacements"], 1)

    def test_replace_requires_single_match(self) -> None:
        self._write("a.txt", "t t")
        result = tool_replace_file_content(
            {"path": "a.txt", "targetContent": "t", "replacementContent": "u"}
        )
        self.assertIn("exactly one", result["error"])

    def test_multi_replace_sequential(self) -> None:
        self._write("a.txt", "a b c")
        with mock.patch(
            "ai_assistant.tools.builtin.file_tools.write_sandboxed_file",
            return_value={"success": True, "bytes_written": 3, "sandbox": "local-allowlist"},
        ) as write:
            result = tool_multi_replace_file_content(
                {
                    "path": "a.txt",
                    "replacements": [
                        {"targetContent": "a", "replacementContent": "1"},
                        {"targetContent": "c", "replacementContent": "3"},
                    ],
                }
            )
        self.assertTrue(result["success"])
        self.assertEqual(result["replacements"], 2)
        written = write.call_args.args[1]
        self.assertEqual(written, "1 b 3")

    def test_multi_replace_empty_list(self) -> None:
        self._write("a.txt", "x")
        result = tool_multi_replace_file_content({"path": "a.txt", "replacements": []})
        self.assertIn("must not be empty", result["error"])

    def test_create_directory_delegates(self) -> None:
        with mock.patch(
            "ai_assistant.tools.builtin.file_tools.create_directory",
            return_value={"success": True, "path": "src/new", "created": True},
        ) as create:
            result = tool_create_directory({"path": "src/new"})
        self.assertTrue(result["success"])
        create.assert_called_once_with(mock.ANY, self.project_root)


class CodeCitationHelpersTests(unittest.TestCase):
    def test_requested_code_symbol_citation(self) -> None:
        symbol = _requested_code_symbol("trích dẫn hàm Foo::Bar() trong file src/x.cpp")
        self.assertEqual(symbol, "Foo::Bar")

    def test_requested_code_symbol_python_def(self) -> None:
        self.assertEqual(_requested_code_symbol("show code def compute()"), "compute")

    def test_requested_code_symbol_none(self) -> None:
        self.assertIsNone(_requested_code_symbol("làm thế nào để tối ưu"))

    def test_citation_is_python(self) -> None:
        self.assertTrue(_citation_is_python("file utils.py, hàm load", "load"))
        self.assertTrue(_citation_is_python("def parse trong module x", "parse"))
        self.assertTrue(_citation_is_python("x", "a.b"))
        self.assertFalse(_citation_is_python("trích dẫn Foo::Bar trong src/x.cpp", "Foo::Bar"))


class BuildCodeCitationTests(FileToolsBase):
    def test_non_citation_task_returns_none(self) -> None:
        self.assertIsNone(_build_code_citation_result("tổng hợp văn bản", "s1", 0.0))

    def test_python_citation_flow(self) -> None:
        task = "trích dẫn mã nguồn function load_data() trong file src/loader.py"
        with (
            mock.patch(
                "ai_assistant.tools.builtin.file_tools._PYTHON_PATH_PATTERN",  # leave as-is
            ),
        ):
            pass  # pylint: disable=pointless-statement
        self._write("loader.py", "def load_data():\n    return 1\n")
        with (
            mock.patch("ai_assistant.tools.builtin.search_tools.tool_search_text") as search,
            mock.patch("ai_assistant.tools.builtin.file_tools.tool_read_file") as read,
        ):
            search.return_value = {
                "results": [{"file": "src/loader.py", "content": "    def load_data():", "line": 1}]
            }
            read.return_value = {
                "path": "src/loader.py",
                "content": "def load_data():\n    return 1\n",
                "showing": "lines 1-2",
            }
            result = _build_code_citation_result(task, "session-1", 0.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["session_id"], "session-1")
        final = next(s for s in result["steps"] if s["type"] == "final_answer")
        self.assertIn("def load_data", final["content"])
        self.assertIn("```python", final["content"])

    def test_cpp_citation_falls_back_to_header(self) -> None:
        task = "trích dẫn C++ hàm Render::init() trong src"
        with (
            mock.patch("ai_assistant.tools.builtin.search_tools.tool_search_text") as search,
            mock.patch("ai_assistant.tools.builtin.file_tools.tool_read_file") as read,
        ):
            search.side_effect = [
                {"results": []},  # first .cpp search -> nothing
                {"results": [{"file": "src/render.h", "content": "void init() {"}]},
            ]
            read.return_value = {
                "path": "src/render.h",
                "content": "void init() { ;; }",
                "showing": "lines 1-1",
            }
            result = _build_code_citation_result(task, "s2", 0.0)
        self.assertIsNotNone(result)
        self.assertIn("```cpp", next(s["content"] for s in result["steps"] if s["type"] == "final_answer"))

    def test_no_definition_returns_none(self) -> None:
        task = "trích dẫn hàm Ghost::gone() trong src"
        with mock.patch("ai_assistant.tools.builtin.search_tools.tool_search_text", return_value={"results": []}):
            self.assertIsNone(_build_code_citation_result(task, "s3", 0.0))

    def test_code_language_mapping(self) -> None:
        self.assertEqual(_code_language("main.cpp"), "cpp")
        self.assertEqual(_code_language("x.py"), "python")
        self.assertEqual(_code_language("lib.h"), "cpp")
        self.assertEqual(_code_language("readme.txt"), "text")

    def test_citation_path_hint(self) -> None:
        hint = _citation_path_hint("xem code trong folder/src/core/util.py")
        self.assertIsNotNone(hint)
        self.assertTrue(hint.endswith("src/core/util.py"))


if __name__ == "__main__":
    unittest.main()
