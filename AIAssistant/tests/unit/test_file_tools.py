"""Unit tests for file read/write/patch and directory tools."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_assistant.tools.builtin.file_tools import (
    _expand_function_end,
    _find_python_symbol_range,
    _find_symbol_range,
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


class FileToolGuardPathTests(FileToolsBase):
    """Every reject/except arm of the read, patch, replace and directory tools."""

    # -- tool_read_file

    def test_read_file_reports_when_no_encoding_fits(self) -> None:
        """All four encodings are tried; exhausting them is an error, not a crash."""
        self._write("data.py", "value = 1\n")
        with mock.patch("builtins.open", side_effect=ValueError("bad codec")):
            result = tool_read_file({"path": "data.py"})
        self.assertIn("Không đọc được encoding", result["error"])

    def test_read_file_reports_an_unexpected_failure(self) -> None:
        """Anything the encoding loop cannot handle falls to the outer guard."""
        self._write("data.py", "value = 1\n")
        with mock.patch("builtins.open", side_effect=RuntimeError("descriptor lost")):
            result = tool_read_file({"path": "data.py"})
        self.assertIn("Lỗi đọc file", result["error"])

    def test_reading_a_class_member_walks_the_ast_class_node(self) -> None:
        """The AST path only runs on a real file, so ``''.join`` stays valid."""
        self._write("model.py", "class Widget:\n    def render(self):\n        return 1\n")
        result = tool_read_file({"path": "model.py", "symbol": "Widget.render"})
        self.assertNotIn("error", result)
        self.assertIn("return 1", result["content"])

    # -- _find_symbol_range / _expand_function_end

    def test_symbol_range_skips_a_qualification_the_line_does_not_carry(self) -> None:
        """The unqualified name matches, so neither candidate list can fill."""
        lines = ["void foo(", ");"]
        self.assertIsNone(_find_symbol_range(lines, "ns::foo"))

    def test_symbol_range_keeps_the_open_range_when_braces_never_close(self) -> None:
        lines = ["void foo() {", "  int x;"]
        self.assertEqual(_find_symbol_range(lines, "foo"), (1, 2))

    def test_expand_function_end_keeps_the_original_when_braces_never_close(self) -> None:
        lines = ["void f() {", "    x();", "    y();"]
        self.assertEqual(_expand_function_end(lines, 0, 1), 1)

    # -- tool_patch_file

    def test_patch_rejects_a_missing_file(self) -> None:
        result = tool_patch_file({"path": "ghost.txt", "find": "a", "replace": "b"})
        self.assertIn("Invalid or missing file", result["error"])

    def test_patch_propagates_a_sandbox_refusal(self) -> None:
        self._write("a.txt", "hello world")
        with mock.patch(
            "ai_assistant.tools.builtin.file_tools.write_sandboxed_file",
            return_value={"error": "outside the sandbox"},
        ):
            result = tool_patch_file({"path": "a.txt", "find": "hello", "replace": "bye"})
        self.assertEqual(result["error"], "outside the sandbox")

    def test_patch_reports_a_read_failure(self) -> None:
        self._write("a.txt", "hello world")
        with mock.patch("builtins.open", side_effect=PermissionError("locked")):
            result = tool_patch_file({"path": "a.txt", "find": "hello", "replace": "bye"})
        self.assertIn("Unable to patch file", result["error"])

    # -- tool_replace_file_content

    def test_replace_rejects_a_missing_file(self) -> None:
        result = tool_replace_file_content(
            {"path": "ghost.txt", "targetContent": "a", "replacementContent": "b"},
        )
        self.assertIn("Invalid or missing file", result["error"])

    def test_replace_rejects_an_empty_target(self) -> None:
        self._write("a.txt", "hello")
        result = tool_replace_file_content(
            {"path": "a.txt", "targetContent": "", "replacementContent": "b"},
        )
        self.assertEqual(result["error"], "Target content must not be empty.")

    def test_replace_propagates_a_sandbox_refusal(self) -> None:
        self._write("a.txt", "hello world")
        with mock.patch(
            "ai_assistant.tools.builtin.file_tools.write_sandboxed_file",
            return_value={"error": "read-only volume"},
        ):
            result = tool_replace_file_content(
                {"path": "a.txt", "targetContent": "hello", "replacementContent": "bye"},
            )
        self.assertEqual(result["error"], "read-only volume")

    def test_replace_reports_a_read_failure(self) -> None:
        self._write("a.txt", "hello world")
        with mock.patch("builtins.open", side_effect=PermissionError("locked")):
            result = tool_replace_file_content(
                {"path": "a.txt", "targetContent": "hello", "replacementContent": "bye"},
            )
        self.assertIn("Unable to replace file content", result["error"])

    # -- tool_multi_replace_file_content

    def test_multi_replace_rejects_a_missing_file(self) -> None:
        result = tool_multi_replace_file_content(
            {"path": "ghost.txt", "replacements": [{"targetContent": "a"}]},
        )
        self.assertIn("Invalid or missing file", result["error"])

    def test_multi_replace_names_the_index_with_an_empty_target(self) -> None:
        self._write("a.txt", "hello world")
        result = tool_multi_replace_file_content(
            {"path": "a.txt", "replacements": [
                {"targetContent": "hello", "replacementContent": "bye"},
                {"targetContent": "", "replacementContent": "x"},
            ]},
        )
        self.assertIn("at index 1", result["error"])

    def test_multi_replace_names_the_index_with_an_ambiguous_target(self) -> None:
        self._write("a.txt", "aa aa")
        result = tool_multi_replace_file_content(
            {"path": "a.txt", "replacements": [
                {"targetContent": "aa", "replacementContent": "b"},
            ]},
        )
        self.assertIn("at index 0", result["error"])
        self.assertIn("found 2", result["error"])

    def test_multi_replace_propagates_a_sandbox_refusal(self) -> None:
        self._write("a.txt", "hello world")
        with mock.patch(
            "ai_assistant.tools.builtin.file_tools.write_sandboxed_file",
            return_value={"error": "read-only volume"},
        ):
            result = tool_multi_replace_file_content(
                {"path": "a.txt", "replacements": [
                    {"targetContent": "hello", "replacementContent": "bye"},
                ]},
            )
        self.assertEqual(result["error"], "read-only volume")

    def test_multi_replace_reports_a_read_failure(self) -> None:
        self._write("a.txt", "hello world")
        with mock.patch("builtins.open", side_effect=PermissionError("locked")):
            result = tool_multi_replace_file_content(
                {"path": "a.txt", "replacements": [
                    {"targetContent": "hello", "replacementContent": "bye"},
                ]},
            )
        self.assertIn("Unable to multi-replace file content", result["error"])

    # -- tool_create_directory

    def test_create_directory_rejects_an_unsafe_path(self) -> None:
        result = tool_create_directory({"path": "../escape"})
        self.assertIn("Invalid directory path", result["error"])


if __name__ == "__main__":
    unittest.main()
