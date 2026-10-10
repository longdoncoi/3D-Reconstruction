"""Unit tests for LSP client helpers and tools."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_assistant.tools import lsp_client


class FakeProc:
    def __init__(self, responses: list[bytes | None]) -> None:
        self._responses = responses
        self.stdin = mock.MagicMock()
        self.stdout = mock.MagicMock()
        self._read_index = 0
        self._poll_calls = 0

    def poll(self) -> int | None:
        self._poll_calls += 1
        return None if self._read_index < len(self._responses) else 0

    def terminate(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        pass


class LspClientHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        patcher = mock.patch(
            "ai_assistant.tools.lsp_client.get_paths",
            return_value=SimpleNamespace(project_root=str(self.root)),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self._write("src/main.py", "def hello():\n    pass\n")

    def _write(self, name: str, content: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    # -- message framing
    def test_make_request_and_notification(self) -> None:
        req = lsp_client._make_request("initialize", {"x": 1}, req_id=5)
        self.assertTrue(req.startswith(b"Content-Length:"))
        self.assertIn(b"initialize", req)
        note = lsp_client._make_notification("initialized", {})
        self.assertIn(b"initialized", note)

    # -- response reader
    def test_read_response_header_and_body(self) -> None:
        body = json.dumps({"id": 1, "result": {"ok": True}}, ensure_ascii=False).encode("utf-8")
        stdout = mock.MagicMock()
        # The reader consumes header lines one by one: the length line, then the
        # blank line that terminates the header block.
        stdout.readline.side_effect = [
            f"Content-Length: {len(body)}\r\n".encode("ascii"),
            b"\r\n",
        ]
        stdout.read.return_value = body
        proc = SimpleNamespace(stdout=stdout, poll=lambda: None)
        resp = lsp_client._read_response(proc, timeout=1.0)
        self.assertEqual(resp, {"id": 1, "result": {"ok": True}})

    def test_read_response_timeout_header(self) -> None:
        stdout = mock.MagicMock()
        stdout.readline.return_value = b""
        proc = SimpleNamespace(stdout=stdout, poll=lambda: 1)
        self.assertIsNone(lsp_client._read_response(proc, timeout=0.01))

    def test_read_response_invalid_length(self) -> None:
        # An unparsable length keeps the reader in the header loop until the
        # deadline expires, so readline must answer forever instead of raising.
        lines = iter([b"Content-Length: abc\r\n", b"\r\n"])
        stdout = mock.MagicMock()
        stdout.readline.side_effect = lambda: next(lines, b"")
        stdout.read.return_value = b""
        proc = SimpleNamespace(stdout=stdout, poll=lambda: None)
        self.assertIsNone(lsp_client._read_response(proc, timeout=0.05))

    def test_read_response_body_timeout(self) -> None:
        body = b"{}"
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        stdout = mock.MagicMock()
        stdout.readline.side_effect = [header, b""]
        # No body chunks returned; poll None to simulate still running? body read loops -> timeout path
        stdout.read.return_value = b""
        proc = SimpleNamespace(stdout=stdout, poll=lambda: None)
        self.assertIsNone(lsp_client._read_response(proc, timeout=0.01))

    # -- uri helpers
    def test_file_uri_and_rel_path(self) -> None:
        abs_path = str((self.root / "src" / "main.py").resolve())
        uri = lsp_client._file_uri(abs_path)
        self.assertTrue(uri.startswith("file://"))
        rel = lsp_client._uri_to_rel_path(uri)
        self.assertEqual(rel.replace("\\", "/"), "src/main.py")

    def test_uri_to_rel_path_unknown_scheme(self) -> None:
        self.assertEqual(lsp_client._uri_to_rel_path("https://example.com/x"), "https://example.com/x")

    # -- format locations
    def test_format_locations_variants(self) -> None:
        abs_header = self._write("src/a.h", "// header\n")
        abs_target = self._write("b.cpp", "int y;\n")
        abs_plain = self._write("c.cpp", "int z;\n")
        raw = [
            {
                "uri": lsp_client._file_uri(str(abs_header)),
                "range": {"start": {"line": 5, "character": 3}},
            },
            {
                "targetUri": lsp_client._file_uri(str(abs_target)),
                "targetRange": {"start": {"line": 1, "character": 0}},
            },
            {
                "uri": lsp_client._file_uri(str(abs_plain)),
                "targetSelectionRange": {"start": {"line": 0, "character": 2}},
            },
            "not-a-dict",
        ]
        formatted = lsp_client._format_locations(raw, "def")
        self.assertEqual(len(formatted), 3)
        self.assertEqual(formatted[0]["path"].replace("\\", "/"), "src/a.h")
        self.assertEqual(formatted[0]["line"], 6)
        self.assertEqual(formatted[0]["character"], 3)
        self.assertEqual(formatted[1]["path"].replace("\\", "/"), "b.cpp")
        self.assertEqual(formatted[1]["line"], 2)
        self.assertEqual(formatted[2]["line"], 1)  # 0-based -> +1
        self.assertEqual(lsp_client._format_locations(None, "x"), [])
        # A falsy non-list result is an empty result, not an entry to unwrap.
        self.assertEqual(lsp_client._format_locations({}, "x"), [])
        # A non-empty dict is wrapped so its single location is still formatted.
        single = lsp_client._format_locations(
            {"uri": lsp_client._file_uri(str(abs_plain)),
             "range": {"start": {"line": 9, "character": 1}}},
            "x",
        )
        self.assertEqual(len(single), 1)
        self.assertEqual(single[0]["path"].replace("\\", "/"), "c.cpp")
        self.assertEqual(single[0]["line"], 10)
        self.assertEqual(single[0]["character"], 1)

    # -- lsp call
    def test_lsp_call_binary_missing(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client.shutil.which", return_value=None):
            result = lsp_client._lsp_call("nonexist", [], str(self.root / "src/main.py"), 1, 0, "textDocument/definition")
        self.assertIn("not found", result["error"])

    def test_lsp_call_file_missing(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client.shutil.which", return_value="clangd"):
            result = lsp_client._lsp_call("clangd", [], str(self.root / "nope.py"), 1, 0, "textDocument/definition")
        self.assertIn("does not exist", result["error"])

    def test_lsp_call_initialize_timeout(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client.shutil.which", return_value="clangd"):
            with mock.patch("ai_assistant.tools.lsp_client.subprocess.Popen") as popen:
                proc = mock.MagicMock()
                proc.stdin = mock.MagicMock()
                proc.stdout = mock.MagicMock()
                proc.stdout.readline.side_effect = [b"", b""]
                proc.poll.return_value = 1
                popen.return_value = proc
                result = lsp_client._lsp_call("clangd", [], str(self.root / "src/main.py"), 1, 0, "textDocument/definition")
        self.assertIn("did not respond", result["error"])

    def test_lsp_call_happy_path(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client.shutil.which", return_value="clangd"):
            with mock.patch("ai_assistant.tools.lsp_client.subprocess.Popen") as popen:
                proc = mock.MagicMock()
                proc.stdin = mock.MagicMock()
                proc.stdout = mock.MagicMock()
                init_body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}, ensure_ascii=False).encode("utf-8")
                def_body = json.dumps({"jsonrpc": "2.0", "id": 2, "result": [{"uri": lsp_client._file_uri(str(self.root / "src/main.py")), "range": {"start": {"line": 0, "character": 4}}}]}, ensure_ascii=False).encode("utf-8")
                proc.stdout.readline.side_effect = [
                    f"Content-Length: {len(init_body)}\r\n".encode("ascii"),
                    b"\r\n",
                    f"Content-Length: {len(def_body)}\r\n".encode("ascii"),
                    b"\r\n",
                ]
                proc.stdout.read.side_effect = [init_body, def_body]
                proc.poll.return_value = None
                popen.return_value = proc
                result = lsp_client._lsp_call("clangd", [], str(self.root / "src/main.py"), 1, 0, "textDocument/definition")
        self.assertIn("result", result)
        self.assertIsInstance(result["result"], list)

    def test_lsp_call_server_error_in_response(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client.shutil.which", return_value="clangd"):
            with mock.patch("ai_assistant.tools.lsp_client.subprocess.Popen") as popen:
                proc = mock.MagicMock()
                proc.stdin = mock.MagicMock()
                proc.stdout = mock.MagicMock()
                init_body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}, ensure_ascii=False).encode("utf-8")
                err_body = json.dumps({"jsonrpc": "2.0", "id": 2, "error": {"message": "boom"}}, ensure_ascii=False).encode("utf-8")
                proc.stdout.readline.side_effect = [
                    f"Content-Length: {len(init_body)}\r\n".encode("ascii"),
                    b"\r\n",
                    f"Content-Length: {len(err_body)}\r\n".encode("ascii"),
                    b"\r\n",
                ]
                proc.stdout.read.side_effect = [init_body, err_body]
                popen.return_value = proc
                result = lsp_client._lsp_call("clangd", [], str(self.root / "src/main.py"), 1, 0, "textDocument/definition")
        self.assertIn("LSP server error", result["error"])

    def test_lsp_call_request_timeout(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client.shutil.which", return_value="clangd"):
            with mock.patch("ai_assistant.tools.lsp_client.subprocess.Popen") as popen:
                proc = mock.MagicMock()
                proc.stdin = mock.MagicMock()
                proc.stdout = mock.MagicMock()
                init_body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}, ensure_ascii=False).encode("utf-8")
                # Initialize succeeds (header + blank line + body); the follow-up
                # request then blocks on an empty stream until the deadline.
                proc.stdout.readline.side_effect = [
                    f"Content-Length: {len(init_body)}\r\n".encode("ascii"),
                    b"\r\n",
                    b"",
                ]
                proc.stdout.read.return_value = init_body
                proc.poll.return_value = 1
                popen.return_value = proc
                result = lsp_client._lsp_call("clangd", [], str(self.root / "src/main.py"), 1, 0, "textDocument/definition")
        self.assertIn("timed out", result["error"])

    def test_lsp_call_popen_exception(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client.shutil.which", return_value="clangd"):
            with mock.patch("ai_assistant.tools.lsp_client.subprocess.Popen", side_effect=RuntimeError("spawn failed")):
                result = lsp_client._lsp_call("clangd", [], str(self.root / "src/main.py"), 1, 0, "textDocument/definition")
        self.assertIn("subprocess failed", result["error"])


class LspToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "main.cpp").write_text("int x;")
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "mod.py").write_text("def f(): pass\n")
        patcher = mock.patch("ai_assistant.tools.lsp_client.get_paths", return_value=SimpleNamespace(project_root=str(self.root)))
        patcher.start()
        self.addCleanup(patcher.stop)
        self._binaries = mock.patch("ai_assistant.tools.lsp_client.lsp_binaries", return_value=("clangd", "pylsp", 15))
        self._binaries.start()
        self.addCleanup(self._binaries.stop)

    def test_go_to_definition_cpp(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client._lsp_call", return_value={
            "result": [{
                "uri": lsp_client._file_uri(str(self.root / "src/main.cpp")),
                "range": {"start": {"line": 0, "character": 0}},
            }]
        }):
            result = lsp_client.tool_go_to_definition({"path": "src/main.cpp", "line": 1, "character": 4})
        self.assertTrue(result["found"])
        self.assertIn("definitions", result)

    def test_go_to_definition_python(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client._lsp_call", return_value={"result": []}):
            result = lsp_client.tool_go_to_definition({"path": "pkg/mod.py", "line": 1, "character": 4})
        self.assertFalse(result["found"])
        self.assertEqual(result["message"], "No definition found.")

    def test_go_to_definition_missing_path(self) -> None:
        result = lsp_client.tool_go_to_definition({"line": 1})
        self.assertIn("required", result["error"])

    def test_go_to_definition_unsupported_ext(self) -> None:
        result = lsp_client.tool_go_to_definition({"path": "readme.txt", "line": 1, "character": 0})
        self.assertIn("Unsupported", result["error"])

    def test_go_to_definition_lsp_error_propagated(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client._lsp_call", return_value={"error": "server down"}):
            result = lsp_client.tool_go_to_definition({"path": "src/main.cpp", "line": 1, "character": 0})
        self.assertEqual(result, {"error": "server down"})

    def test_find_references(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client._lsp_call", return_value={
            "result": [
                {"uri": lsp_client._file_uri(str(self.root / "src/main.cpp")), "range": {"start": {"line": 2, "character": 1}}},
                {"targetUri": lsp_client._file_uri(str(self.root / "src/main.cpp")), "targetRange": {"start": {"line": 3, "character": 0}}},
            ]
        }):
            result = lsp_client.tool_find_references({"path": "src/main.cpp", "line": 1, "character": 0})
        self.assertTrue(result["found"])
        self.assertEqual(result["count"], 2)
        self.assertIn("references", result)

    def test_find_references_not_found(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client._lsp_call", return_value={"result": None}):
            result = lsp_client.tool_find_references({"path": "pkg/mod.py", "line": 1, "character": 0})
        self.assertFalse(result["found"])

    def test_find_references_missing_path(self) -> None:
        result = lsp_client.tool_find_references({})
        self.assertIn("required", result["error"])

    def test_find_references_unsupported_ext(self) -> None:
        result = lsp_client.tool_find_references({"path": "notes.md", "line": 1, "character": 0})
        self.assertIn("Unsupported", result["error"])

    def test_find_references_error(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client._lsp_call", return_value={"error": "boom"}):
            result = lsp_client.tool_find_references({"path": "src/main.cpp", "line": 1, "character": 0})
        self.assertEqual(result["error"], "boom")


if __name__ == "__main__":
    unittest.main()
