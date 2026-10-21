"""Unit tests for LSP client helpers and tools."""
from __future__ import annotations

import json
import os
import shutil
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
        with mock.patch("ai_assistant.tools.lsp_client.resolve_tool_binary", return_value=None):
            result = lsp_client._lsp_call("nonexist", [], str(self.root / "src/main.py"), 1, 0, "textDocument/definition")
        self.assertIn("not found", result["error"])

    def test_the_server_is_resolved_by_the_shared_resolver(self) -> None:
        """The client must not resolve the program itself.

        ``shutil.which`` used to do the job here, and on Windows it
        prepends the *current directory* to the search path — so a
        ``clangd.bat`` planted next to the process was found **and
        executed**, and the "not found on PATH" refusal could never fire
        because ``which`` had already found the planted file. The old test
        mocked that lookup and returned a *relative* name, which is exactly
        the shape a current-directory hit produces — so the suite stayed
        green while the control did not work.
        """
        with mock.patch(
            "ai_assistant.tools.lsp_client.resolve_tool_binary", return_value=None,
        ) as resolver:
            result = lsp_client._lsp_call("clangd", [], str(self.root / "src/main.py"), 1, 0, "textDocument/definition")
        resolver.assert_called_once_with("clangd")
        self.assertTrue(result.get("binary_missing"))

    def test_a_language_server_planted_in_the_working_directory_is_not_run(self) -> None:
        """The current directory is never consulted.

        The server's working directory is a directory the agent may write
        to, so a planted binary there would run with the user's privileges
        the first time a definition lookup happened.
        """
        with tempfile.TemporaryDirectory() as workspace:
            # Named the way the resolver would look for it: on Windows one of
            # the PATHEXT spellings, on POSIX the bare name. It also needs
            # the execute bit there, or the scan would skip it for a reason
            # that has nothing to do with the current directory.
            planted = Path(workspace) / ("clangd.bat" if os.name == "nt" else "clangd")
            planted.write_text("@echo PWNED\r\n", encoding="utf-8")
            os.chmod(planted, 0o700)
            original = os.getcwd()
            os.chdir(workspace)
            try:
                from ai_assistant.exec_policy import resolve_tool_binary
                resolved = resolve_tool_binary("clangd")
            finally:
                os.chdir(original)
        # The CI image may or may not ship a real clangd, so the assertion is
        # about the *planted* file: it must never be the one that is picked.
        self.assertFalse(
            resolved is not None
            and os.path.abspath(str(resolved)) == os.path.abspath(str(planted)),
            "a binary in the current directory must not be resolved",
        )

    def test_lsp_call_file_missing(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client.resolve_tool_binary", return_value="clangd"):
            result = lsp_client._lsp_call("clangd", [], str(self.root / "nope.py"), 1, 0, "textDocument/definition")
        self.assertIn("does not exist", result["error"])

    def test_lsp_call_initialize_timeout(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client.resolve_tool_binary", return_value="clangd"):
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
        with mock.patch("ai_assistant.tools.lsp_client.resolve_tool_binary", return_value="clangd"):
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

    def test_lsp_call_closes_both_pipes_on_the_normal_path(self) -> None:
        """stdout is closed alongside stdin on the way out, not just stdin —
        a leaked PIPE handle would emit a ResourceWarning when reaped."""
        with mock.patch("ai_assistant.tools.lsp_client.resolve_tool_binary", return_value="clangd"):
            with mock.patch("ai_assistant.tools.lsp_client.subprocess.Popen") as popen:
                proc = mock.MagicMock()
                proc.stdin = mock.MagicMock()
                proc.stdout = mock.MagicMock()
                init_body = json.dumps(
                    {"jsonrpc": "2.0", "id": 1, "result": {}}, ensure_ascii=False
                ).encode("utf-8")
                def_body = json.dumps(
                    {"jsonrpc": "2.0", "id": 2, "result": []}, ensure_ascii=False
                ).encode("utf-8")
                proc.stdout.readline.side_effect = [
                    f"Content-Length: {len(init_body)}\r\n".encode("ascii"),
                    b"\r\n",
                    f"Content-Length: {len(def_body)}\r\n".encode("ascii"),
                    b"\r\n",
                ]
                proc.stdout.read.side_effect = [init_body, def_body]
                proc.poll.return_value = None
                popen.return_value = proc
                lsp_client._lsp_call(
                    "clangd", [], str(self.root / "src/main.py"), 1, 0, "textDocument/definition"
                )
        proc.stdin.close.assert_called_once()
        proc.stdout.close.assert_called_once()
        # Closing the channel is followed by a clean terminate-and-reap.
        proc.terminate.assert_called_once()

    def test_lsp_call_server_error_in_response(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client.resolve_tool_binary", return_value="clangd"):
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
        with mock.patch("ai_assistant.tools.lsp_client.resolve_tool_binary", return_value="clangd"):
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
        with mock.patch("ai_assistant.tools.lsp_client.resolve_tool_binary", return_value="clangd"):
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

    def test_an_absolute_path_outside_the_project_is_refused(self) -> None:
        """The read scope is the LSP root, not the whole drive: an absolute
        path used to be handed to the language server verbatim."""
        outside = str(self.root.parent / "elsewhere.cpp")
        with mock.patch("ai_assistant.tools.lsp_client._lsp_call") as call:
            definition = lsp_client.tool_go_to_definition({"path": outside, "line": 1, "character": 0})
            references = lsp_client.tool_find_references({"path": outside, "line": 1, "character": 0})
        self.assertIn("outside", definition["error"])
        self.assertIn("outside", references["error"])
        call.assert_not_called()

    def test_a_sibling_directory_sharing_the_name_prefix_is_refused(self) -> None:
        """``<root>-evil`` shares the root as a string prefix: only a real
        containment check (the kernel's ``resolve_within``) rejects it."""
        sibling = os.path.join(str(self.root) + "-evil", "x.cpp")
        with mock.patch("ai_assistant.tools.lsp_client._lsp_call") as call:
            result = lsp_client.tool_go_to_definition({"path": sibling, "line": 1, "character": 0})
        self.assertIn("outside", result["error"])
        call.assert_not_called()

    def test_an_absolute_path_inside_the_project_still_runs(self) -> None:
        with mock.patch("ai_assistant.tools.lsp_client._lsp_call", return_value={"result": []}):
            result = lsp_client.tool_go_to_definition(
                {"path": str(self.root / "src" / "main.cpp"), "line": 1, "character": 4},
            )
        self.assertFalse(result["found"])

    def test_a_containment_check_that_cannot_be_made_is_an_escape(self) -> None:
        """Different Windows drives share no root: that is an escape, not a
        crash. The comparison itself lives in the kernel
        (``path_policy.resolve_within``)."""
        with mock.patch(
            "ai_assistant.path_policy.os.path.commonpath",
            side_effect=ValueError("mix of drives"),
        ):
            self.assertIsNone(lsp_client._confine_to_project("src/main.cpp", str(self.root)))

    def test_a_junction_out_of_the_project_is_refused(self) -> None:
        """The *real* target decides, not the spelling.

        A junction inside the project that points outside it is an escape,
        and a lexical comparison cannot see it — the file was read through
        a path the gate accepted.
        """
        outside = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(outside, ignore_errors=True))
        Path(outside, "secret.py").write_text("SECRET = 1\n", encoding="utf-8")
        link = Path(self.root) / "escape"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not available on this filesystem")
        self.assertIsNone(lsp_client._confine_to_project(str(link / "secret.py"), str(self.root)))


class ReadResponseEdgeCaseTests(unittest.TestCase):
    """The framing reader's four reject/loop paths.

    These drive :func:`_read_response` directly with a scripted stream and a
    scripted clock, because every one of them is a timing or malformed-header
    condition that a live subprocess cannot be made to produce reliably.
    """

    @staticmethod
    def _reader(lines: list[bytes], reads: list[bytes]):
        line_iter = iter(lines)
        read_iter = iter(reads)
        stdout = mock.MagicMock()
        stdout.readline.side_effect = lambda: next(line_iter, b"")
        stdout.read.side_effect = lambda _n: next(read_iter, b"")
        return SimpleNamespace(stdout=stdout, poll=lambda: None)

    @staticmethod
    def _clock(values: list[float]):
        """A monotonic stand-in that stays valid after the script runs out."""
        ticks = iter(values)
        return mock.patch.object(
            lsp_client.time, "monotonic", side_effect=lambda: next(ticks, 1e9),
        )

    def test_a_non_content_length_header_is_skipped(self):
        """An unrelated header must fall back to the top of the read loop."""
        proc = self._reader(
            [b"X-Other: 1\r\n", b"Content-Length: 2\r\n", b"\r\n"], [b"{}"],
        )
        with self._clock([0.0, 0.0, 0.0, 0.0, 0.0]):
            self.assertEqual(lsp_client._read_response(proc, timeout=5.0), {})

    def test_a_zero_content_length_is_rejected(self):
        """``Content-Length: 0`` is malformed framing, not an empty message."""
        proc = self._reader([b"Content-Length: 0\r\n", b"\r\n"], [])
        with self._clock([0.0, 0.0, 0.0]):
            self.assertIsNone(lsp_client._read_response(proc, timeout=5.0))

    def test_a_body_that_never_arrives_gives_up_at_the_deadline(self):
        proc = self._reader([b"Content-Length: 4\r\n", b"\r\n"], [])
        # Header reads stay inside the window; the first body read does not.
        with self._clock([0.0, 0.0, 0.0, 100.0]):
            self.assertIsNone(lsp_client._read_response(proc, timeout=5.0))

    def test_an_empty_chunk_is_retried_instead_of_endless_framing(self):
        proc = self._reader([b"Content-Length: 2\r\n", b"\r\n"], [b"", b"{}"])
        with self._clock([0.0, 0.0, 0.0, 0.0, 0.0]):
            self.assertEqual(lsp_client._read_response(proc, timeout=5.0), {})

    def test_a_body_that_is_not_json_is_rejected(self):
        proc = self._reader([b"Content-Length: 5\r\n", b"\r\n"], [b"notjs"])
        with self._clock([0.0, 0.0, 0.0, 0.0]):
            self.assertIsNone(lsp_client._read_response(proc, timeout=5.0))


class ProjectDirFallbackTests(unittest.TestCase):
    """Pin the repo-root fallback computed by ``_get_project_dir``."""

    @staticmethod
    def _resolve_without_config() -> Path:
        with mock.patch("ai_assistant.tools.lsp_client.get_paths",
                        side_effect=RuntimeError("no config")):
            return Path(lsp_client._get_project_dir())

    def test_an_unreadable_path_config_falls_back_to_the_repo_root(self):
        resolved = self._resolve_without_config()
        self.assertTrue(resolved.is_absolute())
        # The fallback is four levels above ``src/ai_assistant/tools``, so the
        # repository checkout must be reachable from the result.
        self.assertTrue((resolved / "AIAssistant" / "src" / "ai_assistant").is_dir())

    def test_a_uri_that_cannot_be_made_relative_falls_back_to_the_raw_path(self):
        """``os.path.relpath`` can reject a path (cross-drive, unreadable base).

        The conversion is best-effort, so the raw path must survive rather than
        the location lookup failing the whole reference list.
        """
        from urllib.request import url2pathname
        # A platform-native absolute path, not a hard-coded "D:/…": a file
        # URI is what the converter consumes, and "D:/x" is a *relative*
        # path (so as_uri() raises) on POSIX.
        raw = os.path.abspath("elsewhere/module.py")
        uri = Path(raw).as_uri()
        with mock.patch("ai_assistant.tools.lsp_client._get_project_dir",
                        return_value="C:/"), \
                mock.patch.object(lsp_client.os.path, "relpath",
                                  side_effect=ValueError("different drives")):
            resolved = lsp_client._uri_to_rel_path(uri)
        self.assertEqual(resolved, url2pathname(uri[7:]))


class LspCallHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "main.py").write_text("def f(): pass\n", encoding="utf-8")
        for target, value in (
            ("ai_assistant.tools.lsp_client.get_paths",
             SimpleNamespace(project_root=str(self.root))),
            ("ai_assistant.tools.lsp_client.resolve_tool_binary", "pylsp"),
        ):
            patcher = mock.patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _proc(*responses: bytes) -> mock.MagicMock:
        proc = mock.MagicMock()
        lines: list[bytes] = []
        reads: list[bytes] = []
        for body in responses:
            lines += [f"Content-Length: {len(body)}\r\n".encode("ascii"), b"\r\n"]
            reads.append(body)
        line_iter = iter(lines)
        read_iter = iter(reads)
        proc.stdout.readline.side_effect = lambda: next(line_iter, b"")
        proc.stdout.read.side_effect = lambda _n: next(read_iter, b"")
        proc.poll.return_value = None
        return proc

    def test_an_unreadable_document_is_reported_not_raised(self):
        init = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode()
        with mock.patch("ai_assistant.tools.lsp_client.subprocess.Popen",
                        return_value=self._proc(init, b"{}")):
            with mock.patch("builtins.open", side_effect=OSError("permission denied")):
                result = lsp_client._lsp_call(
                    "pylsp", [], str(self.root / "src/main.py"), 1, 0,
                    "textDocument/definition",
                )
        self.assertIn("Could not read", result["error"])
        self.assertIn("permission denied", result["error"])

    def test_extra_request_params_are_merged_into_the_position_payload(self):
        init = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode()
        final = json.dumps({"jsonrpc": "2.0", "id": 2, "result": []}).encode()
        proc = self._proc(init, final)
        with mock.patch("ai_assistant.tools.lsp_client.subprocess.Popen", return_value=proc):
            result = lsp_client._lsp_call(
                "pylsp", [], str(self.root / "src/main.py"), 3, 2,
                "textDocument/references",
                extra_params={"context": {"includeDeclaration": True}},
            )
        self.assertEqual(result, {"result": []})
        # The second write is the request itself, carrying the merged params.
        request = proc.stdin.write.call_args_list[-1].args[0]
        payload = json.loads(request.split(b"\r\n\r\n", 1)[1])
        self.assertEqual(payload["params"]["context"], {"includeDeclaration": True})
        self.assertEqual(payload["params"]["position"], {"line": 2, "character": 2})

    def test_a_failure_while_closing_the_channel_falls_back_to_kill(self):
        init = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode()
        proc = self._proc(init, b"{}")
        proc.stdin.close.side_effect = RuntimeError("pipe already gone")
        with mock.patch("ai_assistant.tools.lsp_client.subprocess.Popen", return_value=proc):
            lsp_client._lsp_call("pylsp", [], str(self.root / "src/main.py"), 1, 0,
                                 "textDocument/definition")
        proc.kill.assert_called_once()
        proc.terminate.assert_not_called()

    def test_a_kill_that_also_fails_is_swallowed(self):
        """Cleanup must never mask the response the caller is about to read."""
        init = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode()
        proc = self._proc(init, b"{}")
        proc.stdin.close.side_effect = RuntimeError("pipe already gone")
        proc.kill.side_effect = RuntimeError("already reaped")
        with mock.patch("ai_assistant.tools.lsp_client.subprocess.Popen", return_value=proc):
            result = lsp_client._lsp_call(
                "pylsp", [], str(self.root / "src/main.py"), 1, 0,
                "textDocument/definition",
            )
        self.assertIn("result", result)


if __name__ == "__main__":
    unittest.main()
