"""LSP (Language Server Protocol) client wrapper for Coding Agent tools.

Provides ``go_to_definition`` and ``find_references`` by spawning clangd
(for C++/Qt files) or pylsp (for Python files) as a subprocess and
communicating via stdio using the LSP JSON-RPC protocol.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from ai_assistant.config.paths import get_paths
from ai_assistant.exec_policy import resolve_tool_binary
from ai_assistant.path_policy import resolve_within
from ai_assistant.settings import lsp_binaries

_CPP_EXTS = {".cpp", ".c", ".h", ".hpp", ".cc", ".cxx"}
_PY_EXTS = {".py"}


def _get_project_dir() -> str:
    try:
        return str(get_paths().project_root)
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _make_request(method: str, params: dict[str, Any], req_id: int = 1) -> bytes:
    body = json.dumps(
        {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params},
        ensure_ascii=False,
    ).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode()
    return header + body


def _make_notification(method: str, params: dict[str, Any]) -> bytes:
    body = json.dumps(
        {"jsonrpc": "2.0", "method": method, "params": params},
        ensure_ascii=False,
    ).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode()
    return header + body


def _read_response(proc: subprocess.Popen[bytes], timeout: float = 10.0) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    content_length: int | None = None

    while True:
        if time.monotonic() > deadline:
            return None
        line = proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            if proc.poll() is not None:
                return None
            time.sleep(0.02)
            continue
        line_str = line.decode("latin1", errors="replace").strip()
        if not line_str:
            if content_length is not None:
                break
            continue
        if line_str.lower().startswith("content-length:"):
            try:
                content_length = int(line_str.split(":", 1)[1].strip())
            except ValueError:
                pass

    if content_length is None or content_length <= 0:
        return None

    raw_body = bytearray()
    while len(raw_body) < content_length:
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            return None
        chunk = proc.stdout.read(content_length - len(raw_body))  # type: ignore[union-attr]
        if not chunk:
            time.sleep(0.02)
            continue
        raw_body.extend(chunk)

    try:
        response: dict[str, Any] = json.loads(raw_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return response


def _file_uri(abs_path: str) -> str:
    path_obj = Path(abs_path).resolve()
    return path_obj.as_uri()


def _uri_to_rel_path(uri: str) -> str:
    if uri.startswith("file:///"):
        from urllib.request import url2pathname
        raw = url2pathname(uri[7:])
        project_dir = _get_project_dir()
        try:
            return os.path.relpath(raw, project_dir)
        except ValueError:
            return raw
    return uri


def _format_locations(raw_result: Any, label: str) -> list[dict[str, Any]]:
    if not raw_result:
        return []
    items: list[dict[str, Any]] = raw_result if isinstance(raw_result, list) else [raw_result]
    formatted = []
    for item in items:
        if not isinstance(item, dict):
            continue
        uri = item.get("uri") or item.get("targetUri", "")
        range_dict = (
            item.get("range")
            or item.get("targetSelectionRange")
            or item.get("targetRange")
            or {}
        )
        start = range_dict.get("start", {})
        line = int(start.get("line", 0)) + 1
        char = int(start.get("character", 0))
        rel_path = _uri_to_rel_path(uri) if uri else "unknown"
        formatted.append({
            "path": rel_path,
            "line": line,
            "character": char,
            "uri": uri,
        })
    return formatted


def _lsp_call(
    server_bin: str,
    server_args: list[str],
    abs_path: str,
    line_1based: int,
    char_0based: int,
    method: str,
    extra_params: dict[str, Any] | None = None,
    timeout: int = 15,
) -> dict[str, Any]:
    # The language server is resolved by the *shared* resolver
    # (:mod:`ai_assistant.exec_policy`), which walks PATH and never looks at
    # the current directory. ``shutil.which`` used to do the job here, and on
    # Windows it prepends the current directory to the search path — so a
    # ``clangd.bat`` planted next to the process was found *and executed*,
    # and the "not found on PATH" refusal could never fire because ``which``
    # had already found the planted file.
    resolved_bin = resolve_tool_binary(server_bin)
    if not resolved_bin:
        return {
            "error": (
                f"LSP server executable '{server_bin}' was not found on PATH. "
                f"Please ensure it is installed and added to PATH, or configure "
                f"AGENT_CLANGD_BIN / AGENT_PYLSP_BIN."
            ),
            "binary_missing": True,
        }

    if not os.path.isfile(abs_path):
        return {"error": f"File does not exist: {abs_path}"}

    project_dir = _get_project_dir()
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            [resolved_bin, *server_args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        init_params = {
            "processId": os.getpid(),
            "rootUri": _file_uri(project_dir),
            "capabilities": {},
        }
        proc.stdin.write(_make_request("initialize", init_params, req_id=1))  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]

        init_resp = _read_response(proc, timeout=5.0)
        if init_resp is None:
            return {"error": f"LSP server '{server_bin}' did not respond to initialize."}

        proc.stdin.write(_make_notification("initialized", {}))  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]

        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                doc_text = f.read()
        except OSError as e:
            return {"error": f"Could not read {abs_path}: {e}"}

        did_open_params = {
            "textDocument": {
                "uri": _file_uri(abs_path),
                "languageId": "cpp" if abs_path.endswith((".cpp", ".h", ".hpp", ".cc")) else "python",
                "version": 1,
                "text": doc_text,
            }
        }
        proc.stdin.write(_make_notification("textDocument/didOpen", did_open_params))  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]

        req_params: dict[str, Any] = {
            "textDocument": {"uri": _file_uri(abs_path)},
            "position": {
                "line": max(0, line_1based - 1),
                "character": max(0, char_0based),
            },
        }
        if extra_params:
            req_params.update(extra_params)

        proc.stdin.write(_make_request(method, req_params, req_id=2))  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]

        resp = _read_response(proc, timeout=float(timeout))
        if resp is None:
            return {"error": f"LSP request '{method}' timed out after {timeout}s."}

        if "error" in resp:
            err = resp["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return {"error": f"LSP server error: {msg}"}

        return {"result": resp.get("result")}

    except Exception as exc:
        return {"error": f"LSP subprocess failed: {exc}"}

    finally:
        # A PIPE handle left open leaks an fd and emits a ResourceWarning when
        # the Popen is collected, so stdout is closed alongside stdin on the
        # normal path. Any failure here (broken pipe, already-reaped child)
        # still falls back to kill — cleanup must never mask the response.
        if proc is not None:
            try:
                proc.stdin.close()  # type: ignore[union-attr]
                proc.stdout.close()  # type: ignore[union-attr]
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


def _confine_to_project(path: str, project_dir: str) -> str | None:
    """Resolve ``path`` against the LSP root and refuse anything outside it.

    An absolute path used to be accepted verbatim, so these tools handed the
    language server any readable file on disk with a matching extension —
    read scope is the project directory the LSP session is scoped to, not the
    whole drive.

    The check is **fail-closed on the real target**, not on the spelling: a
    junction inside the project that points outside it is an escape, and a
    lexical comparison cannot see it. The shared kernel
    (:func:`ai_assistant.path_policy.resolve_within`) resolves both sides —
    following links and putting Windows' 8.3 short form and the canonical
    long form on the same terms — and refuses anything that does not really
    stay inside the root. ``commonpath`` (not a string prefix test) also
    keeps ``C:\\project2`` from passing as ``C:\\project``; a different
    Windows drive is an escape too.
    """
    return resolve_within(project_dir, path)


def tool_go_to_definition(params: dict[str, Any]) -> dict[str, Any]:
    """Find the definition location of the symbol at *file*:*line*:*character*."""
    path = params.get("path", "")
    line = int(params.get("line", 1))
    character = int(params.get("character", 0))

    if not path:
        return {"error": "Parameter 'path' is required."}

    project_dir = _get_project_dir()
    abs_path = _confine_to_project(path, project_dir)
    if abs_path is None:
        return {"error": f"Path is outside the LSP project directory: {path}"}
    ext = os.path.splitext(abs_path)[1].lower()
    clangd_bin, pylsp_bin, lsp_timeout = lsp_binaries()

    if ext in _CPP_EXTS:
        server_bin, server_args = clangd_bin, ["--log=error"]
    elif ext in _PY_EXTS:
        server_bin, server_args = pylsp_bin, []
    else:
        return {"error": f"Unsupported file extension for LSP: {ext}."}

    response = _lsp_call(server_bin, server_args, abs_path, line, character,
                         "textDocument/definition", timeout=lsp_timeout)

    if response.get("error"):
        return response

    lsp_result = response.get("result")
    locations = _format_locations(lsp_result, "definition")

    if not locations:
        return {"path": path, "line": line, "character": character, "found": False, "message": "No definition found."}
    return {"path": path, "line": line, "character": character, "found": True, "definitions": locations}


def tool_find_references(params: dict[str, Any]) -> dict[str, Any]:
    """Find all references to the symbol at *file*:*line*:*character*."""
    path = params.get("path", "")
    line = int(params.get("line", 1))
    character = int(params.get("character", 0))

    if not path:
        return {"error": "Parameter 'path' is required."}

    project_dir = _get_project_dir()
    abs_path = _confine_to_project(path, project_dir)
    if abs_path is None:
        return {"error": f"Path is outside the LSP project directory: {path}"}
    ext = os.path.splitext(abs_path)[1].lower()
    clangd_bin, pylsp_bin, lsp_timeout = lsp_binaries()

    if ext in _CPP_EXTS:
        server_bin, server_args = clangd_bin, ["--log=error"]
    elif ext in _PY_EXTS:
        server_bin, server_args = pylsp_bin, []
    else:
        return {"error": f"Unsupported file extension for LSP: {ext}."}

    response = _lsp_call(server_bin, server_args, abs_path, line, character,
                         "textDocument/references", timeout=lsp_timeout)

    if response.get("error"):
        return response

    lsp_result = response.get("result")
    locations = _format_locations(lsp_result, "reference")

    if not locations:
        return {"path": path, "line": line, "character": character, "found": False, "message": "No references found."}
    return {"path": path, "line": line, "character": character, "found": True, "count": len(locations), "references": locations}


__all__ = [
    "tool_find_references",
    "tool_go_to_definition",
]
