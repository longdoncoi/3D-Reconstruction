"""LSP (Language Server Protocol) client wrapper for Coding Agent tools.

Provides ``go_to_definition`` and ``find_references`` by spawning clangd
(for C++/Qt files) or pylsp (for Python files) as a subprocess and
communicating via stdio using the LSP JSON-RPC protocol.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from ai_assistant.config.paths import get_paths

_CLANGD_BINARY = os.getenv("AGENT_CLANGD_BIN", "clangd")
_PYLSP_BINARY = os.getenv("AGENT_PYLSP_BIN", "pylsp")
_LSP_TIMEOUT = int(os.getenv("AGENT_LSP_TIMEOUT", "15"))

_CPP_EXTS = {".cpp", ".c", ".h", ".hpp", ".cc", ".cxx"}
_PY_EXTS = {".py"}


def _get_project_dir() -> str:
    try:
        return str(get_paths().project_root)
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _make_request(method: str, params: dict, req_id: int = 1) -> bytes:
    body = json.dumps(
        {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params},
        ensure_ascii=False,
    ).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode()
    return header + body


def _make_notification(method: str, params: dict) -> bytes:
    body = json.dumps(
        {"jsonrpc": "2.0", "method": method, "params": params},
        ensure_ascii=False,
    ).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode()
    return header + body


def _read_response(proc: subprocess.Popen, timeout: float = 10.0) -> dict | None:
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
        return json.loads(raw_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


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


def _format_locations(raw_result: Any, label: str) -> list[dict]:
    if not raw_result:
        return []
    items: list[dict] = raw_result if isinstance(raw_result, list) else [raw_result]
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
    extra_params: dict | None = None,
) -> dict:
    resolved_bin = shutil.which(server_bin)
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
    proc: subprocess.Popen | None = None
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

        resp = _read_response(proc, timeout=float(_LSP_TIMEOUT))
        if resp is None:
            return {"error": f"LSP request '{method}' timed out after {_LSP_TIMEOUT}s."}

        if "error" in resp:
            err = resp["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return {"error": f"LSP server error: {msg}"}

        return {"result": resp.get("result")}

    except Exception as exc:  # noqa: BLE001
        return {"error": f"LSP subprocess failed: {exc}"}

    finally:
        if proc is not None:
            try:
                proc.stdin.close()  # type: ignore[union-attr]
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


def tool_go_to_definition(params: dict) -> dict:
    """Find the definition location of the symbol at *file*:*line*:*character*."""
    path = params.get("path", "")
    line = int(params.get("line", 1))
    character = int(params.get("character", 0))

    if not path:
        return {"error": "Parameter 'path' is required."}

    project_dir = _get_project_dir()
    abs_path = os.path.join(project_dir, path) if not os.path.isabs(path) else path
    ext = os.path.splitext(abs_path)[1].lower()

    if ext in _CPP_EXTS:
        server_bin, server_args = _CLANGD_BINARY, ["--log=error"]
    elif ext in _PY_EXTS:
        server_bin, server_args = _PYLSP_BINARY, []
    else:
        return {"error": f"Unsupported file extension for LSP: {ext}."}

    response = _lsp_call(server_bin, server_args, abs_path, line, character, "textDocument/definition")

    if response.get("error"):
        return response

    lsp_result = response.get("result")
    locations = _format_locations(lsp_result, "definition")

    if not locations:
        return {"path": path, "line": line, "character": character, "found": False, "message": "No definition found."}
    return {"path": path, "line": line, "character": character, "found": True, "definitions": locations}


def tool_find_references(params: dict) -> dict:
    """Find all references to the symbol at *file*:*line*:*character*."""
    path = params.get("path", "")
    line = int(params.get("line", 1))
    character = int(params.get("character", 0))

    if not path:
        return {"error": "Parameter 'path' is required."}

    project_dir = _get_project_dir()
    abs_path = os.path.join(project_dir, path) if not os.path.isabs(path) else path
    ext = os.path.splitext(abs_path)[1].lower()

    if ext in _CPP_EXTS:
        server_bin, server_args = _CLANGD_BINARY, ["--log=error"]
    elif ext in _PY_EXTS:
        server_bin, server_args = _PYLSP_BINARY, []
    else:
        return {"error": f"Unsupported file extension for LSP: {ext}."}

    response = _lsp_call(server_bin, server_args, abs_path, line, character, "textDocument/references")

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
