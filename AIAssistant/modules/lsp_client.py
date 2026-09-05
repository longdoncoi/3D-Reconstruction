"""LSP (Language Server Protocol) client wrapper for Coding Agent tools.

Provides ``go_to_definition`` and ``find_references`` by spawning clangd
(for C++/Qt files) or pylsp (for Python files) as a subprocess and
communicating via stdio using the LSP JSON-RPC protocol.

Design goals:
- No external Python package required beyond stdlib — JSON-RPC is implemented
  directly over subprocess stdio.
- Stateless per-request: a fresh server process is started for each call and
  terminated after the response to keep resource usage bounded.
- Safe: every call is bounded by a timeout, and the server PID is always
  cleaned up even when an error occurs.
- Failure-tolerant: if clangd / pylsp is not installed, the tools return a
  clear error instead of crashing the agent.

Usage from agent_module:
    from .lsp_client import tool_go_to_definition, tool_find_references
    MCP_LOCAL_EXECUTORS["go_to_definition"] = tool_go_to_definition
    MCP_LOCAL_EXECUTORS["find_references"] = tool_find_references
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .config import PROJECT_DIR, logger

# ── Constants ─────────────────────────────────────────────────────────────────

_CLANGD_BINARY = os.getenv("AGENT_CLANGD_BIN", "clangd")
_PYLSP_BINARY  = os.getenv("AGENT_PYLSP_BIN", "pylsp")
_LSP_TIMEOUT   = int(os.getenv("AGENT_LSP_TIMEOUT", "15"))   # seconds per call

_CPP_EXTS  = {".cpp", ".c", ".h", ".hpp", ".cc", ".cxx"}
_PY_EXTS   = {".py"}


# ── Internal JSON-RPC helpers ─────────────────────────────────────────────────

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


def _read_response(proc: subprocess.Popen, timeout: float) -> dict:
    """Read one LSP response from the server's stdout within *timeout* seconds."""
    deadline = time.monotonic() + timeout
    header_buf = b""
    # Read headers line by line
    while time.monotonic() < deadline:
        ch = proc.stdout.read(1)
        if not ch:
            raise EOFError("LSP server closed stdout")
        header_buf += ch
        if header_buf.endswith(b"\r\n\r\n"):
            break
    else:
        raise TimeoutError(f"LSP server did not respond within {timeout}s")

    content_length = 0
    for line in header_buf.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            content_length = int(line.split(b":", 1)[1].strip())
            break

    if content_length <= 0:
        raise ValueError(f"Invalid Content-Length in LSP header: {header_buf!r}")

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("LSP server timeout while reading body")

    body = b""
    while len(body) < content_length and time.monotonic() < deadline:
        chunk = proc.stdout.read(content_length - len(body))
        if not chunk:
            raise EOFError("LSP server closed stdout during body read")
        body += chunk

    return json.loads(body.decode("utf-8"))


def _uri_to_path(uri: str) -> str:
    """Convert a file URI to an absolute OS path."""
    if uri.startswith("file:///"):
        path = uri[len("file:///"):]
        if os.name == "nt":
            # Windows: file:///C:/foo  → C:/foo
            return path.replace("/", os.sep)
        return "/" + path
    if uri.startswith("file://"):
        return uri[len("file://"):]
    return uri


def _path_to_uri(path: str) -> str:
    """Convert an absolute OS path to a file URI."""
    abs_path = os.path.abspath(path)
    if os.name == "nt":
        return "file:///" + abs_path.replace(os.sep, "/")
    return "file://" + abs_path


# ── Core LSP call ─────────────────────────────────────────────────────────────

def _lsp_call(
    server_bin: str,
    server_args: list[str],
    file_path: str,
    line: int,
    character: int,
    method: str,
    timeout: int = _LSP_TIMEOUT,
) -> dict:
    """
    Launch an LSP server, initialise it, send one request, collect the response
    and shut down the server gracefully.

    Returns a dict with either ``result`` (the LSP response) or ``error``.
    """
    if not shutil.which(server_bin):
        return {"error": f"LSP server not found on PATH: {server_bin}. "
                         f"Install it (e.g. 'winget install LLVM' for clangd, "
                         f"'pip install python-lsp-server' for pylsp)."}

    abs_path = os.path.abspath(file_path)
    if not os.path.isfile(abs_path):
        return {"error": f"File not found: {file_path}"}

    workspace_uri = _path_to_uri(PROJECT_DIR)
    file_uri = _path_to_uri(abs_path)

    try:
        content = Path(abs_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"error": f"Cannot read file: {exc}"}

    proc = None
    try:
        proc = subprocess.Popen(
            [server_bin, *server_args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=PROJECT_DIR,
        )

        # ── 1. Initialize ──────────────────────────────────────────────────
        init_req = _make_request("initialize", {
            "processId": os.getpid(),
            "rootUri": workspace_uri,
            "capabilities": {
                "textDocument": {
                    "definition": {"dynamicRegistration": False},
                    "references": {"dynamicRegistration": False},
                }
            },
            "initializationOptions": {},
        }, req_id=1)
        proc.stdin.write(init_req)
        proc.stdin.flush()
        _read_response(proc, min(timeout, 10))  # initialize response

        # ── 2. Initialized notification ────────────────────────────────────
        proc.stdin.write(_make_notification("initialized", {}))
        proc.stdin.flush()

        # ── 3. textDocument/didOpen ────────────────────────────────────────
        ext = os.path.splitext(abs_path)[1].lower()
        lang = "cpp" if ext in _CPP_EXTS else "python"
        proc.stdin.write(_make_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": file_uri,
                "languageId": lang,
                "version": 1,
                "text": content,
            }
        }))
        proc.stdin.flush()
        # Give the server a moment to index before sending the query
        time.sleep(0.5)

        # ── 4. Send actual request ─────────────────────────────────────────
        lsp_request = _make_request(method, {
            "textDocument": {"uri": file_uri},
            "position": {"line": line - 1, "character": character},
            "context": {"includeDeclaration": True},  # for references only
        }, req_id=2)
        proc.stdin.write(lsp_request)
        proc.stdin.flush()

        remaining = timeout - 1  # account for the sleep above
        response = _read_response(proc, max(remaining, 5))

        # ── 5. Shutdown ────────────────────────────────────────────────────
        try:
            proc.stdin.write(_make_request("shutdown", {}, req_id=3))
            proc.stdin.flush()
            proc.stdin.write(_make_notification("exit", {}))
            proc.stdin.flush()
        except OSError:
            pass

        return response

    except TimeoutError:
        return {"error": f"LSP server timed out after {timeout}s"}
    except (OSError, EOFError, json.JSONDecodeError, ValueError) as exc:
        return {"error": f"LSP communication error: {exc}"}
    finally:
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass


# ── High-level result formatting ──────────────────────────────────────────────

def _format_locations(lsp_result: Any, label: str) -> list[dict]:
    """Convert LSP Location / LocationLink list to a compact agent-friendly list."""
    if lsp_result is None:
        return []
    if isinstance(lsp_result, dict):
        lsp_result = [lsp_result]
    if not isinstance(lsp_result, list):
        return []

    items = []
    for loc in lsp_result:
        # Handles both Location and LocationLink
        uri   = loc.get("uri") or loc.get("targetUri", "")
        rng   = loc.get("range") or loc.get("targetSelectionRange", {})
        start = rng.get("start", {})
        path  = _uri_to_path(uri)
        try:
            rel = os.path.relpath(path, PROJECT_DIR)
        except ValueError:
            rel = path
        items.append({
            "file":   rel.replace(os.sep, "/"),
            "line":   start.get("line", 0) + 1,
            "column": start.get("character", 0),
        })
    return items


# ── Public tool functions ─────────────────────────────────────────────────────

def tool_go_to_definition(params: dict) -> dict:
    """Jump to the definition of the symbol at *file*:*line*:*character*.

    Parameters
    ----------
    path : str
        Relative path to the file containing the symbol.
    line : int
        1-based line number of the symbol.
    character : int
        0-based character offset within the line.
    """
    path      = params.get("path", "")
    line      = int(params.get("line", 1))
    character = int(params.get("character", 0))

    if not path:
        return {"error": "Parameter 'path' is required."}

    abs_path = os.path.join(PROJECT_DIR, path) if not os.path.isabs(path) else path
    ext = os.path.splitext(abs_path)[1].lower()

    if ext in _CPP_EXTS:
        server_bin, server_args = _CLANGD_BINARY, ["--log=error"]
    elif ext in _PY_EXTS:
        server_bin, server_args = _PYLSP_BINARY, []
    else:
        return {"error": f"Unsupported file extension for LSP: {ext}. "
                         f"Supported: {_CPP_EXTS | _PY_EXTS}"}

    response = _lsp_call(server_bin, server_args, abs_path, line, character,
                         "textDocument/definition")

    if response.get("error"):
        return response

    lsp_result = response.get("result")
    locations  = _format_locations(lsp_result, "definition")

    if not locations:
        return {"path": path, "line": line, "character": character,
                "found": False, "message": "No definition found."}
    return {"path": path, "line": line, "character": character,
            "found": True, "definitions": locations}


def tool_find_references(params: dict) -> dict:
    """Find all references to the symbol at *file*:*line*:*character*.

    Parameters
    ----------
    path : str
        Relative path to the file containing the symbol.
    line : int
        1-based line number of the symbol.
    character : int
        0-based character offset within the line.
    """
    path      = params.get("path", "")
    line      = int(params.get("line", 1))
    character = int(params.get("character", 0))

    if not path:
        return {"error": "Parameter 'path' is required."}

    abs_path = os.path.join(PROJECT_DIR, path) if not os.path.isabs(path) else path
    ext = os.path.splitext(abs_path)[1].lower()

    if ext in _CPP_EXTS:
        server_bin, server_args = _CLANGD_BINARY, ["--log=error"]
    elif ext in _PY_EXTS:
        server_bin, server_args = _PYLSP_BINARY, []
    else:
        return {"error": f"Unsupported file extension for LSP: {ext}."}

    response = _lsp_call(server_bin, server_args, abs_path, line, character,
                         "textDocument/references")

    if response.get("error"):
        return response

    lsp_result = response.get("result")
    locations  = _format_locations(lsp_result, "reference")

    if not locations:
        return {"path": path, "line": line, "character": character,
                "found": False, "message": "No references found."}
    return {"path": path, "line": line, "character": character,
            "found": True, "count": len(locations), "references": locations}
