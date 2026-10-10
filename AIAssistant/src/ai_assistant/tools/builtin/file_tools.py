"""File reading and modification tools."""
from __future__ import annotations

import ast
import os
import re
from typing import Any

from ai_assistant.config.paths import get_paths
from ai_assistant.tools.sandbox import create_directory
from ai_assistant.tools.sandbox import write_file as write_sandboxed_file

from .path_utils import (
    _AGENT_BLOCKED_EXTS,
    _AGENT_MAX_FILE_READ_CHARS,
    _AGENT_MAX_UNSCOPED_SOURCE_LINES,
    agent_safe_path,
)

_agent_safe_path = agent_safe_path


def tool_read_file(params: dict[str, Any]) -> dict[str, Any]:
    """Read file content with optional line range."""
    path = params.get("path", "")
    abs_path = _agent_safe_path(path)
    if abs_path is None:
        return {"error": f"Đường dẫn không hợp lệ hoặc bị chặn: {path}"}
    if not os.path.isfile(abs_path):
        return {"error": f"File không tồn tại: {path}"}
    ext = os.path.splitext(abs_path)[1].lower()
    if ext in _AGENT_BLOCKED_EXTS:
        return {"error": f"Không thể đọc file binary: {path}"}

    try:
        for enc in ("utf-8", "utf-16", "cp1252", "latin-1"):
            try:
                with open(abs_path, "r", encoding=enc) as f:
                    lines = f.readlines()
                break
            except (UnicodeDecodeError, ValueError):
                continue
        else:
            return {"error": f"Không đọc được encoding của file: {path}"}

        total_lines = len(lines)
        requested_start = params.get("start_line")
        requested_end = params.get("end_line")
        symbol = str(params.get("symbol", "")).strip()
        if symbol and requested_start is None and requested_end is None:
            symbol_range = (
                _find_python_symbol_range(lines, symbol)
                if ext == ".py" else _find_symbol_range(lines, symbol)
            )
            if symbol_range is None:
                return {"error": f"Không tìm thấy symbol '{symbol}' trong file: {path}",
                        "path": path, "total_lines": total_lines}
            requested_start, requested_end = symbol_range

        if (requested_start is None and requested_end is None
                and ext in {".cpp", ".h", ".c", ".hpp", ".cc", ".cxx"}
                and total_lines > _AGENT_MAX_UNSCOPED_SOURCE_LINES):
            return {
                "error": (
                    f"File nguồn có {total_lines} dòng; cần đọc theo phạm vi. "
                    "Hãy dùng search_text/analyze_code để tìm symbol rồi gọi "
                    "read_file với symbol hoặc start_line/end_line."
                ),
                "path": path,
                "total_lines": total_lines,
            }

        start = max(1, requested_start or 1) - 1  # 0-indexed
        end = min(total_lines, requested_end or total_lines)
        if ext in {".cpp", ".h", ".c", ".hpp", ".cc", ".cxx"}:
            end = _expand_function_end(lines, start, end)

        selected = lines[start:end]
        content = "".join(selected)

        if len(content) > _AGENT_MAX_FILE_READ_CHARS:
            content = content[:_AGENT_MAX_FILE_READ_CHARS] + f"\n... [truncated at {_AGENT_MAX_FILE_READ_CHARS} chars]"

        return {
            "path": path,
            "total_lines": total_lines,
            "showing": f"lines {start+1}-{end}",
            "content": content,
        }
    except Exception as e:
        return {"error": f"Lỗi đọc file {path}: {e}"}


def _find_symbol_range(lines: list[str], symbol: str) -> tuple[int, int] | None:
    """Find a bounded source block for a named C/C++ function/method."""
    short_name = symbol.rsplit("::", 1)[-1].strip()
    candidates = []
    fallback = []
    qualified = symbol.strip()
    for i, line in enumerate(lines):
        if not short_name or not re.search(rf"\b{re.escape(short_name)}\s*\(", line):
            continue
        if qualified and qualified in line:
            fallback.append(i)
        lookahead = " ".join(lines[i:min(len(lines), i + 4)])
        if "{" in line or (qualified and qualified in line and "{" in lookahead):
            candidates.append(i)
    candidates = candidates or fallback
    if not candidates:
        return None
    start = candidates[0]
    depth = 0
    opened = False
    end = min(len(lines), start + 80)
    for idx in range(start, min(len(lines), start + 240)):
        depth += lines[idx].count("{") - lines[idx].count("}")
        if "{" in lines[idx]:
            opened = True
        if opened and depth <= 0:
            end = idx + 1
            break
    return start + 1, end


def _find_python_symbol_range(lines: list[str], symbol: str) -> tuple[int, int] | None:
    """Find the exact Python function/method range using the AST."""
    source = "".join(lines)
    short_name = symbol.rsplit("::", 1)[-1].rsplit(".", 1)[-1].strip()
    qualified = symbol.replace("::", ".").strip()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None

    if tree is not None:
        matches = []

        def visit(node: ast.AST, parents: tuple[str, ...] = ()) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node_path = ".".join((*parents, node.name))
                if node_path == qualified or node.name == short_name:
                    decorator_lines = [decorator.lineno for decorator in node.decorator_list]
                    start = min([node.lineno, *decorator_lines])
                    end = getattr(node, "end_lineno", node.lineno)
                    matches.append((start, end, node_path == qualified))
                next_parents = (*parents, node.name)
            elif isinstance(node, ast.ClassDef):
                next_parents = (*parents, node.name)
            else:
                next_parents = parents
            for child in ast.iter_child_nodes(node):
                visit(child, next_parents)

        visit(tree)
        if matches:
            matches.sort(key=lambda item: (not item[2], item[0]))
            return matches[0][0], matches[0][1]

    definition = re.compile(rf"^\s*(?:async\s+)?def\s+{re.escape(short_name)}\s*\(")
    for index, line in enumerate(lines):
        if not definition.search(line):
            continue
        indent = len(line) - len(line.lstrip())
        end = len(lines)
        for candidate in range(index + 1, len(lines)):
            next_line = lines[candidate]
            if next_line.strip() and len(next_line) - len(next_line.lstrip()) <= indent:
                end = candidate
                break
        return index + 1, end
    return None


def _expand_function_end(lines: list[str], start: int, end: int) -> int:
    """Extend a range when it begins at a C/C++ function definition."""
    header = " ".join(lines[start:min(len(lines), start + 8)])
    if not re.search(r"\b[\w:~]+\s*\([^;{}]*\)\s*(?:const\s*)?(?:override\s*)?\{", header):
        return end
    depth = 0
    opened = False
    for idx in range(start, min(len(lines), start + 240)):
        depth += lines[idx].count("{") - lines[idx].count("}")
        if "{" in lines[idx]:
            opened = True
        if opened and depth <= 0:
            return max(end, idx + 1)
    return end


def tool_write_file(params: dict[str, Any]) -> dict[str, Any]:
    """Execute write_file after user approval."""
    path = params.get("path", "")
    content = params.get("content", "")
    abs_path = _agent_safe_path(path)
    if abs_path is None:
        return {"error": f"Đường dẫn không hợp lệ: {path}"}
    return write_sandboxed_file(abs_path, content, str(get_paths().project_root))


def tool_patch_file(params: dict[str, Any]) -> dict[str, Any]:
    path = params.get("path", "")
    find_text = params.get("find", "")
    replacement = params.get("replace", "")
    abs_path = _agent_safe_path(path)
    if abs_path is None or not os.path.isfile(abs_path):
        return {"error": f"Invalid or missing file: {path}"}
    if not find_text:
        return {"error": "Find text must not be empty."}
    try:
        with open(abs_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        matches = content.count(find_text)
        if matches != 1:
            return {"error": f"Patch requires exactly one matching fragment; found {matches}.", "path": path}
        updated = content.replace(find_text, replacement, 1)
        result = write_sandboxed_file(abs_path, updated, str(get_paths().project_root))
        if result.get("error"):
            return result
        return {"success": True, "path": path, "replacements": 1,
                "bytes_written": result.get("bytes_written", 0), "sandbox": result.get("sandbox")}
    except OSError as error:
        return {"error": f"Unable to patch file: {error}"}


def tool_replace_file_content(params: dict[str, Any]) -> dict[str, Any]:
    path = params.get("path", "")
    target = params.get("targetContent", "")
    replacement = params.get("replacementContent", "")
    abs_path = _agent_safe_path(path)
    if abs_path is None or not os.path.isfile(abs_path):
        return {"error": f"Invalid or missing file: {path}"}
    if not target:
        return {"error": "Target content must not be empty."}
    try:
        with open(abs_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        matches = content.count(target)
        if matches != 1:
            return {"error": f"Replace requires exactly one matching fragment; found {matches}.", "path": path}
        updated = content.replace(target, replacement, 1)
        result = write_sandboxed_file(abs_path, updated, str(get_paths().project_root))
        if result.get("error"):
            return result
        return {"success": True, "path": path, "replacements": 1,
                "bytes_written": result.get("bytes_written", 0), "sandbox": result.get("sandbox")}
    except OSError as error:
        return {"error": f"Unable to replace file content: {error}"}


def tool_multi_replace_file_content(params: dict[str, Any]) -> dict[str, Any]:
    path = params.get("path", "")
    replacements = params.get("replacements", [])
    abs_path = _agent_safe_path(path)
    if abs_path is None or not os.path.isfile(abs_path):
        return {"error": f"Invalid or missing file: {path}"}
    if not replacements:
        return {"error": "Replacements list must not be empty."}
    try:
        with open(abs_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        
        for idx, rep in enumerate(replacements):
            target = rep.get("targetContent", "")
            replacement = rep.get("replacementContent", "")
            if not target:
                return {"error": f"Target content at index {idx} must not be empty."}
            matches = content.count(target)
            if matches != 1:
                return {"error": f"Replace at index {idx} requires exactly one matching fragment; found {matches}.", "path": path}
            content = content.replace(target, replacement, 1)

        result = write_sandboxed_file(abs_path, content, str(get_paths().project_root))
        if result.get("error"):
            return result
        return {"success": True, "path": path, "replacements": len(replacements),
                "bytes_written": result.get("bytes_written", 0), "sandbox": result.get("sandbox")}
    except OSError as error:
        return {"error": f"Unable to multi-replace file content: {error}"}


def tool_create_directory(params: dict[str, Any]) -> dict[str, Any]:
    path = params.get("path", "")
    abs_path = _agent_safe_path(path)
    if abs_path is None:
        return {"error": f"Invalid directory path: {path}"}
    paths = get_paths()
    return create_directory(abs_path, str(paths.project_root))


import time

_CODE_CITATION_PATTERN = re.compile(
    r"(?:trích\s*dẫn|trich\s*dan|cite|quote|show\s+code|source\s+code)"
    r"[\s\S]{0,80}?([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)\s*\(",
    re.IGNORECASE,
)
_PYTHON_DEF_PATTERN = re.compile(
    r"(?:async\s+)?def\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(",
    re.IGNORECASE,
)
_PYTHON_SYMBOL_LABEL_PATTERN = re.compile(
    r"(?:h.m|ham|function|method|symbol)\s*[:：]?\s*"
    r"(?:async\s+)?(?:def\s+)?([A-Za-z_]\w*(?:(?:::|\.)[A-Za-z_]\w*)*)",
    re.IGNORECASE,
)
_PYTHON_PATH_PATTERN = re.compile(
    r"(?P<path>[A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)*\.py)\b",
    re.IGNORECASE,
)
_EXPLICIT_CITATION_PATTERN = re.compile(
    r"(?:tr.{0,3}ch\s*d.{0,3}n|cite|quote|show\s+code|source\s+code)",
    re.IGNORECASE,
)

def _requested_code_symbol(task: str) -> str | None:
    match = _CODE_CITATION_PATTERN.search(task)
    if match:
        return match.group(1)
    match = _PYTHON_DEF_PATTERN.search(task)
    if match:
        return match.group(1)
    match = _PYTHON_SYMBOL_LABEL_PATTERN.search(task)
    return match.group(1) if match else None

def _citation_is_python(task: str, symbol: str) -> bool:
    return bool(
        _PYTHON_PATH_PATTERN.search(task)
        or re.search(r"\b(?:python|def|async\s+def)\b", task, re.IGNORECASE)
        or "." in symbol
    )

def _citation_path_hint(task: str) -> str | None:
    match = _PYTHON_PATH_PATTERN.search(task)
    if not match:
        return None
    return match.group("path").replace("\\", "/")

def _code_language(path: str) -> str:
    return {
        ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".h": "cpp", ".hpp": "cpp",
        ".c": "c", ".py": "python", ".js": "javascript", ".ts": "typescript",
    }.get(os.path.splitext(path)[1].lower(), "text")

def _build_code_citation_result(task: str, session_id: str, request_started: float) -> dict | None:
    symbol = _requested_code_symbol(task)
    if not symbol or not _EXPLICIT_CITATION_PATTERN.search(task):
        return None
    is_python = _citation_is_python(task, symbol)
    path_hint = _citation_path_hint(task) if is_python else None
    
    search_symbol = symbol.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
    if path_hint:
        search_path = os.path.dirname(path_hint) or "."
        search_pattern = os.path.basename(path_hint)
    elif is_python:
        search_path = "AIAssistant"
        search_pattern = "*.py"
    else:
        search_path = "src"
        search_pattern = "*.cpp"

    from .search_tools import tool_search_text
    search_params = {
        "query": search_symbol if is_python else symbol,
        "path": search_path,
        "file_pattern": search_pattern,
        "max_results": 50 if is_python else 20,
    }
    search_result = tool_search_text(search_params)
    matches = search_result.get("results", []) if isinstance(search_result, dict) else []
    if path_hint:
        matches = [
            item for item in matches
            if str(item.get("file", "")).replace("\\", "/").endswith(path_hint)
        ]
    if is_python:
        definition = next(
            (
                item for item in matches
                if re.search(
                    rf"\b(?:async\s+)?def\s+{re.escape(search_symbol)}\s*\(",
                    str(item.get("content", "")),
                )
            ),
            None,
        )
    else:
        definition = next((item for item in matches if "{" in str(item.get("content", ""))), None)
    if definition is None and not is_python:
        search_params = {"query": symbol, "path": "src", "file_pattern": "*.h", "max_results": 20}
        search_result = tool_search_text(search_params)
        matches = search_result.get("results", []) if isinstance(search_result, dict) else []
        definition = next((item for item in matches if "{" in str(item.get("content", ""))), None)
    if definition is None:
        return None

    path = str(definition["file"])
    read_params = {"path": path, "symbol": search_symbol if is_python else symbol}
    read_result = tool_read_file(read_params)
    if read_result.get("error"):
        return None
    content = str(read_result.get("content", "")).rstrip()
    if not content:
        return None

    search_verification = {"passed": True, "reason": "Found a source definition for the requested symbol."}
    read_verification = {"passed": True, "reason": "Read the complete source range for the requested symbol."}
    showing = str(read_result.get("showing", ""))
    final_answer = (
        f"### Trích dẫn mã nguồn\n\n"
        f"`{path}` — {showing}\n\n"
        f"```{_code_language(path)}\n{content}\n```"
    )
    display_read_result = dict(read_result)
    display_read_result["presentation"] = "citation_source"
    steps = [
        {"type": "delegation", "agent": "code", "tool": "search_text", "iteration": 0},
        {"type": "tool_call", "tool": "search_text", "params": search_params, "iteration": 0},
        {"type": "tool_result", "tool": "search_text", "result": search_result, "iteration": 0},
        {"type": "verification", "tool": "search_text", "result": search_verification, "iteration": 0},
        {"type": "delegation", "agent": "code", "tool": "read_file", "iteration": 0},
        {"type": "tool_call", "tool": "read_file", "params": read_params, "iteration": 0},
        {"type": "tool_result", "tool": "read_file", "result": display_read_result, "iteration": 0},
        {"type": "verification", "tool": "read_file", "result": read_verification, "iteration": 0},
        {"type": "final_answer", "content": final_answer, "iteration": 0},
    ]
    return {
        "status": "completed", "session_id": session_id, "steps": steps,
        "prior_step_count": 0, "iterations": 0,
        "total_ms": round((time.monotonic() - request_started) * 1000),
    }
