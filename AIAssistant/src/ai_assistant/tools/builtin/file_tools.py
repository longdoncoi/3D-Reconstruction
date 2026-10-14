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
