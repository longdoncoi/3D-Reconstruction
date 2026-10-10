"""Search and discovery tools."""
from __future__ import annotations

import fnmatch
import os
import re
from typing import Any

from ai_assistant.config.paths import get_paths

from .path_utils import _AGENT_BLOCKED_DIRS, agent_safe_path

_agent_safe_path = agent_safe_path


def tool_list_directory(params: dict[str, Any]) -> dict[str, Any]:
    """List directory contents."""
    path = params.get("path", ".")
    project_root = str(get_paths().project_root)
    abs_path: str | None = project_root if path == "." else _agent_safe_path(path)
    if abs_path is None:
        return {"error": f"Đường dẫn không hợp lệ: {path}"}
    if not os.path.isdir(abs_path):
        return {"error": f"Thư mục không tồn tại: {path}"}

    recursive = params.get("recursive", False)
    max_depth = params.get("max_depth", 3)
    entries: list[dict[str, Any]] = []
    count = 0
    max_entries = 150

    try:
        if recursive:
            for root, dirs, files in os.walk(abs_path):
                dirs[:] = sorted(d for d in dirs if d not in _AGENT_BLOCKED_DIRS)
                depth = root.replace(abs_path, "").count(os.sep)
                if depth >= max_depth:
                    dirs.clear()
                    continue
                rel_root = os.path.relpath(root, project_root)
                for d in sorted(dirs):
                    if count >= max_entries:
                        break
                    entries.append({"name": os.path.join(rel_root, d), "type": "directory"})
                    count += 1
                for f in sorted(files):
                    if count >= max_entries:
                        break
                    fp = os.path.join(root, f)
                    try:
                        size = os.path.getsize(fp)
                    except OSError:
                        size = 0
                    entries.append({
                        "name": os.path.join(rel_root, f),
                        "type": "file",
                        "size_bytes": size,
                    })
                    count += 1
                if count >= max_entries:
                    break
        else:
            for item in sorted(os.listdir(abs_path)):
                if item in _AGENT_BLOCKED_DIRS:
                    continue
                if count >= max_entries:
                    break
                fp = os.path.join(abs_path, item)
                rel = os.path.relpath(fp, project_root)
                if os.path.isdir(fp):
                    entries.append({"name": rel, "type": "directory"})
                else:
                    try:
                        size = os.path.getsize(fp)
                    except OSError:
                        size = 0
                    entries.append({"name": rel, "type": "file", "size_bytes": size})
                count += 1

        return {"path": path, "count": len(entries), "entries": entries}
    except Exception as e:
        return {"error": f"Lỗi liệt kê thư mục {path}: {e}"}


def tool_find_files(params: dict[str, Any]) -> dict[str, Any]:
    """Find files by glob without loading file contents into model context."""
    pattern = str(params.get("pattern", "")).strip()
    if not pattern or pattern in {".", ".."}:
        return {"error": "File pattern must not be empty."}
    path = params.get("path", ".")
    project_root = str(get_paths().project_root)
    abs_root: str | None = project_root if path == "." else _agent_safe_path(path)
    if abs_root is None or not os.path.isdir(abs_root):
        return {"error": f"Invalid or missing directory: {path}"}
    max_results = min(max(int(params.get("max_results", 100)), 1), 500)
    matches = []
    try:
        for root, dirs, files in os.walk(abs_root):
            dirs[:] = sorted(d for d in dirs if d not in _AGENT_BLOCKED_DIRS)
            for filename in sorted(files):
                relative_to_root = os.path.relpath(os.path.join(root, filename), abs_root)
                if not fnmatch.fnmatch(filename, pattern) and not fnmatch.fnmatch(relative_to_root, pattern):
                    continue
                matches.append(os.path.relpath(os.path.join(root, filename), project_root))
                if len(matches) >= max_results:
                    return {"pattern": pattern, "path": path, "count": len(matches),
                            "truncated": True, "matches": matches}
        return {"pattern": pattern, "path": path, "count": len(matches),
                "truncated": False, "matches": matches}
    except OSError as error:
        return {"error": f"Unable to find files: {error}"}


def tool_search_text(params: dict[str, Any]) -> dict[str, Any]:
    """Search for text in project files."""
    query = params.get("query", "")
    if not query:
        return {"error": "Query rỗng"}

    search_path = params.get("path", ".")
    project_root = str(get_paths().project_root)
    abs_search: str | None = (
        project_root if search_path == "." else _agent_safe_path(search_path)
    )
    if abs_search is None:
        return {"error": f"Đường dẫn không hợp lệ: {search_path}"}
    if not os.path.isdir(abs_search) and not os.path.isfile(abs_search):
        return {"error": f"Đường dẫn không tồn tại: {search_path}"}

    file_pattern = params.get("file_pattern", "*")
    file_patterns = [pattern.strip() for pattern in re.split(r"[;,]", str(file_pattern)) if pattern.strip()]
    if not file_patterns:
        file_patterns = ["*"]
    file_patterns = [f"*{pattern}" if pattern.startswith(".") else pattern for pattern in file_patterns]
    case_sensitive = params.get("case_sensitive", True)
    max_results = min(params.get("max_results", 50), 100)

    results = []
    search_query = query if case_sensitive else query.lower()
    text_exts = {".cpp", ".h", ".py", ".md", ".txt", ".cmake", ".json", ".xml", ".html", ".css", ".js", ".ts", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".bat", ".sh"}

    try:
        walk_roots = (
            [(os.path.dirname(abs_search), [], [os.path.basename(abs_search)])]
            if os.path.isfile(abs_search) else os.walk(abs_search)
        )
        search_root = os.path.dirname(abs_search) if os.path.isfile(abs_search) else abs_search
        for root, dirs, files in walk_roots:
            dirs[:] = sorted(d for d in dirs if d not in _AGENT_BLOCKED_DIRS)
            for filename in sorted(files):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in text_exts:
                    continue
                relative_to_search = os.path.relpath(os.path.join(root, filename), search_root)
                relative_to_project = os.path.relpath(os.path.join(root, filename), project_root)
                if file_patterns != ["*"] and not any(
                    fnmatch.fnmatch(filename, pattern)
                    or fnmatch.fnmatch(relative_to_search, pattern)
                    or fnmatch.fnmatch(relative_to_project, pattern)
                    for pattern in file_patterns
                ):
                    continue

                fp = os.path.join(root, filename)
                rel = os.path.relpath(fp, project_root)
                try:
                    with open(fp, "r", encoding="utf-8", errors="replace") as f:
                        for line_no, line in enumerate(f, 1):
                            check_line = line if case_sensitive else line.lower()
                            if search_query in check_line:
                                results.append({
                                    "file": rel,
                                    "line": line_no,
                                    "content": line.rstrip()[:200],
                                })
                                if len(results) >= max_results:
                                    return {"query": query, "count": len(results), "truncated": True, "results": results}
                except (OSError, UnicodeDecodeError):
                    continue

        return {"query": query, "count": len(results), "truncated": False, "results": results}
    except Exception as e:
        return {"error": f"Lỗi tìm kiếm: {e}"}
