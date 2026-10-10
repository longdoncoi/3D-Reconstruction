"""Path safety utilities shared by builtin tools."""
from __future__ import annotations

import os

from ai_assistant.config.paths import get_paths

_AGENT_BLOCKED_DIRS = {".git", "build", "__pycache__", ".vs", "node_modules"}
_AGENT_BLOCKED_EXTS = {".exe", ".dll", ".so", ".bin", ".dat", ".pkl", ".gguf", ".onnx", ".pt"}
_AGENT_MAX_FILE_READ_CHARS = 24000
_AGENT_MAX_UNSCOPED_SOURCE_LINES = 240


def agent_safe_path(rel_path: str) -> str | None:
    """Validate and resolve a relative path within project_root. Returns None if unsafe."""
    if not rel_path:
        return None
    project_root = str(get_paths().project_root)
    rel_path = rel_path.replace("\\", "/").strip("/")
    if ".." in rel_path.split("/"):
        return None
    abs_path = os.path.normpath(os.path.join(project_root, rel_path))
    try:
        if os.path.commonpath([abs_path, os.path.normpath(project_root)]) != os.path.normpath(project_root):
            return None
    except ValueError:
        return None
    parts = rel_path.split("/")
    for part in parts:
        if part in _AGENT_BLOCKED_DIRS:
            return None
    return abs_path
