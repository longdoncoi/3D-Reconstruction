"""System and sandbox execution tools."""
from __future__ import annotations

import json
import os
from typing import Any

from ai_assistant.config.paths import get_paths
from ai_assistant.tools.sandbox import run as run_sandboxed_command

from .file_tools import _agent_safe_path


def tool_validate_file(params: dict[str, Any]) -> dict[str, Any]:
    """Validate Python or JSON syntax without executing project code."""
    path = params.get("path", "")
    abs_path = _agent_safe_path(path)
    if abs_path is None or not os.path.isfile(abs_path):
        return {"error": f"Invalid or missing file: {path}"}
    ext = os.path.splitext(abs_path)[1].lower()
    try:
        with open(abs_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        if ext == ".py":
            compile(content, path, "exec")
        elif ext == ".json":
            json.loads(content)
        else:
            return {"error": "Only .py and .json files can be validated."}
        return {"success": True, "path": path, "validation": "syntax_valid"}
    except (SyntaxError, ValueError) as error:
        return {"success": False, "path": path, "error": str(error)}


def tool_run_command(params: dict[str, Any]) -> dict[str, Any]:
    """Execute shell command after user approval."""
    command = params.get("command", "")
    cwd = params.get("cwd", ".")
    timeout = min(params.get("timeout", 30), 120)  # Max 2 minutes

    project_root = str(get_paths().project_root)
    abs_cwd: str | None = project_root if cwd == "." else _agent_safe_path(cwd)
    if abs_cwd is None or not os.path.isdir(abs_cwd):
        return {"error": f"Invalid command working directory: {cwd}"}

    return run_sandboxed_command(command, abs_cwd, timeout)
