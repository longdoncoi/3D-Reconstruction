"""Version control and project status tools."""
from __future__ import annotations

import os
import subprocess
from typing import Any

from ai_assistant.config.paths import get_paths

from .path_utils import _AGENT_BLOCKED_DIRS, agent_safe_path


def tool_get_project_status(params: dict[str, Any]) -> dict[str, Any]:
    """Return a lightweight, read-only project status."""
    project_root = str(get_paths().project_root)
    source_counts: dict[str, int] = {}
    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in _AGENT_BLOCKED_DIRS]
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in {".cpp", ".h", ".py", ".json", ".cmake"}:
                source_counts[ext] = source_counts.get(ext, 0) + 1
    try:
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=project_root,
                                capture_output=True, text=True, timeout=5, check=False).stdout.strip()
        changed = subprocess.run(["git", "status", "--short"], cwd=project_root,
                                 capture_output=True, text=True, timeout=5, check=False).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        branch, changed = "", []
    return {"project_root": project_root, "git_branch": branch, "changed_files": changed[:100],
            "source_file_counts": source_counts}


def tool_git_diff(params: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded Git diff for Code Agent review."""
    path = params.get("path", ".")
    project_root = str(get_paths().project_root)
    if path == ".":
        diff_path = "."
    else:
        abs_path = agent_safe_path(path)
        if abs_path is None or not os.path.exists(abs_path):
            return {"error": f"Invalid or missing diff path: {path}"}
        diff_path = os.path.relpath(abs_path, project_root)
    command = ["git", "diff", "--no-ext-diff"]
    if params.get("staged", False):
        command.append("--cached")
    command.extend(["--", diff_path])
    try:
        completed = subprocess.run(command, cwd=project_root, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=10, check=False)
        if completed.returncode != 0:
            return {"error": completed.stderr.strip() or "git diff failed", "return_code": completed.returncode}
        content = completed.stdout
        limit = 16000
        return {"path": path, "staged": bool(params.get("staged", False)),
                "content": content[:limit], "truncated": len(content) > limit}
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": f"Unable to read Git diff: {error}"}
