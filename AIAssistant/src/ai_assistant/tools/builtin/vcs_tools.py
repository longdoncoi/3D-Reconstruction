"""Version control and project status tools.

Read-only git invocations go through the sandbox policy gate
(:mod:`ai_assistant.tools.sandbox`): the executable is resolved via ``PATH``
(no workspace hijack) and the subcommand surface is allow-listed, so these
tools cannot be repurposed into an execution hatch outside the sandbox.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any

from ai_assistant.config.paths import get_paths
from ai_assistant.exec_policy import resolve_executable
from ai_assistant.tools.sandbox import (
    _executable_rejection,
    _subcommand_rejection,
)

from .path_utils import agent_safe_path, is_blocked_dir


def _run_git(args: list[str], cwd: str, timeout: int) -> subprocess.CompletedProcess[str]:
    """Run a validated read-only git command with the sandbox policy gate.

    Raises ``RuntimeError`` when the invocation violates the allow-list or the
    binary cannot be resolved via ``PATH``; callers translate ``OSError`` /
    ``subprocess.SubprocessError`` to their observable fallback as before.

    The binary is **fail-closed**: when ``git`` is not on ``PATH`` the call
    is refused, not continued with a bare name. A bare name hands the search
    back to the operating system, and on Windows the operating system
    searches the *current directory* first — so the fallback re-opened
    exactly the hijack the resolver exists to close, and it did so silently
    (the shipped container image has no ``git`` at all, so there the
    fallback was the *default* path).
    """
    executable_rejection = _executable_rejection("git")
    if executable_rejection:
        msg = executable_rejection
        raise RuntimeError(msg)
    probe = ["git", *args]
    sub_rejection = _subcommand_rejection(probe, cwd)
    if sub_rejection:
        msg = sub_rejection
        raise RuntimeError(msg)
    binary = resolve_executable("git", os.environ.get("PATH", ""))
    if binary is None:
        msg = "Executable 'git' was not found on PATH."
        raise RuntimeError(msg)
    return subprocess.run([binary, *args], cwd=cwd,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, check=False)


def tool_get_project_status(params: dict[str, Any]) -> dict[str, Any]:
    """Return a lightweight, read-only project status."""
    project_root = str(get_paths().project_root)
    source_counts: dict[str, int] = {}
    for _root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if not is_blocked_dir(d)]
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in {".cpp", ".h", ".py", ".json", ".cmake"}:
                source_counts[ext] = source_counts.get(ext, 0) + 1
    try:
        branch = _run_git(["branch", "--show-current"], cwd=project_root, timeout=5).stdout.strip()
        changed = _run_git(["status", "--short"], cwd=project_root, timeout=5).stdout.splitlines()
    except (OSError, subprocess.SubprocessError, RuntimeError):
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
    command = ["diff", "--no-ext-diff"]
    if params.get("staged", False):
        command.append("--cached")
    command.extend(["--", diff_path])
    try:
        completed = _run_git(command, cwd=project_root, timeout=10)
        if completed.returncode != 0:
            return {"error": completed.stderr.strip() or "git diff failed", "return_code": completed.returncode}
        content = completed.stdout
        limit = 16000
        return {"path": path, "staged": bool(params.get("staged", False)),
                "content": content[:limit], "truncated": len(content) > limit}
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        return {"error": f"Unable to read Git diff: {error}"}
