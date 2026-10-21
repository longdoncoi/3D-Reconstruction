"""Path safety utilities shared by builtin tools.

The rules themselves live in :mod:`ai_assistant.path_policy` — they are
kernel policy, and three different layers need them (the agent tools, the
LSP read gate, the HTTP attachment gate). This module keeps the *agent*
specific parts: the project-root confinement, the blocked directories, the
secret file names and the read budget.
"""
from __future__ import annotations

import os

from ai_assistant.config.paths import get_paths
from ai_assistant.path_policy import (
    BLOCKED_DIR_NAMES,
    is_blocked_dir,
    is_secret_name,
    resolve_within,
)

#: Kept under their historical names: the tool package re-exports them and
#: the walkers use them to prune a directory listing.
_AGENT_BLOCKED_DIRS = BLOCKED_DIR_NAMES
_AGENT_BLOCKED_EXTS = {".exe", ".dll", ".so", ".bin", ".dat", ".pkl", ".gguf", ".onnx", ".pt", ".secret", ".key"}
_AGENT_MAX_FILE_READ_CHARS = 24000
_AGENT_MAX_UNSCOPED_SOURCE_LINES = 240

__all__ = [
    "_AGENT_BLOCKED_DIRS",
    "_AGENT_BLOCKED_EXTS",
    "_AGENT_MAX_FILE_READ_CHARS",
    "_AGENT_MAX_UNSCOPED_SOURCE_LINES",
    "agent_safe_path",
    "is_blocked_dir",
    "is_secret_name",
]


def _blocked_dir_in(relative: str) -> bool:
    """Whether any segment of a project-relative path is a blocked directory.

    The comparison is case-insensitive (a Windows filesystem is
    case-insensitive, so ``.GIT`` is the same directory as ``.git``). The
    case-sensitive form of this rule is why ``read_file(".GIT/config")``
    returned the repository's git configuration — remote URL and credential
    helper included — while ``read_file(".git/config")`` was refused.
    """
    return any(is_blocked_dir(part) for part in relative.replace("\\", "/").split("/"))


def agent_safe_path(rel_path: str) -> str | None:
    """Validate and resolve a relative path within project_root. Returns None if unsafe."""
    if not rel_path:
        return None
    project_root = str(get_paths().project_root)
    rel_path = rel_path.replace("\\", "/").strip("/")
    if ".." in rel_path.split("/"):
        return None
    # The kernel owns the containment decision (both the spelling and the
    # real target) and hands back the caller's own spelling, made absolute —
    # so the path the tool opens is the path it was given.
    abs_path = resolve_within(project_root, rel_path)
    if abs_path is None:
        return None
    # The blocked-directory rule is applied to *both* the path the caller
    # asked for and the real one — a link named ``notes.txt`` that points
    # at ``.git/config`` must not be a way in.
    if _blocked_dir_in(rel_path):
        return None
    try:
        real_relative = os.path.relpath(
            os.path.realpath(abs_path), os.path.realpath(project_root),
        )
    except ValueError:
        return None
    if _blocked_dir_in(real_relative):
        return None
    # The secret check sits here, in the one function every read tool
    # resolves its path through, so it cannot be forgotten by a tool that
    # was written later. Both spellings are checked: the lexical one the
    # caller asked for, and the real one a symlink points at.
    real_basename = os.path.basename(os.path.realpath(abs_path))
    if is_secret_name(os.path.basename(rel_path)) or is_secret_name(real_basename):
        return None
    return abs_path
