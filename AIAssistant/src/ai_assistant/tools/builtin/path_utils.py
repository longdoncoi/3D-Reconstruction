"""Path safety utilities shared by builtin tools."""
from __future__ import annotations

import os

from ai_assistant.config.paths import get_paths

_AGENT_BLOCKED_DIRS = {".git", "build", "__pycache__", ".vs", "node_modules"}
_AGENT_BLOCKED_EXTS = {".exe", ".dll", ".so", ".bin", ".dat", ".pkl", ".gguf", ".onnx", ".pt", ".secret", ".key"}
_AGENT_MAX_FILE_READ_CHARS = 24000
_AGENT_MAX_UNSCOPED_SOURCE_LINES = 240

#: File names that carry secrets and must never be reachable by an agent tool.
#:
#: The list is matched on the *name*, not on the extension, and that is the
#: whole point: ``os.path.splitext(".env")`` is ``(".env", "")`` — a dotfile
#: has no extension, so an extension-based block-list never fires for exactly
#: the files it was written for. ``AIAssistant/.env`` is the file the server
#: loads its own secrets from (the at-rest AES-256-GCM key and every bearer
#: token), and ``read_file`` returned it in clear, without approval, because
#: the guard was looking at an extension that does not exist.
_AGENT_BLOCKED_NAMES = frozenset({
    ".env", ".envrc", ".npmrc", ".pypirc", ".netrc", ".gitconfig",
    "secrets.json", "credentials.json", "id_rsa", "id_ed25519", "id_ecdsa",
})


def is_secret_name(name: str) -> bool:
    """Whether a file *name* is one that carries a secret.

    Two rules, both name-based: the explicit set above, and the ``.env``
    family — ``.env``, ``.env.local``, ``.env.production`` — which is what
    the dotfile case actually looks like in a deployment.
    """
    lowered = name.lower()
    if lowered in _AGENT_BLOCKED_NAMES:
        return True
    return lowered == ".env" or lowered.startswith(".env.")


def agent_safe_path(rel_path: str) -> str | None:
    """Validate and resolve a relative path within project_root. Returns None if unsafe."""
    if not rel_path:
        return None
    project_root = str(get_paths().project_root)
    rel_path = rel_path.replace("\\", "/").strip("/")
    if ".." in rel_path.split("/"):
        return None
    abs_path = os.path.normpath(os.path.join(project_root, rel_path))
    real_root = os.path.realpath(project_root)
    # ``normpath`` alone is not containment: a symlink/junction inside the
    # project can point anywhere. Resolve the real target and check it too;
    # writes are separately checked by ``write_sandboxed_file``, but reads
    # must not follow a link out of the sandbox either.
    real_path = os.path.realpath(abs_path)
    try:
        if os.path.commonpath([abs_path, os.path.normpath(project_root)]) != os.path.normpath(project_root):
            return None
        if os.path.commonpath([real_path, real_root]) != real_root:
            return None
    except ValueError:
        return None
    # The secret check sits here, in the one function every read tool
    # resolves its path through, so it cannot be forgotten by a tool that
    # was written later. Both spellings are checked: the lexical one the
    # caller asked for, and the real one a symlink points at — a link named
    # ``notes.txt`` that points at ``.env`` must not be a way in either.
    if is_secret_name(os.path.basename(abs_path)) or is_secret_name(os.path.basename(real_path)):
        return None
    parts = rel_path.split("/")
    for part in parts:
        if part in _AGENT_BLOCKED_DIRS:
            return None
    return abs_path
