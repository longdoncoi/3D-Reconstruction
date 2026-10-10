"""Policy-first sandbox for destructive agent tools.

Docker isolation is opt-in because the Qt desktop distribution must still work
without Docker.  Even the local fallback uses an allow-list and blocks shell
metacharacters rather than executing arbitrary commands.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ai_assistant.settings import agent_write_roots, load_agent_runtime_settings

_ALLOWED = {"cmake", "ctest", "python", "pytest", "ruff", "git"}
_FORBIDDEN = {"&&", "||", ";", "|", ">", "<", "rm", "del", "format", "shutdown", "`", "$("}

# Interpreters whose command line can carry inline code. For these the only
# accepted trailing arguments are a script path or the benign version/help
# flags: ``-c``/``-m``/``-`` (and their attached forms, e.g. ``-cprint(1)``)
# turn the allow-list into arbitrary code execution.
_INTERPRETERS = frozenset({"python", "python.exe", "python3", "python3.exe"})
_SAFE_INTERPRETER_FLAGS = frozenset({"--version", "--help", "-V", "-h", "-q", "-B", "-s", "-E"})


def _inline_code_requested(argv: list[str]) -> bool:
    """Return True when an interpreter argument would execute inline code."""
    if Path(argv[0]).name.casefold() not in _INTERPRETERS:
        return False
    for argument in argv[1:]:
        if argument in {"-", "--"}:
            return True  # stdin marker / option terminator are never legitimate here
        if not argument.startswith("-"):
            continue  # script path
        if argument in _SAFE_INTERPRETER_FLAGS:
            continue
        if argument.startswith("--"):
            if argument.split("=", 1)[0] in _SAFE_INTERPRETER_FLAGS:
                continue
            return True
        return True  # short option, possibly with an attached value (-c, -cCODE, -m, -mmod)
    return False


def run(command: str, cwd: str, timeout: int) -> dict[str, Any]:
    runtime_settings = load_agent_runtime_settings()
    if not runtime_settings.sandbox_enabled:
        return {"error": "Agent command sandbox is disabled by policy."}
    if any(token in command.casefold() for token in _FORBIDDEN):
        return {"error": "Command rejected by sandbox policy."}
    try:
        argv = shlex.split(command, posix=os.name != "nt")
    except ValueError as error:
        return {"error": f"Invalid command: {error}"}
    if not argv or Path(argv[0]).name.casefold() not in _ALLOWED:
        return {"error": "Executable is not in the agent allow-list."}
    if _inline_code_requested(argv):
        return {"error": "Inline interpreter code (-c, -m, -) is rejected by sandbox policy."}
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONUTF8": "1"}
    if runtime_settings.sandbox_runtime == "docker":
        image = runtime_settings.sandbox_image
        argv = ["docker", "run", "--rm", "--network", "none", "--memory", "1g", "--cpus", "1.0",
                "--pids-limit", "128", "-v", f"{cwd}:/workspace:rw", "-w", "/workspace", image, *argv]
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              timeout=min(timeout, 120), env=env, shell=False,
                              encoding="utf-8", errors="replace", check=False)
        return {"command": argv, "return_code": proc.returncode,
                "stdout": proc.stdout[:5000], "stderr": proc.stderr[:2000],
                "sandbox": "docker" if runtime_settings.sandbox_runtime == "docker" else "local-allowlist"}
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s"}


def write_file(path: str, content: str, project_root: str) -> dict[str, Any]:
    """Atomically write only allow-listed project subtrees and bounded payloads."""
    runtime_settings = load_agent_runtime_settings()
    if not runtime_settings.sandbox_enabled:
        return {"error": "Agent write sandbox is disabled by policy."}
    if len(content.encode("utf-8")) > runtime_settings.max_write_bytes:
        return {"error": "Write exceeds AGENT_MAX_WRITE_BYTES."}
    try:
        relative = Path(path).resolve().relative_to(Path(project_root).resolve())
    except ValueError:
        return {"error": "Write target escapes the project root."}
    if not relative.parts or relative.parts[0] not in agent_write_roots():
        return {"error": "Write target is not in AGENT_WRITE_ALLOWLIST."}
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=target.parent,
                                         delete=False) as temporary:
            temporary.write(content)
            temporary_path = temporary.name
        os.replace(temporary_path, target)
        return {"success": True, "path": str(relative).replace("\\", "/"),
                "bytes_written": len(content.encode("utf-8")), "sandbox": "write-allowlist"}
    except OSError as error:
        return {"error": f"Unable to write sandboxed target: {error}"}


def create_directory(path: str, project_root: str) -> dict[str, Any]:
    """Create an allow-listed directory after explicit approval."""
    if not load_agent_runtime_settings().sandbox_enabled:
        return {"error": "Agent directory sandbox is disabled by policy."}
    try:
        root = Path(project_root).resolve()
        target = Path(path).resolve()
        relative = target.relative_to(root)
    except ValueError:
        return {"error": "Directory target escapes the project root."}
    if not relative.parts or relative.parts[0] not in agent_write_roots():
        return {"error": "Directory target is not in AGENT_WRITE_ALLOWLIST."}
    try:
        existed = target.is_dir()
        target.mkdir(parents=True, exist_ok=True)
        return {"success": True, "path": str(relative).replace("\\", "/"), "created": not existed,
                "sandbox": "directory-allowlist"}
    except OSError as error:
        return {"error": f"Unable to create sandboxed directory: {error}"}
