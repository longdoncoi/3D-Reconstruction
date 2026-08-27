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

_ALLOWED = {"cmake", "ctest", "python", "pytest", "ruff", "git"}
_FORBIDDEN = {"&&", "||", ";", "|", ">", "<", "rm", "del", "format", "shutdown"}
_WRITE_ROOTS = {part.strip() for part in os.getenv(
    "AGENT_WRITE_ALLOWLIST", "src,AIAssistant,AIComputerVision,Config,Docs,tests").split(",") if part.strip()}


def run(command: str, cwd: str, timeout: int) -> dict[str, Any]:
    if os.getenv("AGENT_SANDBOX_ENABLED", "1") != "1":
        return {"error": "Agent command sandbox is disabled by policy."}
    if any(token in command.casefold() for token in _FORBIDDEN):
        return {"error": "Command rejected by sandbox policy."}
    try:
        argv = shlex.split(command, posix=os.name != "nt")
    except ValueError as error:
        return {"error": f"Invalid command: {error}"}
    if not argv or Path(argv[0]).name.casefold() not in _ALLOWED:
        return {"error": "Executable is not in the agent allow-list."}
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONUTF8": "1"}
    runtime = os.getenv("AGENT_SANDBOX_RUNTIME", "local").casefold()
    if runtime == "docker":
        image = os.getenv("AGENT_SANDBOX_IMAGE", "python:3.11-slim")
        argv = ["docker", "run", "--rm", "--network", "none", "--memory", "1g", "--cpus", "1.0",
                "--pids-limit", "128", "-v", f"{cwd}:/workspace:rw", "-w", "/workspace", image, *argv]
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              timeout=min(timeout, 120), env=env, shell=False,
                              encoding="utf-8", errors="replace", check=False)
        return {"command": argv, "return_code": proc.returncode,
                "stdout": proc.stdout[:5000], "stderr": proc.stderr[:2000],
                "sandbox": "docker" if runtime == "docker" else "local-allowlist"}
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s"}


def write_file(path: str, content: str, project_root: str) -> dict[str, Any]:
    """Atomically write only allow-listed project subtrees and bounded payloads."""
    if os.getenv("AGENT_SANDBOX_ENABLED", "1") != "1":
        return {"error": "Agent write sandbox is disabled by policy."}
    if len(content.encode("utf-8")) > int(os.getenv("AGENT_MAX_WRITE_BYTES", "1048576")):
        return {"error": "Write exceeds AGENT_MAX_WRITE_BYTES."}
    try:
        relative = Path(path).resolve().relative_to(Path(project_root).resolve())
    except ValueError:
        return {"error": "Write target escapes the project root."}
    if not relative.parts or relative.parts[0] not in _WRITE_ROOTS:
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
