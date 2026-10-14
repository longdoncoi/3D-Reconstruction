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
import uuid
from pathlib import Path
from typing import Any

from ai_assistant.settings import agent_write_roots, load_agent_runtime_settings

_ALLOWED = {"cmake", "ctest", "python", "pytest", "ruff", "git"}
# ADR 0011: allow-list must restrict subcommand / flag; deny arbitrary code execution.
# An entry ending in "/" is a path prefix (``pytest tests/…``); a bare word must
# match an entry verbatim. Flags (``-q``/``--build``) are accepted by position,
# except the two that turn an allow-listed binary into an escape hatch
# (``cmake -E …`` and ``git -c …``), which carry their own rules below.
_ALLOWED_SUBCOMMANDS = {
    "cmake": {"-E", "--build", "--install", "--version", "-S", "-B", "-P"},
    "git": {
        "status", "log", "diff", "add", "commit", "branch", "checkout", "clone",
        "pull", "push", "fetch", "merge", "rebase", "tag",
        # read-only / local-history operations added with the enforced list
        "show", "stash", "remote", "blame", "ls-files", "describe", "shortlog",
        "bisect", "init", "mv", "reset", "restore", "revert", "cherry-pick",
    },
    "pytest": {"tests/", "-k", "-m", "-x", "-q", "--tb"},
    "python": {"-m", "--version", "--help"},
    "ruff": {"check", "format", "--version", "-q"},
    "ctest": {"-R", "-N", "--output-on-failure", "--timeout"},
}
# Binaries that legitimately take a script / test-path / build-dir positional.
_PATH_POSITIONAL = frozenset({"python", "pytest", "ruff", "ctest", "cmake"})
# ``cmake -E`` sub-sub-commands that cannot launch a process or write outside
# the workspace. ``env`` and ``chdir`` are the documented escapes (they run the
# remainder of the command line), so everything not listed here is refused.
_CMAKE_E_BENIGN = frozenset({"cmake_echo_color", "cat", "md5sum", "sha1sum", "true", "false"})
# Deny-list for paths that should never be modified by agent (policy files, secrets)
_DENIED_WRITE_PATHS = {"sandbox.py", "security.py", ".env", ".env.example", "pyproject.toml"}
# Preserve block for metacharacters but avoid false substring matches using token-level checks
_FORBIDDEN_METACHAR = {"&&", "||", ";", "|", ">", "<", "`", "$("}
# Block dangerous words only when they appear as standalone tokens (mitigate false positive on "del"/"rm")
_FORBIDDEN_STANDALONE = {"rm", "del", "format", "shutdown"}

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


def _policy_rejection(command: str) -> str | None:
    """Reject shell metacharacters and dangerous standalone tokens (ADR 0011).

    Token-level matching keeps ``confirm performance`` usable while ``rm`` as a
    word of its own is refused.
    """
    for meta in _FORBIDDEN_METACHAR:
        if meta in command:
            return "Command rejected by sandbox policy (metacharacter)."
    for token in command.casefold().split():
        if token.lstrip() in _FORBIDDEN_STANDALONE:
            return "Command rejected by sandbox policy (dangerous command token)."
    return None


def _looks_like_path(argument: str) -> bool:
    """True for a script/test/build-dir positional rather than a subcommand."""
    return "/" in argument or "\\" in argument or argument.startswith(".") or argument.endswith(".py")


def _subcommand_rejection(argv: list[str]) -> str | None:
    """Restrict the subcommand surface of allow-listed binaries (ADR 0011).

    Naming the executable alone is not an allow-list: ``cmake -E env <cmd>``
    runs an arbitrary process and ``git -c core.sshCommand=<cmd>`` is RCE from
    config on the next fetch, so those two flags get a positive list while the
    remaining subcommands must appear in ``_ALLOWED_SUBCOMMANDS`` (or be a
    path positional / a plain flag).
    """
    executable = Path(argv[0]).name.casefold()
    subcmd = argv[1] if len(argv) > 1 else None
    if subcmd is None:
        return None
    if executable == "cmake" and subcmd == "-E":
        sub_sub = argv[2] if len(argv) > 2 else None
        if sub_sub in _CMAKE_E_BENIGN:
            return None
        return "Sandbox: dangerous cmake -E subcommand blocked."
    if executable == "git" and "-c" in argv[1:]:
        return "Sandbox: git -c config injection blocked."
    if subcmd.startswith("-"):
        return None  # ordinary flag; the dangerous ones are handled above
    allowed: set[str] = _ALLOWED_SUBCOMMANDS.get(executable, set())
    if subcmd in allowed:
        return None
    if any(entry.endswith("/") and subcmd.startswith(entry) for entry in allowed):
        return None
    if executable in _PATH_POSITIONAL and _looks_like_path(subcmd):
        return None
    return f"Sandbox: subcommand '{subcmd}' is not in the {executable} allow-list."


def _docker_argv(argv: list[str], cwd: str, image: str) -> tuple[list[str], str]:
    """Wrap the command in a bounded, network-less container.

    The generated ``--name`` lets the timeout path force-remove the container:
    killing the ``docker run`` client does NOT stop the container it started.
    """
    container_name = f"agent-sbx-{uuid.uuid4().hex[:12]}"
    wrapped = ["docker", "run", "--rm", "--name", container_name,
               "--network", "none", "--memory", "1g", "--cpus", "1.0",
               "--pids-limit", "128", "-v", f"{cwd}:/workspace:rw",
               "-w", "/workspace", image, *argv]
    return wrapped, container_name


def run(command: str, cwd: str, timeout: int) -> dict[str, Any]:
    runtime_settings = load_agent_runtime_settings()
    if not runtime_settings.sandbox_enabled:
        return {"error": "Agent command sandbox is disabled by policy."}
    rejection = _policy_rejection(command)
    if rejection:
        return {"error": rejection}
    try:
        argv = shlex.split(command, posix=os.name != "nt")
    except ValueError as error:
        return {"error": f"Invalid command: {error}"}
    if not argv or Path(argv[0]).name.casefold() not in _ALLOWED:
        return {"error": "Executable is not in the agent allow-list."}
    sub_rejection = _subcommand_rejection(argv)
    if sub_rejection:
        return {"error": sub_rejection}
    if _inline_code_requested(argv):
        return {"error": "Inline interpreter code (-c, -m, -) is rejected by sandbox policy."}
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONUTF8": "1"}
    container_name = None
    sandbox_label = "local-allowlist"
    if runtime_settings.sandbox_runtime == "docker":
        argv, container_name = _docker_argv(argv, cwd, runtime_settings.sandbox_image)
        sandbox_label = "docker"
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              timeout=min(timeout, 120), env=env, shell=False,
                              encoding="utf-8", errors="replace", check=False)
        return {"command": argv, "return_code": proc.returncode,
                "stdout": proc.stdout[:5000], "stderr": proc.stderr[:2000],
                "sandbox": sandbox_label}
    except subprocess.TimeoutExpired:
        if container_name:
            _force_remove_container(container_name)
        return {"error": f"Command timed out after {timeout}s"}


def _force_remove_container(container_name: str) -> None:
    """Best-effort ``docker rm -f`` so a timed-out sandbox cannot outlive the call.

    ADR 0011: the local runtime already has a hard ``timeout``; the Docker
    runtime must be equally bounded, otherwise ``docker run`` timing out leaves
    the container running with the workspace mounted read-write.
    """
    try:
        subprocess.run(["docker", "rm", "-f", container_name],
                       capture_output=True, timeout=15, shell=False, check=False)
    except (OSError, subprocess.SubprocessError):
        # The container may already be gone, or the daemon unreachable — the
        # command result is still the timeout error either way.
        pass


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
    # ADR 0015 / ADR 0011: deny writing to policy / sandbox / secrets paths
    target_name = Path(path).name
    if target_name in _DENIED_WRITE_PATHS or any(p in str(Path(path).resolve()).lower() for p in ["sandbox.py", ".env", "security.py"]):
        return {"error": "Write target is denied by sandbox policy (policy file)."}
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
