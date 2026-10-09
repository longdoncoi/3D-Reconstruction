from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArchitectureSettings:
    """Validated, explicit runtime configuration for the new platform.

    Existing legacy environment variables remain isolated in compatibility
    adapters.  New components consume this object instead of reading process
    environment at import time.
    """

    profile: str
    data_dir: Path
    enable_mcp: bool
    enable_a2a: bool
    allow_remote_a2a: bool
    trusted_a2a_endpoints: frozenset[str]
    agent_capabilities: frozenset[str]
    capability_scopes: Mapping[str, frozenset[str]]
    allowed_plugins: frozenset[str]
    allowed_origins: tuple[str, ...]

    @classmethod
    def load(cls, base_dir: Path) -> "ArchitectureSettings":
        base_dir = Path(base_dir).resolve()
        profile = os.environ.get("AI_ASSISTANT_PROFILE", "desktop").strip().lower()
        config_dir = base_dir / "config"
        payload: dict = {}
        for path in (
            config_dir / "base.toml", config_dir / f"{profile}.toml", config_dir / "plugins.toml",
            config_dir / "agents.toml",
        ):
            if path.exists():
                with path.open("rb") as handle:
                    payload.update(tomllib.load(handle))
        runtime = payload.get("runtime", {})
        plugins = payload.get("plugins", {})
        a2a = payload.get("a2a", {})
        agents = payload.get("agents", {})
        api = payload.get("api", {})
        data_dir = Path(os.environ.get("APP_DATA_DIR", str(base_dir.parent))) / "AIAssistant"
        scopes_raw = agents.get("scopes", {})
        capability_scopes = {
            str(name): frozenset(str(scope) for scope in scopes)
            for name, scopes in scopes_raw.items()
            if isinstance(scopes, list)
        }
        return cls(
            profile=profile,
            data_dir=data_dir,
            enable_mcp=bool(runtime.get("enable_mcp", True)),
            enable_a2a=bool(runtime.get("enable_a2a", True)),
            allow_remote_a2a=bool(runtime.get("allow_remote_a2a", False)),
            trusted_a2a_endpoints=frozenset(str(item) for item in a2a.get("trusted_endpoints", [])),
            agent_capabilities=frozenset(str(item) for item in agents.get("capabilities", [])),
            capability_scopes=capability_scopes,
            allowed_plugins=frozenset(str(item) for item in plugins.get("enabled", ["builtin.legacy-tools"])),
            allowed_origins=tuple(str(item) for item in api.get("allowed_origins", [])),
        )


@dataclass(frozen=True, slots=True)
class AgentRuntimeSettings:
    """Agent-runtime toggles parsed in exactly one place.

    These knobs previously used raw ``os.getenv`` scattered across the sandbox,
    checkpoint adapter and completion adapter. Centralising the parsing keeps
    the defaults (and their bool/int semantics) consistent. It is read on demand
    so tests can still override the environment at runtime.
    """

    sandbox_enabled: bool
    sandbox_runtime: str
    sandbox_image: str
    max_write_bytes: int
    checkpoint_backend: str
    checkpoint_url: str
    checkpoint_path: str
    native_tool_calls: bool

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AgentRuntimeSettings":
        values = os.environ if env is None else env
        return cls(
            sandbox_enabled=values.get("AGENT_SANDBOX_ENABLED", "1") == "1",
            sandbox_runtime=values.get("AGENT_SANDBOX_RUNTIME", "local").casefold(),
            sandbox_image=values.get("AGENT_SANDBOX_IMAGE", "python:3.11-slim"),
            max_write_bytes=int(values.get("AGENT_MAX_WRITE_BYTES", "1048576")),
            checkpoint_backend=values.get("AGENT_CHECKPOINT_BACKEND", "memory").casefold(),
            checkpoint_url=values.get("AGENT_CHECKPOINT_URL", ""),
            checkpoint_path=values.get("AGENT_CHECKPOINT_PATH", "AIAssistant/Cache/checkpoints.sqlite"),
            native_tool_calls=values.get("AGENT_NATIVE_TOOL_CALLS", "0") == "1",
        )


def load_agent_runtime_settings(env: Mapping[str, str] | None = None) -> AgentRuntimeSettings:
    """Read the agent-runtime toggles from ``env`` (defaults to ``os.environ``)."""
    return AgentRuntimeSettings.from_env(env)
