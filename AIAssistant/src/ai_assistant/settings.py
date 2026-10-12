from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# ── Canonical local bind address (single source of truth, ADR 0001) ─────────
# The server binds loopback only; every default URL and the transport policy's
# implicit loopback origins derive from these two values instead of repeating
# "127.0.0.1:8080" across adapters and scripts.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_BASE_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"


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
    a2a_card_name: str = "3D-Reconstruction AI Assistant"
    a2a_card_url: str = DEFAULT_BASE_URL

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
        data_dir_value = (data_dir() or Path(base_dir.parent)) / "AIAssistant"
        scopes_raw = agents.get("scopes", {})
        capability_scopes = {
            str(name): frozenset(str(scope) for scope in scopes)
            for name, scopes in scopes_raw.items()
            if isinstance(scopes, list)
        }
        return cls(
            profile=profile,
            data_dir=data_dir_value,
            enable_mcp=bool(runtime.get("enable_mcp", True)),
            enable_a2a=bool(runtime.get("enable_a2a", True)),
            allow_remote_a2a=bool(runtime.get("allow_remote_a2a", False)),
            trusted_a2a_endpoints=frozenset(str(item) for item in a2a.get("trusted_endpoints", [])),
            agent_capabilities=frozenset(str(item) for item in agents.get("capabilities", [])),
            capability_scopes=capability_scopes,
            allowed_plugins=frozenset(str(item) for item in plugins.get("enabled", ["builtin.legacy-tools"])),
            allowed_origins=tuple(str(item) for item in api.get("allowed_origins", [])),
            # A2A transport configuration (ADR 0001: centralized in settings, not import-time env reads)
            a2a_card_name=str(runtime.get("a2a_card_name", "3D-Reconstruction AI Assistant")),
            a2a_card_url=str(runtime.get("a2a_card_url", DEFAULT_BASE_URL)),
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


def use_langgraph_agent(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the LangGraph decision loop is enabled.

    Parsed here instead of a legacy module-level global so the agent layer reads
    the toggle through ``settings`` (ADR 0001). Defaults to enabled.
    """
    values = os.environ if env is None else env
    return values.get("USE_LANGGRAPH_AGENT", "1") != "0"


def force_langgraph_agent(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the LangGraph loop is forced on regardless of routing.

    Counterpart of :func:`use_langgraph_agent`; parsed here so both halves of
    the same toggle live in one module (ADR 0001).
    """
    values = os.environ if env is None else env
    return values.get("FORCE_LANGGRAPH_AGENT", "1") == "1"


def vision_llm_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the vision LLM path is active (``AI_ENABLE_VISION_LLM``)."""
    values = os.environ if env is None else env
    return values.get("AI_ENABLE_VISION_LLM", "1").strip().lower() in {"1", "true", "yes", "on"}


# ── One-time process environment bootstrap ──────────────────────────────────
# ``.env`` parsing and the library-quieting defaults used to be inlined in
# ``legacy/config.py`` and ran as import-time side effects there, which made
# the shim a second environment owner (ADR 0001). They now live here, behind
# functions the composition root calls explicitly.

#: Environment keys that get a safe default when the operator did not set them.
_STARTUP_ENV_DEFAULTS: dict[str, str] = {
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "TF_ENABLE_ONEDNN_OPTS": "0",
}

_startup_env_applied = False


def load_local_dotenv(base_dir: str | Path) -> None:
    """Merge ``<base_dir>/.env`` into ``os.environ`` without overwriting.

    Pure parser kept in one place so no other module re-implements ``KEY=value``
    splitting. Missing files and unreadable lines are ignored: the file is a
    convenience for local runs, never a required input.
    """
    env_path = Path(base_dir) / ".env"
    try:
        raw_lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value)


def apply_startup_environment_defaults(*, reload: bool = False) -> None:
    """Apply the process-wide env defaults and warning filters exactly once.

    Idempotent so calling it from more than one bootstrap path is safe; pass
    ``reload`` to force a re-run (tests that reset ``os.environ``).
    """
    global _startup_env_applied
    if _startup_env_applied and not reload:
        return
    for key, value in _STARTUP_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", module="keras")
    _startup_env_applied = True


# ── Runtime toggles read on demand ───────────────────────────────────────────
# ADR 0001 forbids reading the environment at import time anywhere except this
# module and ``bootstrap``. These accessors keep that boundary explicit while
# allowing modules to consume their knob lazily from ``settings`` at call time.

def observability_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether metrics/tracing exporters are active (``AGENT_OBSERVABILITY=1``)."""
    values = os.environ if env is None else env
    return values.get("AGENT_OBSERVABILITY", "0") == "1"


def langsmith_settings(env: Mapping[str, str] | None = None) -> dict[str, str] | None:
    """Return LangSmith credentials when tracing is enabled, else ``None``.

    ``None`` is returned unless ``LANGSMITH_TRACING`` is truthy and
    ``LANGSMITH_API_KEY`` is set, so the observability adapter never has to
    re-implement the toggle.
    """
    values = os.environ if env is None else env
    if values.get("LANGSMITH_TRACING", "").lower() not in {"true", "1", "yes"}:
        return None
    api_key = values.get("LANGSMITH_API_KEY", "")
    if not api_key:
        return None
    return {
        "project": values.get("LANGSMITH_PROJECT", "3d-reconstruction"),
        "api_key": api_key,
        "endpoint": values.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    }


def a2a_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the A2A transport layer is active (``A2A_ENABLED=1``)."""
    values = os.environ if env is None else env
    return values.get("A2A_ENABLED", "0") == "1"


def a2a_remote_agent_urls(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Return the trusted remote agent card URLs from ``A2A_REMOTE_AGENTS``."""
    values = os.environ if env is None else env
    return tuple(
        url.strip()
        for url in values.get("A2A_REMOTE_AGENTS", "").split(",")
        if url.strip()
    )


def a2a_trusted_endpoints(env: Mapping[str, str] | None = None) -> frozenset[str]:
    """Deployment-wide A2A endpoint allowlist (ADR 0002).

    Merges the two operator signals into one enforcement set so the allowlist
    actually reaches :class:`~ai_assistant.adapters.a2a_client.TrustedA2AClient`:

    - ``[a2a] trusted_endpoints`` from ``config/{profile}.toml`` (via the
      cached :class:`ArchitectureSettings`), and
    - the ``A2A_REMOTE_AGENTS`` environment override (comma-separated).

    Neither source is trusted by default; the returned set is empty until an
    operator configures one. The remote A2A adapter refuses any endpoint
    outside this set.
    """
    values = os.environ if env is None else env
    endpoints = {url.rstrip("/") for url in a2a_remote_agent_urls(values)}
    endpoints.update(url.rstrip("/") for url in load_settings().trusted_a2a_endpoints)
    return frozenset(url for url in endpoints if url)


# ── Cached platform settings (parsed once through ArchitectureSettings) ─────
# ADR 0001: the TOML files are parsed in exactly one place. This cache lets
# on-demand accessors (e.g. a2a_trusted_endpoints) reuse that parser without a
# composition root, while ``invalidate_settings_cache`` keeps tests honest.
_settings_cache: ArchitectureSettings | None = None


def load_settings() -> ArchitectureSettings:
    """Return the platform settings, parsed once and cached.

    The base directory is resolved from this file's location
    (``src/ai_assistant/settings.py`` → package root two levels up), so
    accessors work whether or not a composition root has run.
    """
    global _settings_cache
    if _settings_cache is None:
        package_root = Path(__file__).resolve().parents[2]
        _settings_cache = ArchitectureSettings.load(package_root)
    return _settings_cache


def invalidate_settings_cache() -> None:
    """Drop the cached settings so the next ``load_settings`` re-reads config."""
    global _settings_cache
    _settings_cache = None


def a2a_card_name(env: Mapping[str, str] | None = None) -> str:
    """Display name for this server's published Agent Card."""
    values = os.environ if env is None else env
    return values.get("A2A_CARD_NAME", "3D-Reconstruction AI Assistant")


def a2a_card_url(env: Mapping[str, str] | None = None) -> str:
    """Public base URL where this server is reachable."""
    values = os.environ if env is None else env
    return values.get("A2A_CARD_URL", DEFAULT_BASE_URL)


def agent_write_roots(env: Mapping[str, str] | None = None) -> frozenset[str]:
    """Subtree allow-list for sandboxed writes (``AGENT_WRITE_ALLOWLIST``)."""
    values = os.environ if env is None else env
    return frozenset(
        part.strip()
        for part in values.get(
            "AGENT_WRITE_ALLOWLIST", "src,AIAssistant,AIComputerVision,Config,Docs,tests"
        ).split(",")
        if part.strip()
    )


def lsp_binaries(env: Mapping[str, str] | None = None) -> tuple[str, str, int]:
    """Resolve the LSP server executables and request timeout.

    Returns ``(clangd_binary, pylsp_binary, timeout_seconds)``.
    """
    values = os.environ if env is None else env
    try:
        timeout = int(values.get("AGENT_LSP_TIMEOUT", "15"))
    except ValueError:
        timeout = 15
    return (
        values.get("AGENT_CLANGD_BIN", "clangd"),
        values.get("AGENT_PYLSP_BIN", "pylsp"),
        timeout,
    )


def data_dir(env: Mapping[str, str] | None = None) -> Path | None:
    """Return ``APP_DATA_DIR`` when set (the user-data root), else ``None``.

    Single owner of the ``APP_DATA_DIR`` read: ``ArchitectureSettings.load``
    and deploy-time resolvers (e.g. the supervisor audit path) must go through
    this accessor so the environment variable is parsed in exactly one place
    (ADR 0001).
    """
    values = os.environ if env is None else env
    raw = values.get("APP_DATA_DIR", "").strip()
    return Path(raw) if raw else None


# ── Inference backend toggles (ADR 0001: parsed in settings, read on demand) ─

def agent_inference_backend(env: Mapping[str, str] | None = None) -> str:
    """Return the configured inference backend mode (``AGENT_INFERENCE_BACKEND``)."""
    values = os.environ if env is None else env
    return values.get("AGENT_INFERENCE_BACKEND", "llama_cpp").casefold()


def agent_hybrid_policy(env: Mapping[str, str] | None = None) -> str:
    """Return the hybrid routing privacy policy (``AGENT_HYBRID_POLICY``)."""
    values = os.environ if env is None else env
    return values.get("AGENT_HYBRID_POLICY", "local_only").casefold()


def agent_inference_url(env: Mapping[str, str] | None = None) -> str:
    """Return the OpenAI-compatible remote endpoint (``AGENT_INFERENCE_URL``)."""
    values = os.environ if env is None else env
    return values.get("AGENT_INFERENCE_URL", "").rstrip("/")


def agent_inference_timeout(env: Mapping[str, str] | None = None) -> int:
    """Return the remote completion timeout in seconds (``AGENT_INFERENCE_TIMEOUT``)."""
    values = os.environ if env is None else env
    return int(values.get("AGENT_INFERENCE_TIMEOUT", "60"))


def agent_inference_model(env: Mapping[str, str] | None = None) -> str:
    """Return the remote completion model id (``AGENT_INFERENCE_MODEL``)."""
    values = os.environ if env is None else env
    return values.get("AGENT_INFERENCE_MODEL", "default")
