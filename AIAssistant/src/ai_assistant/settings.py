from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── Canonical local bind address (single source of truth, ADR 0001) ─────────
# The server binds loopback only; every default URL and the transport policy's
# implicit loopback origins derive from these two values instead of repeating
# "127.0.0.1:8080" across adapters and scripts.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_BASE_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"


def _env_flag(name: str, default: bool = False, env: Mapping[str, str] | None = None) -> bool:
    """Read a boolean switch from the environment.

    ``1``/``true``/``yes``/``on`` (case-insensitive) are true; everything else
    — including an unset variable — is false. An explicitly set variable always
    wins, including ``0``: that is how an operator turns a surface off for one
    session even though the deployment file enables it. An *unset* variable
    falls back to ``default``.
    """
    values = os.environ if env is None else env
    raw = values.get(name, "").strip()
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


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
    #: ADR 0032: the interactive OpenAPI surface (``/docs``, ``/redoc``,
    #: ``/openapi.json``) is an *app-level* FastAPI route, so the transport
    #: guard on the routers never reaches it. It is therefore off unless a
    #: deployment opts in — a network-reachable bind must not publish its own
    #: route schema by default.
    expose_api_docs: bool = False
    #: Host header allow-list for the trusted-host middleware. Empty means the
    #: middleware is not installed (the operator has not declared a policy).
    allowed_hosts: tuple[str, ...] = ()

    @classmethod
    def load(cls, base_dir: Path, config_dir: Path | None = None) -> "ArchitectureSettings":
        """Load profile configuration for this deployment.

        ``config_dir`` defaults to the *tracked* TOML tree resolved from this
        file (``AIAssistant/config``), not to ``base_dir / "config"``: the
        composition root historically passed ``…/src`` as ``base_dir``, which
        made every ``*.toml`` silently ignored — capabilities, the least
        privilege scope map and the origin allow-list all fell back to code
        defaults while the files sat unread on disk (ADR 0015). Callers that
        build a settings tree from their own directory (tests, embedded
        profiles) pass ``config_dir`` explicitly.
        """
        base_dir = Path(base_dir).resolve()
        profile = os.environ.get("AI_ASSISTANT_PROFILE", "desktop").strip().lower()
        config_dir = Path(config_dir) if config_dir is not None else settings_config_dir()
        payload: dict[str, Any] = {}
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
            # ADR 0032: the docs surface and the host allow-list are policy,
            # so they are resolved once here instead of per request. The env
            # variable wins over the TOML file so an operator can expose the
            # docs for one debugging session without editing the deployment.
            expose_api_docs=_env_flag("AGENT_EXPOSE_API_DOCS", bool(api.get("expose_docs", False))),
            allowed_hosts=tuple(
                part.strip()
                for part in (
                    os.environ.get("AGENT_ALLOWED_HOSTS", "")
                    or ",".join(str(item) for item in api.get("allowed_hosts", []))
                ).split(",")
                if part.strip()
            ),
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
    #: Production gates (ADR 0023): refuse non-isolated command execution and
    #: refuse to run on a non-durable checkpoint backend instead of silently
    #: degrading. Both default to off so the Qt desktop distribution keeps
    #: working without Docker or a database.
    sandbox_require_isolation: bool = False
    checkpoint_required: bool = False
    #: Global agent-run budget (0 = unlimited). One run per session is
    #: already enforced by the service; this caps total concurrent runs.
    max_concurrent_runs: int = 0

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
            sandbox_require_isolation=values.get("AGENT_SANDBOX_REQUIRE_ISOLATION", "0") == "1",
            checkpoint_required=values.get("AGENT_CHECKPOINT_REQUIRED", "0") == "1",
            max_concurrent_runs=int(values.get("AGENT_MAX_CONCURRENT_RUNS", "0")),
        )


def load_agent_runtime_settings(env: Mapping[str, str] | None = None) -> AgentRuntimeSettings:
    """Read the agent-runtime toggles from ``env`` (defaults to ``os.environ``)."""
    return AgentRuntimeSettings.from_env(env)


def vision_llm_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the vision LLM path is active (``AI_ENABLE_VISION_LLM``)."""
    values = os.environ if env is None else env
    return values.get("AI_ENABLE_VISION_LLM", "1").strip().lower() in {"1", "true", "yes", "on"}


def rag_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the RAG pipeline is active (``AI_ENABLE_RAG``)."""
    values = os.environ if env is None else env
    return values.get("AI_ENABLE_RAG", "1").strip().lower() in {"1", "true", "yes", "on"}


def app_data_dir(env: Mapping[str, str] | None = None) -> str:
    """Return the ``APP_DATA_DIR`` deployment override (``""`` when unset)."""
    values = os.environ if env is None else env
    return values.get("APP_DATA_DIR", "").strip()


def storage_key(env: Mapping[str, str] | None = None) -> str:
    """Return the explicit at-rest key override (``""`` when unset).

    ``AI_STORAGE_KEY`` accepts ``base64:<url-safe>``, ``hex:<digits>`` or a
    bare URL-safe base64 value; the persistence adapter validates the decoded
    length. Keeping the raw string here preserves ADR 0001's single owner for
    environment reads while the crypto adapter owns key parsing.
    """
    values = os.environ if env is None else env
    return values.get("AI_STORAGE_KEY", "").strip()


def storage_key_location(env: Mapping[str, str] | None = None) -> tuple[Path, bool]:
    """Resolve where the generated local at-rest key lives.

    Returns ``(path, explicit)``. ``AI_STORAGE_KEY_FILE`` is the explicit
    deployment/secret-store path; when it is set a missing file must fail
    startup instead of silently generating a replacement key (which would make
    existing state unreadable). Without it the key lives outside
    ``APP_DATA_DIR`` (``%LOCALAPPDATA%`` on Windows, ``$XDG_CONFIG_HOME`` on
    POSIX) so a backup of the data volume does not automatically copy the key
    with the ciphertext.
    """
    values = os.environ if env is None else env
    raw = values.get("AI_STORAGE_KEY_FILE", "").strip()
    if raw:
        return Path(raw).expanduser(), True
    if sys.platform == "win32":
        root = values.get("LOCALAPPDATA", "").strip() or values.get("APPDATA", "").strip()
        base = Path(root).expanduser() if root else Path.home() / "AppData" / "Local"
    else:
        root = values.get("XDG_CONFIG_HOME", "").strip()
        base = Path(root).expanduser() if root else Path.home() / ".config"
    return base / "3d-reconstruction" / "ai_assistant.key", False


def max_request_bytes(env: Mapping[str, str] | None = None) -> int:
    """Return the request-body cap for every HTTP surface (2 MiB default)."""
    values = os.environ if env is None else env
    return int(values.get("AGENT_MAX_REQUEST_BYTES", "2097152"))


def rate_limit_per_minute(env: Mapping[str, str] | None = None) -> int:
    """Return the per-peer HTTP request budget (``0`` = no limit).

    Fixed window so the number is explainable in an ops runbook; the middleware
    answers ``429`` with ``Retry-After`` when the budget is spent.
    """
    values = os.environ if env is None else env
    return int(values.get("AGENT_RATE_LIMIT_PER_MINUTE", "0"))


def apply_utf8_environment_default() -> None:
    """Force this process into UTF-8 mode (console/Qt encoding compatibility).

    The *only* sanctioned writer of ``PYTHONUTF8`` (ADR 0014): the logging
    bootstrap and the Qt process manager both route their need for it through
    here instead of assigning ``os.environ`` themselves.
    """
    os.environ["PYTHONUTF8"] = "1"


@dataclass(frozen=True, slots=True)
class TransportTokens:
    """The four surface secrets the transport policy enforces (ADR 0008).

    Parsed here so ``settings`` stays the sole environment owner (ADR 0014):
    the HTTP security adapter receives its values from this accessor and
    never touches ``os.environ`` itself.
    """

    admin: str
    agent: str
    a2a: str
    mcp: str


def transport_tokens(env: Mapping[str, str] | None = None) -> TransportTokens:
    """Read the ``AI_*_TOKEN`` transport secrets from ``env``."""
    values = os.environ if env is None else env
    return TransportTokens(
        admin=values.get("AI_ADMIN_TOKEN", "").strip(),
        agent=values.get("AI_AGENT_TOKEN", "").strip(),
        a2a=values.get("AI_A2A_TOKEN", "").strip(),
        mcp=values.get("AI_MCP_TOKEN", "").strip(),
    )


def a2a_tenants(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Read the A2A bearer-token → tenant mapping from ``env``.

    ``AI_A2A_TENANTS`` is a JSON object mapping each accepted bearer token to
    the tenant id that owns its tasks. When empty the surface stays
    single-tenant: the single ``AI_A2A_TOKEN`` maps every task to ``"default"``.
    """
    import json

    values = os.environ if env is None else env
    raw = values.get("AI_A2A_TENANTS", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(token): str(tenant) for token, tenant in parsed.items() if token and tenant}


def a2a_max_subscribers(env: Mapping[str, str] | None = None) -> int:
    """Admission cap for concurrent A2A SSE subscribers.

    A deadline bounds each stream; this bounds how many can be open at once.
    Invalid and non-positive values fall back to the documented default rather
    than disabling the guard.
    """
    values = os.environ if env is None else env
    raw = values.get("AI_A2A_MAX_SUBSCRIBERS", "").strip()
    try:
        limit = int(raw)
    except ValueError:
        return 32
    return limit if limit > 0 else 32


def task_retention_days(env: Mapping[str, str] | None = None) -> int:
    """Days to keep terminal A2A tasks (``AI_TASK_RETENTION_DAYS``).

    ``0`` disables pruning (forensic deployments). Invalid and negative values
    keep the documented 30-day default rather than accidentally retaining
    everything.
    """
    values = os.environ if env is None else env
    raw = values.get("AI_TASK_RETENTION_DAYS", "").strip()
    try:
        days = int(raw)
    except ValueError:
        return 30
    return days if days >= 0 else 30


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


def otlp_exporter_endpoint(env: Mapping[str, str] | None = None) -> str | None:
    """Return the OTLP collector URL for span export, or ``None``.

    Spans only exist while ``AGENT_OBSERVABILITY=1`` (see
    :func:`observability_enabled`), so the endpoint is honoured only then too.
    The URL is the full traces path (e.g. ``http://host:4318/v1/traces``) —
    the standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` base-URL semantics are
    deliberately not mirrored here to keep one unambiguous knob.
    """
    values = os.environ if env is None else env
    if not observability_enabled(values):
        return None
    endpoint = values.get("AGENT_OTLP_ENDPOINT", "").strip()
    return endpoint or None


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


def settings_config_dir() -> Path:
    """The tracked configuration tree: ``AIAssistant/config``.

    Resolved from this file (``src/ai_assistant/settings.py`` → two levels up),
    so every reader — ``load_settings`` and ``ArchitectureSettings.load`` —
    sees the same ``base.toml``/``<profile>.toml``/``agents.toml`` no matter
    which directory a caller believes is its base.
    """
    return Path(__file__).resolve().parents[2] / "config"


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


# ── Socket binding (ADR 0001: one owner for the uvicorn bind) ───────────────

def agent_bind_host(env: Mapping[str, str] | None = None) -> str:
    """Return the uvicorn bind address (``AGENT_HOST``).

    Default stays the canonical loopback host: exposing the interface is a
    deployment decision that has to be made explicitly (the Docker image sets
    ``AGENT_HOST=0.0.0.0``), and ADR 0008 still gates the mutating surfaces.
    """
    values = os.environ if env is None else env
    return values.get("AGENT_HOST", "").strip() or DEFAULT_HOST


def agent_bind_port(env: Mapping[str, str] | None = None) -> int:
    """Return the uvicorn port (``AGENT_PORT``), clamped to a valid range."""
    values = os.environ if env is None else env
    raw = values.get("AGENT_PORT", "").strip()
    try:
        port = int(raw) if raw else DEFAULT_PORT
    except ValueError:
        return DEFAULT_PORT
    return port if 1 <= port <= 65535 else DEFAULT_PORT
