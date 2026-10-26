"""FastAPI router and entry point for the 3D-Reconstruction AI server.

This file is the composition root (bootstrap) that wires together all
HTTP adapters, modules, and services. Business logic lives in sub-modules:

  - /admin/*            → src/ai_assistant/adapters/http/admin_routes.py
  - /v1/chat/*          → src/ai_assistant/adapters/http/chat_routes.py
  - /health, /metrics   → src/ai_assistant/adapters/http/health_routes.py
  - /v1/agent/*         → src/ai_assistant/adapters/http/agent_routes.py (injected AgentService)
  - /a2a/*              → src/ai_assistant/adapters/a2a_router.py
  - /mcp                → src/ai_assistant/adapters/mcp.py (mounted behind the
                          MCP transport guard, least-privilege principal)

Composition is explicit and lazy. Importing this module only defines it — no
log file is opened, no config is read, no ``FastAPI`` app is built and asyncio
is left alone, and no third-party framework is imported: FastAPI, LangGraph, llama.cpp and the MCP SDK all load inside :func:`compose()` (or the helper that needs them), so importing this module pulls only the standard library and ``ai_assistant.settings``. The whole object graph (logging, model registry, tool catalog,
container, services, app, routers) is built once by :func:`compose`, which
runs on first access of a composed name (module ``__getattr__``), from
``main()``/``bootstrap_runtime()``, or when called directly.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator

# Ensure package root is on path before any ai_assistant import. Both inserts
# fire only when the file is launched from a source tree without PYTHONPATH;
# every in-process import (the test suite included) already carries both
# dirs, so the branches are unreachable here — declared exclusions, not
# unmeasured code.
_PACKAGE_ROOT = str(Path(__file__).resolve().parents[3])
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)  # pragma: no cover - source-launch bootstrap
_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)  # pragma: no cover - source-launch bootstrap

from ai_assistant.settings import (
    agent_bind_host,
    agent_bind_port,
    apply_startup_environment_defaults,
    load_agent_runtime_settings,
    load_local_dotenv,
    settings_config_dir,
    task_retention_days,
    vision_llm_enabled,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ai_assistant.adapters.http.security import TransportPolicy
    from ai_assistant.adapters.mcp import McpToolServer
    from ai_assistant.agents.service import AgentService
    from ai_assistant.bootstrap.container import PlatformContainer
    from ai_assistant.bootstrap.modules import RuntimeModules
    from ai_assistant.config import AppPaths, ModelRegistry, RAGSettings
    from ai_assistant.tools import ToolRegistry

# The composition root owns the process environment (ADR 0001): ``compose()``
# merges ``.env`` and applies the library-quieting defaults *before* importing
# the A2A/transport modules that snapshot their settings. Importing this module
# therefore still reads no configuration.


def _ensure_package_root_on_path(package_root: str) -> None:
    """Make ``python bootstrap/server.py`` work from any working directory."""
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


def _silence_connection_reset(transport_cls: type) -> None:
    """Wrap ``_call_connection_lost`` so an abrupt client reset is not an error.

    Windows' proactor event loop raises ``ConnectionResetError`` from the pipe
    shutdown callback after a client disconnects abruptly; the traceback is
    noise, so the wrapper swallows exactly that exception.
    """
    original = transport_cls._call_connection_lost  # type: ignore[attr-defined]

    def _silenced(self: Any, exc: Exception) -> None:
        try:
            original(self, exc)
        except ConnectionResetError:
            pass

    transport_cls._call_connection_lost = _silenced  # type: ignore[attr-defined]


def _patch_proactor_transport() -> None:
    """Install the Windows proactor pipe-reset shim (no-op on POSIX)."""
    if sys.platform != "win32":
        return
    from asyncio.proactor_events import _ProactorBasePipeTransport

    _silence_connection_reset(_ProactorBasePipeTransport)


_ensure_package_root_on_path(str(Path(__file__).resolve().parents[2]))

# ── Canonical module-level state (replaces legacy config import) ───────────

_SERVER_START_TIME = time.monotonic()
BASE_DIR = str(Path(__file__).resolve().parents[2])
# ``BASE_DIR`` (…/AIAssistant/src) is the data/model root the legacy resolvers
# expect. The *tracked* deployment assets — ``config/*.toml`` and the ``logs/``
# directory the Qt server manager tails — live one level up in the package
# root; reading them from ``src/`` silently found nothing.
PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def _resolve_model_idx(argv: list[str], models: list[dict[str, Any]]) -> int:
    """Resolve ``-1``/``99999``/non-numeric model selectors to a safe index.

    Kept as a function (not an inline guard) so the clamping is exercised by a
    test instead of being excluded from coverage.
    """
    try:
        idx = int(argv[1]) if len(argv) > 1 else 0
    except (ValueError, IndexError):
        return 0
    if idx < 0 or idx >= len(models):
        return 0
    return idx


# The inference defaults live in ``ai_assistant.llm.defaults`` and load with
# the backend inside :func:`compose()` — never at import.


# ── Runtime module graph ──────────────────────────────────────────────────────
# The LLM/RAG handles and the chatbot-agent holder live on the single
# ``RuntimeModules`` instance compose() builds and injects into the tool
# catalog, the services and the routers (bootstrap.modules). No service
# locator remains: every consumer receives the handles it uses.
#
# This module is deliberately the *process-wide* composition root for the
# desktop server: ``_COMPOSED`` and the composed names below are its single
# instance (guarded by ``_COMPOSE_LOCK`` and built lazily). Running two
# servers in one process is not supported; an embedding host must compose in
# a separate process instead of importing this module twice.


# ── Lifespan ──────────────────────────────────────────────────────────────────
async def _discover_a2a_peers(registry: Any) -> None:
    """Populate the remote A2A registry from the deployment allowlist.

    Discovery is a startup concern, not an import concern: the blocking HTTP
    fetch runs in a worker thread. With no configured endpoints — the default
    — this is a no-op, so a stock desktop install never touches the network.
    """
    import asyncio

    from ai_assistant.settings import a2a_trusted_endpoints

    endpoints = a2a_trusted_endpoints()
    if not endpoints:
        return
    discovered = await asyncio.to_thread(registry.discover, sorted(endpoints))
    logger.info("A2A discovery: %d/%d endpoint(s) reachable", len(discovered), len(endpoints))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        from ai_assistant.adapters.persistence.checkpointing import cleanup_old_checkpoints
        cleanup_old_checkpoints()
    except Exception as e:
        logger.warning(f"Lỗi dọn dẹp checkpoint: {e}")

    registry = getattr(getattr(app, "state", None), "a2a_registry", None)
    if registry is not None:
        try:
            await _discover_a2a_peers(registry)
        except Exception as e:
            logger.warning(f"A2A discovery failed: {e}")

    try:
        retention_days = task_retention_days()
        if retention_days > 0:
            removed = platform.tasks.prune(retention_days)
            if removed:
                logger.info(
                    "Pruned %d task(s) past the %d-day retention window",
                    removed, retention_days,
                )
    except Exception as e:
        logger.warning(f"Task retention pruning failed: {e}")

    total = time.monotonic() - _SERVER_START_TIME
    logger.info("Server ready in %.1fs — http://%s:%d", total,
                agent_bind_host(), agent_bind_port())
    print(f"[SUCCESS] AI Server started successfully ({total:.1f}s)", flush=True)
    try:
        if not platform.settings.enable_mcp:
            # MCP disabled by deployment profile: still yield, otherwise the
            # asynccontextmanager would raise "generator didn't yield" and the
            # server could not start with enable_mcp = false.
            logger.info("MCP endpoint disabled by configuration")
            yield
        elif mcp_server.available:
            async with mcp_server.lifespan():
                yield
        else:
            logger.warning("MCP SDK is not installed; MCP endpoint is unavailable")
            yield
    finally:
        # Close the long-term memory store before the task store: the memory
        # file is smaller and its WAL flush is the cheaper of the two, and a
        # shutdown must not leave either handle open (ADR 0043).
        if memory_store is not None:
            memory_store.close()
        platform.tasks.close()
        logger.info("Server shutdown after %.1fs", time.monotonic() - _SERVER_START_TIME)


# ── FastAPI Application ───────────────────────────────────────────────────────
# Built in compose(): create_app + transport policy + router mounting.


def _refresh_agent_routes() -> None:
    """Reset agent state and schema cache on reload without modifying route table."""
    compose()
    agent_service.reset_state()
    app.openapi_schema = None


def _register_optional_protocols(
    app: FastAPI,
    platform: PlatformContainer,
    mcp_server: McpToolServer,
    transport_policy: TransportPolicy,
) -> None:
    """Wire the A2A and MCP surfaces according to the deployment profile.

    The remote-specialist resolver is injected into ``AgentService`` by
    :func:`compose`; this function only mounts the HTTP surface, so enabling or
    disabling A2A can never leave a stale process-wide lookup behind.
    """
    compose()
    from ai_assistant.adapters.a2a import build_a2a_router
    from ai_assistant.adapters.http.security import (
        mcp_exposed_without_token,
        mcp_guarded_app,
    )

    if platform.settings.enable_a2a:
        app.include_router(build_a2a_router(
            platform.tasks, "3D-Reconstruction AI Assistant", "3.0.0",
            policy=transport_policy,
            card_base_url=platform.settings.a2a_card_url,
        ))
    if platform.settings.enable_mcp and mcp_server.available:
        if mcp_exposed_without_token(agent_bind_host(), transport_policy):
            # Fail closed (ADR 0008): an exposed bind without AI_MCP_TOKEN
            # leaves nothing for the guard to authenticate, so the surface
            # stays unmounted and the reason is logged at startup rather than
            # surfacing as anonymous 403s from every caller.
            logger.warning(
                "MCP endpoint disabled: bind host %s reaches the network and AI_MCP_TOKEN is unset",
                agent_bind_host(),
            )
        else:
            mcp_server.register_plugin_tools(platform.plugins)
            app.mount("/mcp", mcp_guarded_app(mcp_server.asgi_app(), transport_policy))


def _build_supervisor_delegate(
    enable_a2a: bool, a2a_registry: Any, plugin_tools: frozenset[str],
) -> Any:
    """Bind this composition's A2A resolver and plugin row to ``delegate``.

    The helper exists so both branches are directly testable: the composition
    root no longer has an untested ``if enable_a2a`` arm.  The supervisor and
    ``functools`` are imported here, not at module import time, so importing
    ``bootstrap.server`` still pulls only the standard library and settings.
    """
    from functools import partial

    from ai_assistant.orchestration.supervisor import delegate as supervisor_delegate

    kwargs: dict[str, Any] = {"plugin_tools": plugin_tools}
    if enable_a2a:
        kwargs["remote_lookup"] = a2a_registry.resolve
    return partial(supervisor_delegate, **kwargs)


# ── Composition ───────────────────────────────────────────────────────────────
# The object graph is built here, exactly once, in dependency order.
# Importing this module only defines it: main(), bootstrap_runtime() and the
# first access of a composed name (module __getattr__ below) call compose().

# Type declarations for the names compose() binds. Annotations only — no name
# is bound at runtime, so attribute access before composition still falls
# through to __getattr__.
logger: logging.Logger
LOG_FILE_PATH: Path
_paths: AppPaths
_registry: ModelRegistry
MODELS: list[dict[str, Any]]
FALLBACK_TEXT_MODEL: dict[str, Any]
MODEL_IDX: int
active_model_desc: str
_rag_settings: RAGSettings
EMBED_MODEL_NAME: str
USE_VISION_LLM: bool
runtime_modules: RuntimeModules
_tool_registry: ToolRegistry
platform: PlatformContainer
agent_service: AgentService
mcp_server: McpToolServer
app: FastAPI
transport_policy: TransportPolicy
#: The sealed long-term memory store (ADR 0043); compose assigns ``None``
#: when memory is disabled, and the lifespan closes it on shutdown.
memory_store: Any

_COMPOSED = False
_COMPOSE_LOCK = threading.RLock()

# The names compose() creates. Module __getattr__ composes on the first access
# of any of them, so ``server.app`` behaves exactly like an eager import while
# ``import server`` alone opens no log file, reads no config and builds no app.
_COMPOSED_NAMES = frozenset({
    "logger", "LOG_FILE_PATH", "_paths", "_registry", "MODELS",
    "FALLBACK_TEXT_MODEL", "MODEL_IDX", "active_model_desc", "_rag_settings",
    "EMBED_MODEL_NAME", "USE_VISION_LLM", "runtime_modules",
    "_tool_registry", "platform", "agent_service", "mcp_server",
    "app", "transport_policy", "memory_store",
})


def _pending_actions_path() -> str:
    """Canonical on-disk location for pending approvals and UI actions.

    Where durable state lives is a composition-root decision (ADR 0003): the
    agent layer receives the file-backed store already pointed here and never
    names a path itself.
    """
    from ai_assistant.config.paths import get_paths

    paths = get_paths()
    data_root = paths.data_dir if paths.data_dir.name == "AIAssistant" else paths.data_dir / "AIAssistant"
    return str(data_root / "pending_agent_actions.json")


def _memory_store_path() -> str:
    """Canonical on-disk location of the long-term memory store (ADR 0043).

    Same rule as the pending-action file: the path is a composition-root
    decision, the agent layer only ever sees the injected port.
    """
    from ai_assistant.config.paths import get_paths

    paths = get_paths()
    data_root = paths.data_dir if paths.data_dir.name == "AIAssistant" else paths.data_dir / "AIAssistant"
    return str(data_root / "memory.sqlite")


def compose() -> None:
    """Build the object graph exactly once, in dependency order.

    Idempotent and re-entrant (``RLock``): entry points call it
    unconditionally and internal helpers guard their first use of composed
    state with it. A failed build rolls the ``_COMPOSED`` flag back, so the
    next access retries instead of treating a half-built graph as ready.
    """
    global _COMPOSED
    try:
        _compose_once()
    except BaseException:
        _COMPOSED = False
        raise


def _compose_once() -> None:
    """The actual build; the public wrapper owns failure rollback."""
    global _COMPOSED, logger, LOG_FILE_PATH, _paths, _registry, MODELS
    global FALLBACK_TEXT_MODEL, MODEL_IDX, active_model_desc, _rag_settings
    global EMBED_MODEL_NAME, USE_VISION_LLM
    global _tool_registry, platform, agent_service, mcp_server
    global runtime_modules, app, transport_policy, memory_store
    with _COMPOSE_LOCK:
        if _COMPOSED:
            return
        # Set before the body so the compose() → _register_optional_protocols()
        # → compose() back-edge terminates instead of rebuilding. The public
        # wrapper resets it if the body raises.
        _COMPOSED = True

        # Environment first (ADR 0001): the modules imported below snapshot
        # A2A/transport settings at import time, so ``.env`` must be merged
        # before the first of those imports.
        load_local_dotenv(PACKAGE_ROOT)
        apply_startup_environment_defaults()

        # Heavy imports live here — not at module top — so importing this
        # module pulls no third-party framework (ADR 0022). ``settings`` is
        # the only ai_assistant module imported at top level: it owns the
        # process environment and is pure standard library.
        from ai_assistant.adapters.a2a_protocol import A2ARouter, RemoteAgentRegistry
        from ai_assistant.adapters.http import (
            build_admin_router,
            build_agent_router,
            build_chat_router,
            build_health_router,
        )
        from ai_assistant.adapters.http.security import TransportPolicy
        from ai_assistant.adapters.mcp import McpToolServer
        from ai_assistant.adapters.orchestration import LangGraphAgentOrchestrator
        from ai_assistant.adapters.persistence.checkpointing import build_checkpointer
        from ai_assistant.adapters.persistence.pending_store import PendingActionStore
        from ai_assistant.adapters.persistence.storage_crypto import load_storage_cipher
        from ai_assistant.agents.service import AgentService
        from ai_assistant.application.agent_runs import AgentRunService
        from ai_assistant.bootstrap.app_factory import create_app
        from ai_assistant.bootstrap.container import build_container, merge_plugin_tools
        from ai_assistant.bootstrap.modules import (
            LlmModule,
            RagModule,
            RuntimeModules,
            attachment_unsealer,
        )
        from ai_assistant.config import (
            RAGSettings,
            load_model_registry,
            resolve_paths,
            setup_logging,
        )
        from ai_assistant.llm.defaults import CHARS_PER_TOKEN, LLM_N_CTX
        from ai_assistant.llm.prompts import DefaultPromptBuilder
        from ai_assistant.tools import action_manifest
        from ai_assistant.tools.factory import create_tool_registry, platform_executors

        # Logging first: every later step reports through it.
        logger, LOG_FILE_PATH = setup_logging(str(PACKAGE_ROOT / "logs"))
        _patch_proactor_transport()

        # Model registry → model list, selector index and RAG settings.
        _paths = resolve_paths(Path(BASE_DIR))
        _registry = load_model_registry(settings_config_dir())
        MODELS = [
            {
                "repo_id": m.repo_id,
                "filename": m.filename,
                "desc": m.desc,
                **({"is_vision": m.is_vision} if m.is_vision else {}),
                **({"mmproj_repo_id": m.mmproj_repo_id} if m.mmproj_repo_id else {}),
                **({"mmproj_filename": m.mmproj_filename} if m.mmproj_filename else {}),
            }
            for m in _registry.models
        ]
        FALLBACK_TEXT_MODEL = {
            "repo_id": _registry.fallback.repo_id,
            "filename": _registry.fallback.filename,
            "desc": _registry.fallback.desc,
        }
        MODEL_IDX = _resolve_model_idx(sys.argv, MODELS)
        active_model_desc = str(MODELS[MODEL_IDX]["desc"])
        _rag_settings = RAGSettings.from_env_and_dict({})
        EMBED_MODEL_NAME = _rag_settings.embed_model_name
        # Vision path for the local model. A deployment toggle (ADR 0001: env
        # is read through a settings accessor), not a constant — hard-coding
        # ``False`` silently disabled the vision branch while
        # ``vision_llm_enabled()`` reported the operator's choice.
        USE_VISION_LLM = vision_llm_enabled()

        # At-rest key loading is fail-closed: a malformed environment key, a
        # missing AI_STORAGE_KEY_FILE or an unreadable key file aborts compose
        # before any task/pending/cache adapter can open state it cannot
        # decrypt. It runs before the runtime modules are built because the
        # RAG module seals the cache with the same key.
        storage_cipher = load_storage_cipher()

        # Runtime module graph: instance-based handles owned here and injected
        # into the tool catalog, the services and the routers. The model wiring
        # context is captured on the LLM instance, so nothing has to read a
        # module global later (admin reload included).
        # The prompt builder is a separate collaborator from the LLM
        # runtime handle: the chatbot specialist needs both, and the chat
        # route broke when the two were still one object (ADR 0029).
        # ``unseal`` opens a chat attachment that was sealed at rest by the
        # Qt client with the same platform key (ADR 0031).
        _unseal_attachment = attachment_unsealer(storage_cipher)

        runtime_modules = RuntimeModules(
            llm=LlmModule(
                registry=_registry,
                paths=_paths,
                model_idx=MODEL_IDX,
                use_vision_llm=USE_VISION_LLM,
            ),
            rag=RagModule(cipher=storage_cipher),
            prompt_builder=DefaultPromptBuilder(
                chars_per_token=CHARS_PER_TOKEN,
                unseal=_unseal_attachment,
            ),
        )

        # The composition root owns the single tool catalog, coordinator and
        # agent service. The shared tool gateway, LangGraph checkpointer and
        # A2A transport are injected explicitly, so no module-level service
        # locator remains.
        _tool_registry = create_tool_registry(rag_context=runtime_modules.rag)
        platform = build_container(
            Path(BASE_DIR),
            list(_tool_registry.get_all()),
            platform_executors(_tool_registry),
            runtime_modules.execute_a2a_task,
            storage_cipher=storage_cipher,
        )
        # Third-party plugins register into the same catalogue the agent sees.
        # The returned names form the live ``Specialist.PLUGIN`` row; without
        # this bridge a plugin tool was executable by a direct gateway caller
        # but invisible to routing and refused by ``authorise``.
        plugin_tools = merge_plugin_tools(_tool_registry, platform.plugins)
        # One registry instance owned by this composition root. The supervisor
        # gets a delegate bound to its resolver, and the A2A router gets the
        # same instance — no process-wide lookup that a second composition
        # (tests, embedded hosts) could observe or overwrite.
        from functools import partial

        from ai_assistant.domain.specialists import Specialist
        from ai_assistant.orchestration.supervisor import ALLOWED_TOOLS
        from ai_assistant.orchestration.supervisor import authorise as supervisor_authorise

        a2a_registry = RemoteAgentRegistry()
        # Always inject the bound delegate: with neither A2A nor plugin tools
        # it behaves exactly like the module default, and composition no
        # longer has a second (None) shape for the graph to reason about.
        delegate_fn = _build_supervisor_delegate(
            platform.settings.enable_a2a, a2a_registry, plugin_tools,
        )
        authorise_fn = (
            partial(
                supervisor_authorise,
                allowed_tools={**ALLOWED_TOOLS, Specialist.PLUGIN: plugin_tools},
            )
            if plugin_tools else None
        )
        from ai_assistant.adapters.persistence.memory_store import MemoryStore
        from ai_assistant.agents.short_term_memory import BoundedShortTermMemory
        from ai_assistant.application.memory import MemoryService

        # Short-term memory (ADR 0043): the composition root is the only place
        # that decides whether the buffer exists — ``AGENT_MEMORY_ENABLED=0``
        # leaves the facade with ``None``, i.e. the original "client ships
        # everything" behaviour.
        runtime_settings = load_agent_runtime_settings()
        # Long-term memory (ADR 0043): same switch, same owner. The sealed
        # store is built here (never in the agent layer), pruned of expired
        # rows once at startup, and closed by the lifespan on shutdown.
        memory_store = (
            MemoryStore(_memory_store_path(), cipher=platform.storage_cipher)
            if runtime_settings.memory_enabled else None
        )
        if memory_store is not None:
            memory_store.prune(
                runtime_settings.memory_retention_days,
                max_bytes=runtime_settings.memory_retention_max_mb * 1024 * 1024,
            )
        agent_service = AgentService(
            tool_gateway=platform.gateway,
            llm_runtime=runtime_modules.llm,
            tool_registry=_tool_registry,
            checkpointer=build_checkpointer(platform.storage_cipher),
            a2a_router=A2ARouter(registry=a2a_registry),
            task_coordinator=platform.coordinator,
            pending_actions=PendingActionStore(
                _pending_actions_path(), cipher=platform.storage_cipher,
            ),
            max_concurrent_runs=runtime_settings.max_concurrent_runs,
            short_term_memory=(
                BoundedShortTermMemory(max_messages=runtime_settings.memory_stm_max_messages)
                if runtime_settings.memory_enabled else None
            ),
            memory_service=MemoryService(
                memory_store, top_k=runtime_settings.memory_ltm_top_k,
            ),
            parallel_max=runtime_settings.parallel_max,
            delegate_fn=delegate_fn,
            authorise_delegation_fn=authorise_fn,
        )
        # The MCP adapter owns one server instance bound to the shared gateway;
        # ``mcp_server`` is what the admin routes import (the old
        # ``legacy.mcp_server`` re-export shim was deleted with legacy).
        mcp_server = McpToolServer(platform.gateway)
        # Bound late on the graph: the A2A executor (already handed to the
        # container above) reads ``agent_run_service`` per call, which closes
        # the container ↔ agent-service construction cycle without a
        # module-global service.
        runtime_modules.agent_run_service = AgentRunService(
            capability_scopes=platform.settings.capability_scopes,
            orchestrator=LangGraphAgentOrchestrator(
                execute=agent_service.execute,
                approve=agent_service.approve,
                ui_action_result=agent_service.ui_action_result,
            ),
        )

        # FastAPI application: transport trust policy (ADR 0008) plus every
        # HTTP surface, mounted once, sharing one parsed policy object so
        # loopback/origin/token enforcement cannot drift between adapters.
        app = create_app(platform.settings, lifespan)
        # Lifespan discovery reads the registry from the app state: the app is
        # the object FastAPI hands back to its own lifespan, so no module-level
        # handle is needed.
        app.state.a2a_registry = a2a_registry
        transport_policy = TransportPolicy.from_env(
            allowed_origins=platform.settings.allowed_origins,
            allow_remote_a2a=platform.settings.allow_remote_a2a,
        )
        app.include_router(build_health_router(
            llm_module=runtime_modules.llm,
            rag_module=runtime_modules.rag,
            platform=platform,
            server_start_time=_SERVER_START_TIME,
            embed_model_name=EMBED_MODEL_NAME,
            chars_per_token=CHARS_PER_TOKEN,
            model_idx=MODEL_IDX,
            app=app,
        ))
        app.include_router(build_chat_router(
            llm_module=runtime_modules.llm,
            rag_module=runtime_modules.rag,
            get_chatbot_agent=runtime_modules.chatbot_agent.get,
            llm_n_ctx=LLM_N_CTX,
            logger=logger,
            policy=transport_policy,
            unseal=_unseal_attachment,
        ))
        app.include_router(build_admin_router(
            llm_module=runtime_modules.llm,
            rag_module=runtime_modules.rag,
            action_manifest_module=action_manifest,
            reset_agent_state_fn=agent_service.reset_state,
            models=MODELS,
            model_idx=MODEL_IDX,
            refresh_agent_routes_fn=_refresh_agent_routes,
            rebuild_chatbot_agent_fn=runtime_modules.chatbot_agent.rebuild,
            logger=logger,
            request_shutdown=runtime_modules.request_shutdown,
            policy=transport_policy,
        ))
        app.include_router(build_agent_router(agent_service, policy=transport_policy))
        _register_optional_protocols(app, platform, mcp_server, transport_policy)


def __getattr__(name: str) -> Any:
    """Compose on first access of a composed name (PEP 562).

    Only fires for names missing from the module dict: a composed name before
    composition (build the graph, then return it), or a genuinely unknown
    attribute (``AttributeError``, as if there were no hook at all).
    """
    if name in _COMPOSED_NAMES:
        compose()
        return globals()[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# ── Runtime Bootstrap ─────────────────────────────────────────────────────────
def bootstrap_runtime() -> None:
    """Initialize RAG and LLM once before Uvicorn accepts requests."""
    compose()
    from ai_assistant.config import safe_relpath

    logger.info(
        "Starting runtime | model index=%d | log=%s",
        MODEL_IDX, safe_relpath(LOG_FILE_PATH, BASE_DIR),
    )
    runtime_modules.rag.initialize_rag(
        enable_reranker=not MODELS[MODEL_IDX].get("is_vision", False),
    )
    # Startup performs a model swap too: the owner now refuses ``load_model``
    # outside the drain window, so bootstrap takes the same gate an admin
    # reload does instead of relying on there being no leases yet.
    with runtime_modules.llm.model_reload():
        runtime_modules.llm.load_model()
    if runtime_modules.llm.is_vision_model:
        runtime_modules.rag.release_embedding_for_vision()


def _stop_server_when_requested(server: Any, shutdown_requested: threading.Event) -> None:
    """Set ``uvicorn.Server.should_exit`` once the admin route asks to stop.

    ``/admin/shutdown`` used to call ``os._exit``, which skips the lifespan
    ``finally`` (task-store close, checkpoints, flush). The route now only
    sets the event; this watcher performs the graceful stop from the serving
    process. Extracted so the transition is unit-tested.
    """
    shutdown_requested.wait()
    server.should_exit = True


def main() -> None:
    """Process entry point: warm the runtime, then serve until terminated.

    Extracted from the ``__main__`` guard so the two failure paths are unit
    tests rather than coverage exclusions: a runtime that will not load exits
    1 (the Qt host reads the log), and a serve that dies is re-raised after
    being logged.
    """
    import uvicorn

    try:
        compose()
        bootstrap_runtime()
    except Exception:
        logger.exception("AI server startup failed")
        sys.exit(1)
    server = uvicorn.Server(uvicorn.Config(
        app, host=agent_bind_host(), port=agent_bind_port(),
        log_config=None, access_log=False,
    ))
    threading.Thread(
        target=_stop_server_when_requested,
        args=(server, runtime_modules.shutdown_requested),
        name="ai-server-shutdown", daemon=True,
    ).start()
    try:
        server.run()
    except BaseException:
        logger.exception("Uvicorn terminated during server startup")
        raise


if __name__ == "__main__":  # one-line interpreter guard (coverage exclude_also)
    main()
