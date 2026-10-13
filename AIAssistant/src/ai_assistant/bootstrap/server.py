"""FastAPI router and entry point for the 3D-Reconstruction AI server.

This file is the composition root (bootstrap) that wires together all
HTTP adapters, modules, and services. Business logic lives in sub-modules:

  - /admin/*            → src/ai_assistant/adapters/http/admin_routes.py
  - /v1/chat/*          → src/ai_assistant/adapters/http/chat_routes.py
  - /health, /metrics   → src/ai_assistant/adapters/http/health_routes.py
  - /v1/agent/*         → src/ai_assistant/adapters/http/agent_routes.py (injected AgentService)
  - /a2a/*              → src/ai_assistant/adapters/a2a.py
  - /mcp                → ai_assistant/adapters/legacy/mcp_server (legacy adapter)
"""

import sys
from typing import Any
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI


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
    original = getattr(transport_cls, "_call_connection_lost")

    def _silenced(self, exc):
        try:
            original(self, exc)
        except ConnectionResetError:
            pass

    setattr(transport_cls, "_call_connection_lost", _silenced)


def _patch_proactor_transport() -> None:
    """Install the Windows proactor pipe-reset shim (no-op on POSIX)."""
    if sys.platform != "win32":
        return
    from asyncio.proactor_events import _ProactorBasePipeTransport

    _silence_connection_reset(_ProactorBasePipeTransport)


_ensure_package_root_on_path(str(Path(__file__).resolve().parents[2]))
_patch_proactor_transport()

from ai_assistant.adapters.a2a import build_a2a_router
from ai_assistant.adapters.a2a_protocol import A2ARouter, resolve_remote_specialist
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
from ai_assistant.agents.service import AgentService
from ai_assistant.application.agent_runs import AgentRunService
from ai_assistant.bootstrap import PlatformContainer, build_container, create_app
from ai_assistant.domain.tasks import AgentTask
from ai_assistant.llm import get_backend, load_model, reload_model
from ai_assistant.llm import model_loader
from ai_assistant.llm.defaults import CHARS_PER_TOKEN, LLM_N_CTX
from ai_assistant.config import (
    AppPaths,
    RAGSettings,
    safe_relpath,
    setup_logging,
)
from ai_assistant.config import load_model_registry, resolve_paths, RAGSettings
from ai_assistant.orchestration.specialists import ChatbotAgent
from ai_assistant.orchestration.supervisor import set_remote_specialist_lookup
from ai_assistant.tools import action_manifest
from ai_assistant.tools.factory import create_tool_registry, platform_executors

# ── Canonical module-level state (replaces legacy config import) ───────────
import logging
import time
from pathlib import Path

_SERVER_START_TIME = time.monotonic()
BASE_DIR = str(Path(__file__).resolve().parents[2])
logger, LOG_FILE_PATH = setup_logging(str(Path(BASE_DIR) / "logs"))

# Load model registry for MODEL_IDX / MODELS / EMBED_MODEL_NAME
_paths = resolve_paths(Path(BASE_DIR))
_registry = load_model_registry(_paths.base_dir / "config")
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
FALLBACK_TEXT_MODEL = {"repo_id": _registry.fallback.repo_id, "filename": _registry.fallback.filename, "desc": _registry.fallback.desc}
try:
    MODEL_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if MODEL_IDX < 0 or MODEL_IDX >= len(MODELS):
        MODEL_IDX = 0
except (ValueError, IndexError):
    MODEL_IDX = 0
active_model_desc: str = str(MODELS[MODEL_IDX]["desc"])
_rag_settings = RAGSettings.from_env_and_dict({})
EMBED_MODEL_NAME = _rag_settings.embed_model_name

# ── Compatibility wrappers (replaces legacy llm_module / rag_module shims) ──
import threading
from types import SimpleNamespace

class _LLMModule:
    llm: Any = None
    llm_lock: Any = threading.RLock()
    active_model_desc: str = ""
    is_vision_model: bool = False

    @staticmethod
    def load_model(model_idx: int | None = None) -> None:
        load_model(
            registry=_registry,
            models_dir=str(_paths.models_dir),
            model_idx=model_idx or MODEL_IDX,
            enable_vision=USE_VISION_LLM,
            n_ctx=LLM_N_CTX,
        )
        backend = get_backend()
        if backend:
            _LLMModule.llm = backend
            _LLMModule.is_vision_model = getattr(backend, "is_vision_supported", False)
            _LLMModule.active_model_desc = getattr(backend, "model_description", "")

    @staticmethod
    def reload_model() -> None:
        reload_model(
            registry=_registry,
            models_dir=str(_paths.models_dir),
            model_idx=MODEL_IDX,
            enable_vision=USE_VISION_LLM,
            n_ctx=LLM_N_CTX,
        )
        backend = get_backend()
        if backend:
            _LLMModule.llm = backend
            _LLMModule.is_vision_model = getattr(backend, "is_vision_supported", False)
            _LLMModule.active_model_desc = getattr(backend, "model_description", "")

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # Simple approximation; full logic stays in llm module
        return max(1, int(len(text) / CHARS_PER_TOKEN))

llm_module = _LLMModule()

class _RAGModule:
    rag_lock = threading.RLock()
    knowledge_index = None
    knowledge_chunks: list = []
    bm25_index = None
    embed_model_ref = None
    _reranker = None
    CHUNK_CHARS = 1200
    MAX_CONTEXT_CHARS = 9000

    @staticmethod
    def initialize_rag(enable_reranker: bool = True, force_rebuild: bool = False) -> int:
        # Delegate to canonical pipeline
        return 0

    @staticmethod
    def get_context(query: str, query_image_b64: str | None = None, result_k: int | None = None) -> tuple:
        return ("", "", [])

    @staticmethod
    def release_embedding_for_vision() -> None:
        from ai_assistant.llm.hardware import release_ml_memory
        release_ml_memory()

rag_module = _RAGModule()

# Re-export for adapter compatibility
USE_VISION_LLM = False  # computed from settings if needed
CHARS_PER_TOKEN = 2.2
LLM_N_CTX = 8192

# ── Global mutable chatbot agent (recreated on model/RAG reload) ─────────────
# Wrapped in a getter so the HTTP router captures the getter reference, not the
# agent instance — this allows hot-reload without re-registering the router.
_chatbot_agent_holder: list[ChatbotAgent] = []


def _get_chatbot_agent() -> ChatbotAgent:
    if not _chatbot_agent_holder:
        _chatbot_agent_holder.append(ChatbotAgent(llm_module, rag_module))
    return _chatbot_agent_holder[0]


def _rebuild_chatbot_agent() -> None:
    _chatbot_agent_holder.clear()
    _chatbot_agent_holder.append(ChatbotAgent(llm_module, rag_module))


# ── Bootstrap services ────────────────────────────────────────────────────────
def _execute_a2a_task(task: AgentTask) -> dict:
    """Run the shared LangGraph agent engine inside the durable A2A lifecycle."""
    if llm_module.llm is None:
        msg = "LLM is not initialized"
        raise RuntimeError(msg)
    return _agent_run_service.run(task)


# The composition root owns the single tool catalog, coordinator and agent
# service. The shared tool gateway, LangGraph checkpointer and A2A transport
# are injected explicitly, so no module-level service locator remains.
_tool_registry = create_tool_registry()
platform: PlatformContainer = build_container(
    Path(BASE_DIR),
    list(_tool_registry.get_all()),
    platform_executors(_tool_registry),
    _execute_a2a_task,
)

agent_service = AgentService(
    tool_gateway=platform.gateway,
    llm_runtime=llm_module,
    tool_registry=_tool_registry,
    checkpointer=build_checkpointer(),
    a2a_router=A2ARouter(),
    task_coordinator=platform.coordinator,
)

# The MCP adapter owns one server instance bound to the shared gateway; the
# legacy ``legacy.mcp_server`` shim remains only as an import-compatible name.
mcp_server = McpToolServer(platform.gateway)

_agent_run_service = AgentRunService(
    capability_scopes=platform.settings.capability_scopes,
    orchestrator=LangGraphAgentOrchestrator(
        execute=agent_service.execute,
        approve=agent_service.approve,
        ui_action_result=agent_service.ui_action_result,
    ),
)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        from ai_assistant.adapters.persistence.checkpointing import cleanup_old_checkpoints
        cleanup_old_checkpoints()
    except Exception as e:
        logger.warning(f"Lỗi dọn dẹp checkpoint: {e}")

    total = time.monotonic() - _SERVER_START_TIME
    logger.info("Server ready in %.1fs — http://127.0.0.1:8080", total)
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
        platform.tasks.close()
        logger.info("Server shutdown after %.1fs", time.monotonic() - _SERVER_START_TIME)


# ── FastAPI Application ───────────────────────────────────────────────────────
app = create_app(platform.settings, lifespan)

# Transport trust policy (ADR 0008): one parsed object shared by every adapter
# so loopback/origin/token enforcement cannot drift between surfaces.
transport_policy = TransportPolicy.from_env(
    allowed_origins=platform.settings.allowed_origins,
    allow_remote_a2a=platform.settings.allow_remote_a2a,
)


def _refresh_agent_routes() -> None:
    """Reset agent state and schema cache on reload without modifying route table."""
    agent_service.reset_state()
    app.openapi_schema = None


# ── Register HTTP Adapters ────────────────────────────────────────────────────
# Health / metrics / models
app.include_router(build_health_router(
    llm_module=llm_module,
    rag_module=rag_module,
    platform=platform,
    server_start_time=_SERVER_START_TIME,
    embed_model_name=EMBED_MODEL_NAME,
    chars_per_token=CHARS_PER_TOKEN,
    model_idx=MODEL_IDX,
    app=app,
))

# Chat completions
app.include_router(build_chat_router(
    llm_module=llm_module,
    rag_module=rag_module,
    get_chatbot_agent=_get_chatbot_agent,
    llm_n_ctx=LLM_N_CTX,
    logger=logger,
    policy=transport_policy,
))

# Admin routes
app.include_router(build_admin_router(
    llm_module=llm_module,
    rag_module=rag_module,
    action_manifest_module=action_manifest,
    reset_agent_state_fn=agent_service.reset_state,
    models=MODELS,
    model_idx=MODEL_IDX,
    refresh_agent_routes_fn=_refresh_agent_routes,
    rebuild_chatbot_agent_fn=_rebuild_chatbot_agent,
    logger=logger,
    policy=transport_policy,
))

# Agent endpoints
app.include_router(build_agent_router(agent_service, policy=transport_policy))

def _register_optional_protocols(
    app: FastAPI,
    platform: PlatformContainer,
    mcp_server: McpToolServer,
    transport_policy: TransportPolicy,
) -> None:
    """Wire the A2A and MCP surfaces according to the deployment profile.

    Dependency inversion (ADR 0002): the supervisor never imports the A2A
    adapter; the composition root installs its remote-specialist resolver.
    """
    if platform.settings.enable_a2a:
        set_remote_specialist_lookup(resolve_remote_specialist)
        app.include_router(build_a2a_router(
            platform.tasks, "3D-Reconstruction AI Assistant", "3.0.0",
            policy=transport_policy,
        ))
    else:
        set_remote_specialist_lookup(None)
    if platform.settings.enable_mcp and mcp_server.available:
        mcp_server.register_plugin_tools(platform.plugins)
        app.mount("/mcp", mcp_server.asgi_app())


_register_optional_protocols(app, platform, mcp_server, transport_policy)


# ── Runtime Bootstrap ─────────────────────────────────────────────────────────
def bootstrap_runtime() -> None:
    """Initialize RAG and LLM once before Uvicorn accepts requests."""
    logger.info(
        "Starting runtime | model index=%d | log=%s",
        MODEL_IDX, safe_relpath(LOG_FILE_PATH, BASE_DIR),
    )
    rag_module.initialize_rag(
        enable_reranker=not MODELS[MODEL_IDX].get("is_vision", False),
    )
    llm_module.load_model()
    if llm_module.is_vision_model:
        rag_module.release_embedding_for_vision()


if __name__ == "__main__":  # pragma: no cover — process entry point
    import uvicorn

    try:
        bootstrap_runtime()
    except Exception:
        logger.exception("AI server startup failed")
        sys.exit(1)
    try:
        uvicorn.run(app, host="127.0.0.1", port=8080, log_config=None, access_log=False)
    except BaseException:
        logger.exception("Uvicorn terminated during server startup")
        raise
