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

import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

_PACKAGE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport

        _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

        def _silenced_call_connection_lost(self, exc):  # noqa: ANN001, ANN202
            try:
                _orig_call_connection_lost(self, exc)
            except ConnectionResetError:
                pass

        _ProactorBasePipeTransport._call_connection_lost = _silenced_call_connection_lost
    except Exception:
        pass

from ai_assistant.adapters.a2a import build_a2a_router
from ai_assistant.adapters.http import (
    build_admin_router,
    build_agent_router,
    build_chat_router,
    build_health_router,
)
from ai_assistant.adapters.http.security import TransportPolicy
from ai_assistant.adapters.orchestration import LangGraphAgentOrchestrator
from ai_assistant.agents.service import AGENT_TOOLS, AgentService, platform_executors
from ai_assistant.application.agent_runs import AgentRunService
from ai_assistant.bootstrap import PlatformContainer, build_container, create_app
from ai_assistant.domain.tasks import AgentTask
from ai_assistant.legacy import llm_module, mcp_server, rag_module
from ai_assistant.legacy.config import (
    _SERVER_START_TIME,
    BASE_DIR,
    CHARS_PER_TOKEN,
    EMBED_MODEL_NAME,
    LLM_N_CTX,
    LOG_FILE_PATH,
    MODEL_IDX,
    MODELS,
    _safe_relpath,
    logger,
)
from ai_assistant.orchestration.specialists import ChatbotAgent
from ai_assistant.tools import action_manifest

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
        raise RuntimeError("LLM is not initialized")
    return _agent_run_service.run(task)


platform: PlatformContainer = build_container(
    BASE_DIR,
    AGENT_TOOLS,
    platform_executors(),
    _execute_a2a_task,
)

# The composition root owns the single agent service. Its shared tool gateway
# is injected from the DI container, so no module-level service locator remains.
agent_service = AgentService(tool_gateway=platform.gateway, llm_runtime=llm_module)

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
        if platform.settings.enable_mcp and mcp_server.MCP_AVAILABLE:
            async with mcp_server.lifespan():
                yield
        elif platform.settings.enable_mcp:
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

# Optional protocol adapters
if platform.settings.enable_a2a:
    app.include_router(build_a2a_router(
        platform.tasks, "3D-Reconstruction AI Assistant", "3.0.0",
        policy=transport_policy,
    ))
if platform.settings.enable_mcp and mcp_server.MCP_AVAILABLE:
    mcp_server.configure_tool_gateway(platform.gateway)
    app.mount("/mcp", mcp_server.asgi_app())


# ── Runtime Bootstrap ─────────────────────────────────────────────────────────
def bootstrap_runtime() -> None:
    """Initialize RAG and LLM once before Uvicorn accepts requests."""
    logger.info(
        "Starting runtime | model index=%d | log=%s",
        MODEL_IDX, _safe_relpath(LOG_FILE_PATH, BASE_DIR),
    )
    rag_module.initialize_rag(
        enable_reranker=not MODELS[MODEL_IDX].get("is_vision", False),
    )
    llm_module.load_model()
    if llm_module.is_vision_model:
        rag_module.release_embedding_for_vision()


if __name__ == "__main__":
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
