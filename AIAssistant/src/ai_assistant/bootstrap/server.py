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
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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

from fastapi import FastAPI

from ai_assistant.settings import apply_startup_environment_defaults, load_local_dotenv

# The composition root owns the process environment (ADR 0001): merge ``.env``
# and apply the library-quieting defaults *before* the imports below read the
# environment — the A2A banner and the transport policy both do so at import
# time. Without this step the file was parsed by tests only and the running
# server never saw the tokens an operator put there (LANGSMITH_API_KEY, …).
load_local_dotenv(Path(__file__).resolve().parents[3])  # project root (``.env.example``'s directory)
apply_startup_environment_defaults()


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

# ── Canonical module-level state (replaces legacy config import) ───────────
from pathlib import Path

from ai_assistant.adapters.a2a import build_a2a_router
from ai_assistant.adapters.a2a_protocol import A2ARouter, resolve_remote_specialist
from ai_assistant.adapters.http import (
    build_admin_router,
    build_agent_router,
    build_chat_router,
    build_health_router,
)
from ai_assistant.adapters.http.security import (
    TransportPolicy,
    mcp_exposed_without_token,
    mcp_guarded_app,
)
from ai_assistant.adapters.mcp import McpToolServer
from ai_assistant.adapters.orchestration import LangGraphAgentOrchestrator
from ai_assistant.adapters.persistence.checkpointing import build_checkpointer
from ai_assistant.agents.service import AgentService
from ai_assistant.application.agent_runs import AgentRunService
from ai_assistant.bootstrap import PlatformContainer, build_container, create_app
from ai_assistant.config import (
    RAGSettings,
    load_model_registry,
    resolve_paths,
    safe_relpath,
    setup_logging,
)
from ai_assistant.domain.tasks import AgentTask
from ai_assistant.llm import get_backend, load_model, reload_model
from ai_assistant.llm.defaults import CHARS_PER_TOKEN, LLM_N_CTX
from ai_assistant.orchestration.specialists import ChatbotAgent
from ai_assistant.orchestration.supervisor import set_remote_specialist_lookup
from ai_assistant.settings import (
    agent_bind_host,
    agent_bind_port,
    settings_config_dir,
    vision_llm_enabled,
)
from ai_assistant.tools import action_manifest
from ai_assistant.tools.factory import create_tool_registry, platform_executors

_SERVER_START_TIME = time.monotonic()
BASE_DIR = str(Path(__file__).resolve().parents[2])
# ``BASE_DIR`` (…/AIAssistant/src) is the data/model root the legacy resolvers
# expect. The *tracked* deployment assets — ``config/*.toml`` and the ``logs/``
# directory the Qt server manager tails — live one level up in the package
# root; reading them from ``src/`` silently found nothing.
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
logger, LOG_FILE_PATH = setup_logging(str(PACKAGE_ROOT / "logs"))

# Load model registry for MODEL_IDX / MODELS / EMBED_MODEL_NAME
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
FALLBACK_TEXT_MODEL = {"repo_id": _registry.fallback.repo_id, "filename": _registry.fallback.filename, "desc": _registry.fallback.desc}
def _resolve_model_idx(argv: list[str], models: list[dict]) -> int:
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


MODEL_IDX = _resolve_model_idx(sys.argv, MODELS)
active_model_desc: str = str(MODELS[MODEL_IDX]["desc"])
_rag_settings = RAGSettings.from_env_and_dict({})
EMBED_MODEL_NAME = _rag_settings.embed_model_name

# ── Compatibility wrappers (replaces legacy llm_module / rag_module shims) ──
import threading


class _LLMModule:
    llm: Any = None
    llm_lock: Any = threading.RLock()
    active_model_desc: str = ""
    is_vision_model: bool = False

    @staticmethod
    def _adopt_backend(backend: Any) -> bool:
        """Bind a freshly loaded backend to the module state.

        Returns ``False`` (and logs) when the backend is missing, so a failed
        load is visible instead of leaving the previous model advertised as
        current with neither branch excluded from coverage.
        """
        if backend is None:
            logger.error("No inference backend after load; keeping previous module state")
            return False
        _LLMModule.llm = backend
        _LLMModule.is_vision_model = getattr(backend, "is_vision_supported", False)
        _LLMModule.active_model_desc = getattr(backend, "model_description", "")
        return True

    @staticmethod
    def load_model(model_idx: int | None = None) -> None:
        load_model(
            registry=_registry,
            models_dir=str(_paths.models_dir),
            model_idx=model_idx or MODEL_IDX,
            enable_vision=USE_VISION_LLM,
            n_ctx=LLM_N_CTX,
        )
        _LLMModule._adopt_backend(get_backend())

    @staticmethod
    def reload_model() -> None:
        reload_model(
            registry=_registry,
            models_dir=str(_paths.models_dir),
            model_idx=MODEL_IDX,
            enable_vision=USE_VISION_LLM,
            n_ctx=LLM_N_CTX,
        )
        _LLMModule._adopt_backend(get_backend())

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
    embed_model_ref: Any = None
    _reranker = None
    vector_retriever = None  # legacy compatibility
    bm25_retriever = None     # legacy compatibility (tests/unit/test_legacy_rag_cache.py)
    _pipeline: Any = None     # legacy compatibility
    CHUNK_CHARS = 1200
    MAX_CONTEXT_CHARS = 9000
    ENABLE_RAG = True  # legacy compatibility (tests/unit/test_legacy_rag_cache.py)

    @staticmethod
    def _build_retrievers(index: Any = None, bm25: Any = None, chunks: list | None = None,
                          encoder: Any = None, settings: Any = None):
        """Build the dense/sparse retriever pair for the current index state."""
        from ai_assistant.rag.retrievers import build_retrievers
        if index is None or chunks is None or settings is None:
            return (None, None)
        return build_retrievers(index, bm25, chunks, encoder, settings)

    @staticmethod
    def _reset_index_state() -> None:
        """Drop every retrieval handle so an empty state is observable everywhere."""
        rag_module.knowledge_index = None
        rag_module.knowledge_chunks = []
        rag_module.bm25_index = None
        rag_module.vector_retriever = None
        rag_module.bm25_retriever = None
        rag_module._pipeline = None

    @staticmethod
    def initialize_rag(enable_reranker: bool = True, force_rebuild: bool = False) -> int:
        """Load (or rebuild) the RAG index and wire the retrieval pipeline.

        ADR 0015 — the returned number is the count of chunks actually wired:
        cache hit → load; cache miss → scan + build + persist; no documents or
        any error → observable empty state (``/health`` reports
        ``rag_chunks == 0`` and degrades) with the reason logged with traceback.
        """
        try:
            count = _RAGModule._wire_index(enable_reranker, force_rebuild)
        except Exception:
            # Fail safe *and* visible: the traceback is logged, the state is
            # cleared and /health degrades instead of pretending success.
            logger.exception("RAG initialization failed; retrieval stays disabled")
            _RAGModule._reset_index_state()
            return 0
        return count

    @staticmethod
    def _wire_index(enable_reranker: bool, force_rebuild: bool) -> int:
        """Build/load the index and publish it on ``rag_module`` (may raise)."""
        from ai_assistant.config.paths import get_paths
        from ai_assistant.config.rag_config import RAGSettings
        from ai_assistant.rag.embeddings import lazy_embedding_encoder
        from ai_assistant.rag.index import (
            is_cache_valid,
            load_cache,
            load_or_build_index,
            scan_documents,
        )
        from ai_assistant.rag.loaders import DocumentLoaderRegistry
        from ai_assistant.rag.retrieval import RetrievalPipeline

        settings = RAGSettings.from_env_and_dict({})
        if not settings.enabled:
            logger.info("RAG disabled by AI_ENABLE_RAG")
            _RAGModule._reset_index_state()
            return 0
        paths = get_paths()
        encoder = lazy_embedding_encoder(settings)

        index = chunks = bm25 = None
        if not force_rebuild and is_cache_valid(paths, settings):
            index, chunks, bm25 = load_cache(paths, settings)

        if index is None:
            registry = DocumentLoaderRegistry.create_default(
                settings.chunk_chars, settings.overlap_chars)
            scanned = scan_documents(paths, registry)
            if not scanned:
                logger.warning("RAG: no documents found under %s — index not built",
                               paths.project_root)
                _RAGModule._reset_index_state()
                return 0
            # Cache is missing/stale: force the build path so the fs hash is
            # computed once (scan already happened above).
            index, chunks, bm25 = load_or_build_index(
                scanned, encoder, paths, settings, force_rebuild=True)

        if index is None or not chunks:
            logger.warning("RAG: index unavailable after load/build")
            _RAGModule._reset_index_state()
            return 0

        vector_retriever, bm25_retriever = _RAGModule._build_retrievers(
            index, bm25, chunks, encoder, settings)
        pipeline = RetrievalPipeline(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            # The cross-encoder is an optional quality boost; it stays None
            # until a loader exists (pipeline treats None as a no-op).
            reranker=getattr(rag_module, "_reranker", None),
            knowledge_chunks=chunks,
            max_context_chars=settings.max_context_chars,
            similarity_threshold=settings.similarity_threshold,
            use_reranker=bool(enable_reranker and settings.use_reranker),
            reranker_top_k=settings.reranker_top_k,
        )
        with rag_module.rag_lock:
            rag_module.knowledge_index = index
            rag_module.knowledge_chunks = list(chunks)
            rag_module.bm25_index = bm25
            rag_module.vector_retriever = vector_retriever
            rag_module.bm25_retriever = bm25_retriever
            rag_module._pipeline = pipeline
            rag_module.embed_model_ref = encoder
        count = len(chunks)
        logger.info("RAG ready: %d chunks (vector=%s bm25=%s)",
                    count, vector_retriever is not None, bm25_retriever is not None)
        return count

    @staticmethod
    def get_context(query: str, query_image_b64: str | None = None, result_k: int | None = None) -> tuple:
        pipeline = getattr(rag_module, "_pipeline", None)
        if pipeline is None:
            return ("", "", [])
        try:
            doc_ctx, code_ctx, image_chunks = pipeline.get_context(query, query_image_b64, result_k)
        except Exception:
            logger.exception("RAG retrieval failed")
            return ("", "", [])
        return doc_ctx, code_ctx, image_chunks

    @staticmethod
    def _release_ml_memory() -> None:
        # Protocol compliance for RAGModuleLike; delegates to vision release
        _RAGModule.release_embedding_for_vision()

    @staticmethod
    def release_embedding_for_vision() -> None:
        from ai_assistant.llm.hardware import release_ml_memory
        release_ml_memory()

rag_module = _RAGModule()

# Vision path for the local model. It is a deployment toggle (ADR 0001: env is
# read through a settings accessor), not a constant — hard-coding ``False``
# here silently disabled the vision branch while ``vision_llm_enabled()``
# reported the operator's choice.
USE_VISION_LLM = vision_llm_enabled()
# CHARS_PER_TOKEN / LLM_N_CTX re-exported via import above; do not redefine.

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
_tool_registry = create_tool_registry(rag_context=rag_module)
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
# ``mcp_server`` name below is what the admin routes import (the old
# ``legacy.mcp_server`` re-export shim was deleted with the legacy package).
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


if __name__ == "__main__":  # process entry point (excluded by coverage's default __main__ rule)
    import uvicorn

    try:
        bootstrap_runtime()
    except Exception:
        logger.exception("AI server startup failed")
        sys.exit(1)
    try:
        uvicorn.run(app, host=agent_bind_host(), port=agent_bind_port(),
                    log_config=None, access_log=False)
    except BaseException:
        logger.exception("Uvicorn terminated during server startup")
        raise
