"""Instance-based runtime module handles for the composition root.

These classes own the mutable runtime state that the HTTP routers and the
agent services reach into: the loaded LLM backend, the RAG retrieval
pipeline, and the hot-reloadable chatbot agent. They are plain data holders
with behaviour — no class-level mutable state, no module globals — so the
composition root can build one ``RuntimeModules`` graph, inject it where
needed, and tests can construct isolated instances directly.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_assistant.application.agent_runs import AgentRunService
    from ai_assistant.domain.tasks import AgentTask
    from ai_assistant.llm.contracts import PromptBuilder
    from ai_assistant.orchestration.specialists import ChatbotAgent
    from ai_assistant.ports import FieldCipher, LLMBackend

logger = logging.getLogger(__name__)


def attachment_unsealer(cipher: "FieldCipher") -> Callable[[bytes], bytes]:
    """Return a callable that opens a chat attachment sealed at rest.

    The Qt client seals the attached file with the platform at-rest key
    (ADR 0031), and the AI service has to open it before the model can see
    the image. The key stays in the cipher that the composition root owns;
    this only hands out the one operation the readers need.
    """
    from ai_assistant.ports import CHAT_ATTACHMENT_CONTEXT

    def unseal(raw: bytes) -> bytes:
        return cipher.open_bytes(raw, context=CHAT_ATTACHMENT_CONTEXT)

    return unseal


class LlmModule:
    """Live LLM runtime handle: the loaded backend and its lock.

    Instance-based replacement for the former ``_LLMModule`` class-level
    state. The composition root creates one instance (with the model-registry
    wiring it needs), injects it into the routers and the agent service, and
    calls ``load_model()`` at startup.
    """

    def __init__(self, *, registry: Any, paths: Any, model_idx: int,
                 use_vision_llm: bool) -> None:
        self.llm: Any = None
        self.llm_lock: Any = threading.RLock()
        self.active_model_desc: str = ""
        self.is_vision_model: bool = False
        self._registry = registry
        self._paths = paths
        self._model_idx = model_idx
        self._use_vision_llm = use_vision_llm
        # Serializes load/release against each other (download + construct can
        # take seconds). The lock is distinct from ``llm_lock`` so readers do
        # not block for the whole load; the swap itself is one short critical
        # section. This instance is the ONLY owner of the backend — the old
        # module-level ``_global_backend`` was a second reference, so dropping
        # ``self.llm`` left the weights alive and VRAM was never reclaimed.
        self._load_lock = threading.RLock()
        # Model-reload drain: a run holds a lease for its whole lifetime, and a
        # reload waits for every lease before swapping the backend. New leases
        # wait while a reload is in progress, so a run sees one model end to end.
        self._model_condition = threading.Condition()
        self._active_leases = 0
        self._reloading = False

    @contextmanager
    def model_lease(self) -> Iterator[LLMBackend | None]:
        """Pin the current backend for one run; reloads wait for the lease."""
        with self._model_condition:
            while self._reloading:
                self._model_condition.wait()
            self._active_leases += 1
        try:
            yield self.llm
        finally:
            with self._model_condition:
                self._active_leases -= 1
                self._model_condition.notify_all()

    @contextmanager
    def model_reload(self) -> Iterator[None]:
        """Wait for active runs, block new leases, then swap the model."""
        with self._model_condition:
            self._reloading = True
            while self._active_leases:
                self._model_condition.wait()
        try:
            yield
        finally:
            with self._model_condition:
                self._reloading = False
                self._model_condition.notify_all()

    def _require_reload_window(self) -> None:
        """Refuse a backend swap outside :meth:`model_reload`.

        The drain invariant belongs to the owner, not to the route that
        happens to call it: ``release()`` used to be public and could free
        the model while a run held a lease. Every mutation now has to prove
        it is inside the drain window, so a future caller cannot bypass the
        guarantee by calling the primitive directly.
        """
        with self._model_condition:
            if not self._reloading:
                msg = "model load/release must run inside model_reload()"
                raise RuntimeError(msg)

    def _adopt_backend(self, backend: Any) -> bool:
        """Bind a freshly loaded backend to this instance.

        Returns ``False`` (and logs) when the backend is missing, so a failed
        load is visible instead of leaving the previous model advertised as
        current with neither branch excluded from coverage.
        """
        if backend is None:
            logger.error("No inference backend after load; keeping previous module state")
            return False
        self.llm = backend
        self.is_vision_model = getattr(backend, "is_vision_supported", False)
        self.active_model_desc = getattr(backend, "model_description", "")
        return True

    def _clear_backend_locked(self) -> bool:
        """Drop the backend and every field that describes it, as one unit.

        Returns whether a model was actually loaded. The metadata is cleared
        in the same critical section as the reference, so ``/health`` cannot
        report ``llm_loaded: false`` while ``/v1/models`` still advertises the
        previous description (ADR 0041: one owner, one transition).
        """
        had_backend = self.llm is not None
        self.llm = None
        self.active_model_desc = ""
        self.is_vision_model = False
        return had_backend

    def release(self) -> bool:
        """Drop the only reference to the backend so its memory can be freed.

        Returns whether a model was loaded. ``release_ml_memory()`` runs only
        after the last reference is gone — calling it while a stray owner
        still holds the object is what made the old ``/admin/release-vram``
        return ``ok`` without releasing anything.
        """
        from ai_assistant.llm.hardware import release_ml_memory

        self._require_reload_window()
        with self._load_lock:
            with self.llm_lock:
                had_backend = self._clear_backend_locked()
            if had_backend:
                release_ml_memory()
        return had_backend

    def load_model(self, model_idx: int | None = None) -> None:
        from ai_assistant.llm import load_backend
        from ai_assistant.llm.defaults import LLM_N_CTX
        from ai_assistant.llm.hardware import release_ml_memory

        self._require_reload_window()
        with self._load_lock:
            # Free the previous weights before the new download/constructor:
            # two models must never be resident at once on a GPU.
            with self.llm_lock:
                had_backend = self._clear_backend_locked()
            if had_backend:
                release_ml_memory()
            backend = load_backend(
                registry=self._registry,
                models_dir=str(self._paths.models_dir),
                # ``0`` is a valid model index, so the fallback must be on
                # ``None`` — ``model_idx or self._model_idx`` silently ignored it.
                model_idx=self._model_idx if model_idx is None else model_idx,
                enable_vision=self._use_vision_llm,
                n_ctx=LLM_N_CTX,
            )
            with self.llm_lock:
                self._adopt_backend(backend)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        from ai_assistant.llm.defaults import CHARS_PER_TOKEN

        # Simple approximation; full logic stays in llm module
        return max(1, int(len(text) / CHARS_PER_TOKEN))


class RagModule:
    """Live RAG runtime handle: the retrieval pipeline and its index state.

    Instance-based replacement for the former ``_RAGModule`` class-level
    state. All index state lives on the instance, so the composition root
    owns exactly one and tests can build isolated doubles.

    The at-rest cipher is injected: the cache on disk holds the user's
    document corpus, so it is sealed with the same key as the task store
    (ADR 0027/0028). There is no "no cipher" mode — a cache that cannot be
    sealed is not written.
    """

    def __init__(self, *, cipher: FieldCipher) -> None:
        self.rag_lock = threading.RLock()
        self.knowledge_index = None
        self.knowledge_chunks: list[Any] = []
        self.bm25_index = None
        self.embed_model_ref: Any = None
        self._reranker = None
        self._pipeline: Any = None     # the live retrieval handle (/health and admin reset it)
        self._cipher = cipher
        self.CHUNK_CHARS = 1200
        self.MAX_CONTEXT_CHARS = 9000

    def _build_retrievers(self, index: Any = None, bm25: Any = None,
                          chunks: list[dict[str, Any]] | None = None,
                          encoder: Any = None, settings: Any = None) -> tuple[Any | None, Any | None]:
        """Build the dense/sparse retriever pair for the current index state."""
        from ai_assistant.rag.retrievers import build_retrievers
        if index is None or chunks is None or settings is None:
            return (None, None)
        return build_retrievers(index, bm25, chunks, encoder, settings)

    def reset_index_state(self) -> None:
        """Drop every retrieval handle so an empty state is observable everywhere.

        ADR 0032: this is the *only* owner of the "the RAG state is empty"
        transition. It used to exist twice with two different field sets — the
        module reset cleared ``_pipeline`` but not the encoder/reranker
        references, while the HTTP admin route cleared the encoder/reranker but
        not ``_pipeline``. The two owners disagreed, so ``/health`` could
        report ``rag_chunks == 0`` while retrieval was still serving the
        previous corpus out of the stale pipeline.

        Every field is cleared together, under the lock, so no reader can
        observe a half-reset state.
        """
        with self.rag_lock:
            self.knowledge_index = None
            self.knowledge_chunks = []
            self.bm25_index = None
            self._pipeline = None
            self.embed_model_ref = None
            self._reranker = None

    def initialize_rag(self, enable_reranker: bool = True, force_rebuild: bool = False) -> int:
        """Load (or rebuild) the RAG index and wire the retrieval pipeline.

        ADR 0015 — the returned number is the count of chunks actually wired:
        cache hit → load; cache miss → scan + build + persist; no documents or
        any error → observable empty state (``/health`` reports
        ``rag_chunks == 0`` and degrades) with the reason logged with traceback.
        """
        try:
            count = self._wire_index(enable_reranker, force_rebuild)
        except Exception:
            # Fail safe *and* visible: the traceback is logged, the state is
            # cleared and /health degrades instead of pretending success.
            logger.exception("RAG initialization failed; retrieval stays disabled")
            self.reset_index_state()
            return 0
        return count

    def _wire_index(self, enable_reranker: bool, force_rebuild: bool) -> int:
        """Build/load the index and publish it on this instance (may raise)."""
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
            self.reset_index_state()
            return 0
        paths = get_paths()
        encoder = lazy_embedding_encoder(settings)

        index = chunks = bm25 = None
        if not force_rebuild and is_cache_valid(paths, settings):
            index, chunks, bm25 = load_cache(paths, settings, cipher=self._cipher)

        if index is None:
            registry = DocumentLoaderRegistry.create_default(
                settings.chunk_chars, settings.overlap_chars)
            scanned = scan_documents(paths, registry)
            if not scanned:
                logger.warning("RAG: no documents found under %s — index not built",
                               paths.project_root)
                self.reset_index_state()
                return 0
            # Cache is missing/stale: force the build path so the fs hash is
            # computed once (scan already happened above).
            index, chunks, bm25 = load_or_build_index(
                scanned, encoder, paths, settings, force_rebuild=True,
                cipher=self._cipher)

        if index is None or not chunks:
            logger.warning("RAG: index unavailable after load/build")
            self.reset_index_state()
            return 0

        from typing import cast
        vector_retriever, bm25_retriever = self._build_retrievers(
            index, bm25, cast("list[dict[str, Any]]", chunks), encoder, settings)
        pipeline = RetrievalPipeline(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            # The cross-encoder is an optional quality boost; it stays None
            # until a loader exists (pipeline treats None as a no-op).
            reranker=getattr(self, "_reranker", None),
            knowledge_chunks=chunks,
            max_context_chars=settings.max_context_chars,
            similarity_threshold=settings.similarity_threshold,
            use_reranker=bool(enable_reranker and settings.use_reranker),
            reranker_top_k=settings.reranker_top_k,
        )
        with self.rag_lock:
            self.knowledge_index = index
            self.knowledge_chunks = list(chunks)
            self.bm25_index = bm25
            self._pipeline = pipeline
            self.embed_model_ref = encoder
        count = len(chunks)
        logger.info("RAG ready: %d chunks (vector=%s bm25=%s)",
                    count, vector_retriever is not None, bm25_retriever is not None)
        return count

    def get_context(self, query: str, query_image_b64: str | None = None, result_k: int | None = None) -> tuple[str, str, list[Any]]:
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is None:
            return ("", "", [])
        try:
            doc_ctx, code_ctx, image_chunks = pipeline.get_context(query, query_image_b64, result_k)
        except Exception:
            logger.exception("RAG retrieval failed")
            return ("", "", [])
        return doc_ctx, code_ctx, image_chunks

    @property
    def reranker_active(self) -> bool:
        """Whether a cross-encoder is wired into the current pipeline.

        The HTTP health surface needs this fact; exposing the ``_reranker``
        field itself made the adapter depend on a private implementation
        detail (ADR 0032 keeps the state transition inside this module).
        """
        return self._reranker is not None

    def release_ml_memory(self) -> None:
        """Ask the runtime to free the resident models (VRAM/RAM).

        Public on purpose: the admin router used to call a private helper,
        which made a documented adapter-to-runtime call look like an
        implementation leak.
        """
        self.release_embedding_for_vision()

    def release_embedding_for_vision(self) -> None:
        from ai_assistant.llm.hardware import release_ml_memory
        release_ml_memory()


class ChatbotAgentHolder:
    """Owns the single hot-reloadable ``ChatbotAgent`` instance.

    The chat router captures a getter reference, not the agent instance, so a
    model/RAG reload can replace the agent without re-registering the router.
    """

    def __init__(self, llm_module: LlmModule, rag_module: RagModule,
                 prompt_builder: "PromptBuilder") -> None:
        self._llm_module = llm_module
        self._rag_module = rag_module
        self._prompt_builder = prompt_builder
        self._agent: ChatbotAgent | None = None

    def get(self) -> ChatbotAgent:
        if self._agent is None:
            from ai_assistant.orchestration.specialists import ChatbotAgent
            self._agent = ChatbotAgent(
                self._llm_module, self._rag_module, self._prompt_builder)
        return self._agent

    def rebuild(self) -> None:
        from ai_assistant.orchestration.specialists import ChatbotAgent
        self._agent = ChatbotAgent(
            self._llm_module, self._rag_module, self._prompt_builder)


class RuntimeModules:
    """The composition root's runtime module graph.

    One instance is built by ``compose()`` and injected into the routers and
    services; the module owns the handles, so nothing reaches for a global.
    ``agent_run_service`` is bound after the agent service exists (the A2A
    task executor is handed to the container first), so it starts as ``None``
    and :meth:`execute_a2a_task` reports the unbound state instead of failing
    with an ``AttributeError`` deep inside a task run.
    """

    def __init__(self, llm: LlmModule, rag: RagModule, prompt_builder: "PromptBuilder") -> None:
        self.llm = llm
        self.rag = rag
        self.chatbot_agent = ChatbotAgentHolder(llm, rag, prompt_builder)
        self.agent_run_service: AgentRunService | None = None
        # The admin shutdown route signals this event instead of calling
        # ``os._exit``: the serving process sets ``uvicorn.Server.should_exit``
        # and lets the normal lifespan cleanup run.
        self.shutdown_requested = threading.Event()

    def request_shutdown(self) -> None:
        """Ask the serving process to stop gracefully."""
        self.shutdown_requested.set()

    def execute_a2a_task(self, task: AgentTask) -> dict[str, Any]:
        """Run the shared LangGraph agent engine inside the durable A2A lifecycle."""
        if self.llm.llm is None:
            msg = "LLM is not initialized"
            raise RuntimeError(msg)
        if self.agent_run_service is None:
            msg = "Agent run service is not initialized"
            raise RuntimeError(msg)
        return self.agent_run_service.run(task)


__all__ = [
    "ChatbotAgentHolder",
    "LlmModule",
    "RagModule",
    "RuntimeModules",
]
