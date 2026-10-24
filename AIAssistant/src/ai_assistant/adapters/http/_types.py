"""Structural types for the dependencies injected into the HTTP routers.

The routers accept live module handles (the injected ``LlmModule`` and
``RagModule`` instances), the application ``platform`` object and a logger.
Typing them as ``object`` or ``types.ModuleType`` would disable static
analysis for the adapter layer; these protocols describe exactly the surface
each router touches, so the adapters depend on interfaces and stay honest
about what they reach into on the runtime handles (ADR 0001).
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol

from ...ports import LLMRuntime, ToolCatalog


class LoggerLike(Protocol):
    """Minimal logging surface required by the adapters."""

    def debug(self, message: str, *args: Any) -> None: ...
    def info(self, message: str, *args: Any) -> None: ...
    def warning(self, message: str, *args: Any) -> None: ...
    def error(self, message: str, *args: Any) -> None: ...
    def exception(self, message: str, *args: Any) -> None: ...


class LLMModuleLike(LLMRuntime, Protocol):
    """The injected LLM runtime handle (a live instance or an equivalent fake).

    The runtime-lifetime members (``llm``, ``llm_lock`` and ``model_lease``)
    are inherited from :class:`~ai_assistant.ports.LLMRuntime`; this protocol
    only adds the admin/chat surface the HTTP adapters need. Keeping one
    source for the lease contract is what stops the two views from drifting.
    """

    active_model_desc: str
    is_vision_model: bool

    def estimate_tokens(self, text: str) -> int: ...
    def release(self) -> bool: ...
    def load_model(self) -> None: ...
    def model_reload(self) -> AbstractContextManager[None]: ...


class RAGModuleLike(Protocol):
    """The injected RAG runtime handle (a live instance or an equivalent fake).

    Only public members are part of the adapter contract: the router asks the
    module whether a reranker is active and tells it to release memory; it
    does not reach into ``_reranker`` or call a private helper.
    """

    knowledge_chunks: list[Any]
    knowledge_index: Any
    bm25_index: Any
    embed_model_ref: Any
    rag_lock: Any
    CHUNK_CHARS: int
    MAX_CONTEXT_CHARS: int

    @property
    def reranker_active(self) -> bool: ...

    def reset_index_state(self) -> None:
        """Drop every retrieval handle (ADR 0032: the module owns this)."""
        ...

    def release_ml_memory(self) -> None: ...
    def release_embedding_for_vision(self) -> None: ...
    def initialize_rag(self, *, force_rebuild: bool, enable_reranker: bool) -> int: ...


class ActionManifestLike(Protocol):
    def reload_manifest(self) -> None: ...


class PluginSpecLike(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...


class PlatformSettingsLike(Protocol):
    @property
    def enable_mcp(self) -> bool: ...
    @property
    def enable_a2a(self) -> bool: ...
    @property
    def allow_remote_a2a(self) -> bool: ...


class PlatformLike(Protocol):
    # Declared as read-only properties so a frozen/slots container (a read-only
    # attribute) satisfies the protocol; a bare protocol attribute would demand
    # a settable variable (ADR 0001, mypy "read-only attribute" error).
    # The catalogue is the shared ``ToolCatalog`` port (ADR 0032) — the
    # adapter used to declare its own duck-typed duplicate of it.
    @property
    def plugins(self) -> ToolCatalog: ...
    @property
    def settings(self) -> PlatformSettingsLike: ...


class ChatbotAgentLike(Protocol):
    """The in-process chatbot agent used by the chat router.

    ``build_messages`` returns the prompt-ready message list plus free-form
    metadata consumed by ``clean_answer``, mirroring the legacy ChatbotAgent
    contract.
    """

    def build_messages(
        self,
        messages_raw: list[dict[str, Any]],
        user_query: str,
        query_image_b64: str | None,
        language: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...

    def clean_answer(
        self, answer: str, chatbot_metadata: dict[str, Any], finish_reason: str
    ) -> str: ...
