"""Structural types for the dependencies injected into the HTTP routers.

The routers accept live legacy module handles (``llm_module``, ``rag_module``),
the application ``platform`` object and a logger. Typing them as ``object`` or
``types.ModuleType`` would disable static analysis for the adapter layer;
these protocols describe exactly the surface each router touches, so the
adapters depend on interfaces and stay honest about what they reach into on
the legacy singletons (ADR 0001).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class LoggerLike(Protocol):
    """Minimal logging surface required by the adapters."""

    def debug(self, message: str, *args: Any) -> None: ...
    def info(self, message: str, *args: Any) -> None: ...
    def warning(self, message: str, *args: Any) -> None: ...
    def error(self, message: str, *args: Any) -> None: ...
    def exception(self, message: str, *args: Any) -> None: ...


class LLMModuleLike(Protocol):
    """The live llm_module handle (legacy singleton or an equivalent fake)."""

    llm: Any
    llm_lock: Any
    active_model_desc: str
    is_vision_model: bool

    def estimate_tokens(self, text: str) -> int: ...
    def load_model(self) -> None: ...


class RAGModuleLike(Protocol):
    """The live rag_module handle (legacy singleton or an equivalent fake)."""

    knowledge_chunks: list[Any]
    knowledge_index: Any
    bm25_index: Any
    embed_model_ref: Any
    _reranker: Any
    rag_lock: Any
    CHUNK_CHARS: int
    MAX_CONTEXT_CHARS: int

    def _release_ml_memory(self) -> None: ...
    def release_embedding_for_vision(self) -> None: ...
    def initialize_rag(self, *, force_rebuild: bool, enable_reranker: bool) -> int: ...


class AttachmentHelpers(Protocol):
    """Optional image helpers used by the chat router's ImportError fallback."""

    def _is_image_file(self, path: str) -> bool: ...
    def _image_to_data_uri(self, path: str) -> str: ...


class ActionManifestLike(Protocol):
    def reload_manifest(self) -> None: ...


class PluginSpecLike(Protocol):
    name: str
    description: str


class PluginRegistryLike(Protocol):
    def specs(self) -> Sequence[PluginSpecLike]: ...


class PlatformSettingsLike(Protocol):
    enable_mcp: bool
    enable_a2a: bool
    allow_remote_a2a: bool


class PlatformLike(Protocol):
    plugins: PluginRegistryLike
    settings: PlatformSettingsLike


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
