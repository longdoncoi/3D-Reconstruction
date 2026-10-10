"""Domain contracts for LLM backends and prompt formatting."""
from __future__ import annotations

from typing import Any, Protocol

from ..ports import LLMBackend

__all__ = ["LLMBackend", "PromptBuilder"]


class PromptBuilder(Protocol):
    """Abstract interface for constructing LLM context."""
    
    def build_messages(
        self,
        messages: list[dict[str, Any]],
        doc_ctx: str,
        code_ctx: str,
        language: str = "vi",
        suppress_citations: bool = False,
    ) -> list[dict[str, Any]]:
        """Construct the full message list for a text model."""
        ...
        
    def build_vision_messages(
        self,
        messages: list[dict[str, Any]],
        doc_ctx: str,
        code_ctx: str,
        image_chunks: list[str] | None = None,
        language: str = "vi",
        suppress_citations: bool = False,
    ) -> list[dict[str, Any]]:
        """Construct the full message list for a vision-capable model."""
        ...

    def is_character_query(self, query: str) -> bool:
        """Detect if the query asks about project roles/characters."""
        ...

    def strip_citations(self, answer: str) -> str:
        """Remove source citations from an answer."""
        ...
