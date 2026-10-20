"""Conversational specialist: chat history plus optional RAG context."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_assistant.config.logging import get_agent_logger
from ai_assistant.llm.inference import strip_think_tags

if TYPE_CHECKING:
    from ai_assistant.llm.contracts import PromptBuilder
    from ai_assistant.ports import RagContextPort, VisionRuntimePort

logger = get_agent_logger("chatbot")


class ChatbotAgent:
    """Specialist responsible for conversational turns and RAG prompt assembly.

    The prompt building lives behind the :class:`PromptBuilder` port, not on
    the LLM runtime handle. It used to be reached through ``self._llm``,
    which only worked while the runtime and the prompt builder were the same
    object: after the runtime was split into ``LlmModule`` (backend handle) and
    ``DefaultPromptBuilder`` (prompt assembly), the specialist kept calling
    the old names on the new object and every request failed with an
    ``AttributeError``. The dependency is now explicit and typed, so a missing
    method fails at construction instead of on the first request.
    """

    def __init__(self, llm_runtime: "VisionRuntimePort", rag_runtime: "RagContextPort",
                 prompt_builder: "PromptBuilder") -> None:
        self._llm = llm_runtime
        self._rag = rag_runtime
        self._prompts = prompt_builder

    def build_messages(
        self,
        messages: list[dict[str, Any]],
        query: str,
        image_uri: str | None,
        language: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        logger.info("Chatbot Agent: retrieve context | query=%s", query[:120])
        doc_ctx, code_ctx, image_chunks = self._rag.get_context(query, query_image_b64=image_uri)
        if not image_uri:
            image_chunks = []
        suppress_citations = self._prompts.is_character_query(query)
        if self._llm.is_vision_model:
            prepared = self._prompts.build_vision_messages(
                messages,
                doc_ctx,
                code_ctx,
                image_chunks,
                suppress_citations=suppress_citations,
                language=language,
            )
        else:
            prepared = self._prompts.build_messages(
                messages,
                doc_ctx,
                code_ctx,
                suppress_citations=suppress_citations,
                language=language,
            )
        logger.info(
            "Chatbot Agent: context ready | docs=%d chars code=%d chars images=%d",
            len(doc_ctx or ""),
            len(code_ctx or ""),
            len(image_chunks or []),
        )
        return prepared, {"suppress_citations": suppress_citations}

    def clean_answer(self, answer: str, metadata: dict[str, Any], finish_reason: str) -> str:
        answer = strip_think_tags(answer)

        if metadata.get("suppress_citations"):
            answer = self._prompts.strip_citations(answer)
        if finish_reason == "length":
            answer += "\n\n⚠️ Response may be incomplete because the token limit was reached."
        return answer


__all__ = ["ChatbotAgent"]
