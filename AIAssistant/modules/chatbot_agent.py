"""Standalone conversational agent: chat history plus optional RAG context.

This module deliberately has no dependency on the tool/desktop/coding agents.
The existing chat endpoint remains the public API; the class owns retrieval and
prompt assembly used by that endpoint only.
"""
from __future__ import annotations

from typing import Any

from .agent_logging import get_agent_logger

logger = get_agent_logger("chatbot")


class ChatbotAgent:
    def __init__(self, llm_runtime: Any, rag_runtime: Any) -> None:
        self._llm = llm_runtime
        self._rag = rag_runtime

    def build_messages(self, messages: list[dict], query: str, image_uri: str | None,
                       language: str) -> tuple[list[dict], dict[str, Any]]:
        logger.info("Chatbot Agent: retrieve context | query=%s", query[:120])
        doc_ctx, code_ctx, image_chunks = self._rag.get_context(query, query_image_b64=image_uri)
        if not image_uri:
            image_chunks = []
        suppress_citations = self._llm._is_character_query(query)
        if self._llm.is_vision_model:
            prepared = self._llm.build_vision_messages(
                messages, doc_ctx, code_ctx, image_chunks,
                suppress_citations=suppress_citations, language=language,
            )
        else:
            prepared = self._llm.build_text_messages(
                messages, doc_ctx, code_ctx,
                suppress_citations=suppress_citations, language=language,
            )
        logger.info("Chatbot Agent: context ready | docs=%d chars code=%d chars images=%d",
                    len(doc_ctx or ""), len(code_ctx or ""), len(image_chunks or []))
        return prepared, {"suppress_citations": suppress_citations}

    def clean_answer(self, answer: str, metadata: dict[str, Any], finish_reason: str) -> str:
        import re
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        
        if metadata.get("suppress_citations"):
            answer = self._llm._strip_reference_citations_for_character_answer(answer)
        if finish_reason == "length":
            answer += "\n\n⚠️ Response may be incomplete because the token limit was reached."
        return answer
