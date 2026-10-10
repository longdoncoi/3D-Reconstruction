"""Chat completion route (/v1/chat/completions).

Extracted from StartChatbotServer.py. Handles the chatbot conversational
endpoint — RAG retrieval, vision, context limit check, and streaming inference.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from .security import TransportPolicy, chat_guards

if TYPE_CHECKING:
    from ._types import ChatbotAgentLike, LLMModuleLike, LoggerLike, RAGModuleLike


class ChatMessage(BaseModel):
    """A single message in a chat request."""

    role: str
    content: str = Field(..., max_length=32000)
    attachments: list[str] | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        if value not in ("user", "assistant", "system", "assistant_agent"):
            raise ValueError("role must be user, assistant, system, or assistant_agent")
        return value


class ChatRequest(BaseModel):
    """Incoming /v1/chat/completions request body."""

    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1, le=4096)
    language: str = Field(default="vi", pattern="^(vi|en)$")


def build_chat_router(
    llm_module: "LLMModuleLike",
    rag_module: "RAGModuleLike",
    get_chatbot_agent: Callable[[], "ChatbotAgentLike"],
    llm_n_ctx: int,
    logger: "LoggerLike",
    policy: TransportPolicy | None = None,
) -> APIRouter:
    """Create the /v1/chat/completions APIRouter.

    Parameters
    ----------
    llm_module:        Live llm_module reference.
    rag_module:        Live rag_module reference.
    get_chatbot_agent: Callable returning the current ChatbotAgent instance.
                       Using a getter (not the agent directly) allows hot-reload
                       to propagate without re-registering the router.
    llm_n_ctx:         Context window size from config.
    logger:            Application logger.
    policy:            Transport trust policy (ADR 0008) — loopback peer,
                       allowed origin and optional ``AI_AGENT_TOKEN`` bearer.
    """
    router = APIRouter(tags=["chat"], dependencies=chat_guards(policy))

    @router.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest, http_req: Request):  # noqa: ANN201
        if llm_module.llm is None:
            raise HTTPException(status_code=503, detail="LLM is not initialized")

        req_start = time.monotonic()
        user_query = request.messages[-1].content
        attachments = request.messages[-1].attachments or []
        query_image_b64: str | None = None

        # image_utils ships with the full runtime; fall back to the rag_module
        # helpers when the RAG extras are not installed (offline CI).
        def fallback_is_image(_path: str) -> bool:
            return False

        def fallback_to_data_uri(_path: str) -> str:
            return ""

        try:
            from ai_assistant.rag.image_utils import image_to_data_uri, is_image_file

            image_check: Callable[[str], bool] = is_image_file
            data_uri_fn: Callable[[str], str] = image_to_data_uri
        except ImportError:
            image_check = getattr(rag_module, "_is_image_file", fallback_is_image)
            data_uri_fn = getattr(rag_module, "_image_to_data_uri", fallback_to_data_uri)

        for attachment in attachments:
            if image_check(attachment):
                try:
                    query_image_b64 = data_uri_fn(attachment)
                    break
                except Exception as error:
                    logger.warning("Failed to read attachment for retrieval: %s", error)

        client_host = http_req.client.host if http_req.client else "unknown"
        logger.info(
            "[MODE: CHAT] Query from %s: %s…",
            client_host,
            user_query[:60].replace("\n", " "),
        )
        from ai_assistant.orchestration.supervisor import Specialist

        logger.info("[SUPERVISOR] routed chat request to %s", Specialist.CHATBOT.value)

        rag_start = time.monotonic()
        messages_raw = [
            message.model_dump()
            for message in request.messages
            if message.role != "assistant_agent"
        ]
        if not messages_raw:
            raise HTTPException(
                status_code=422,
                detail="No conversational messages were supplied",
            )

        chatbot_agent = get_chatbot_agent()
        messages, chatbot_metadata = chatbot_agent.build_messages(
            messages_raw, user_query, query_image_b64, request.language,
        )
        rag_ms = (time.monotonic() - rag_start) * 1000

        if llm_module.is_vision_model:
            estimated_tokens = 0
            for message in messages:
                content = message.get("content")
                parts: list[dict[str, Any]] = (
                    content if isinstance(content, list) else [{"text": content or ""}]
                )
                for part in parts:
                    estimated_tokens += llm_module.estimate_tokens(part.get("text", ""))
        else:
            estimated_tokens = sum(
                llm_module.estimate_tokens(message.get("content", ""))
                for message in messages
            )

        if estimated_tokens >= llm_n_ctx - 512:
            raise HTTPException(
                status_code=400,
                detail="Conversation exceeds the model context window",
            )
        max_tokens = min(
            request.max_tokens,
            max(512, llm_n_ctx - estimated_tokens - 400),
        )

        try:
            with llm_module.llm_lock:
                started = time.monotonic()
                answer, finish_reason = "", "stop"
                stream_fn = (
                    llm_module.llm.generate
                    if hasattr(llm_module.llm, "generate")
                    else llm_module.llm.create_chat_completion
                )
                for chunk in stream_fn(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=request.temperature,
                    repeat_penalty=1.1,
                    stream=True,
                ):
                    choice = chunk["choices"][0]
                    answer += choice.get("delta", {}).get("content", "")
                    finish_reason = choice.get("finish_reason") or finish_reason
                llm_ms = (time.monotonic() - started) * 1000
        except Exception as error:
            logger.exception("LLM inference failed")
            raise HTTPException(
                status_code=500,
                detail=f"Inference error: {error}",
            ) from error

        answer = answer.strip()
        answer = chatbot_agent.clean_answer(answer, chatbot_metadata, finish_reason)

        total_ms = (time.monotonic() - req_start) * 1000
        logger.info("Done | rag=%.0fms llm=%.0fms total=%.0fms", rag_ms, llm_ms, total_ms)
        return {
            "id": f"chatcmpl-{int(req_start * 1000)}",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {},
            "x_meta": {
                "rag_ms": round(rag_ms),
                "llm_ms": round(llm_ms),
                "total_ms": round(total_ms),
                "estimated_tokens": estimated_tokens,
                "max_tokens_used": max_tokens,
                "reranker_active": rag_module._reranker is not None,
                "vision_model": llm_module.is_vision_model,
            },
        }

    return router
