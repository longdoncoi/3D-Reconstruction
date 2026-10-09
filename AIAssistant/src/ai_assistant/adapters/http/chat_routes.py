"""Chat completion route (/v1/chat/completions).

Extracted from StartChatbotServer.py. Handles the chatbot conversational
endpoint — RAG retrieval, vision, context limit check, and streaming inference.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from types import ModuleType


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
    llm_module: "ModuleType",
    rag_module: "ModuleType",
    get_chatbot_agent: Callable,
    llm_n_ctx: int,
    logger: object,
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
    """
    router = APIRouter(tags=["chat"])

    @router.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest, http_req: Request):  # noqa: ANN201
        if llm_module.llm is None:
            raise HTTPException(status_code=503, detail="LLM is not initialized")

        req_start = time.monotonic()
        user_query = request.messages[-1].content
        attachments = request.messages[-1].attachments or []
        query_image_b64 = None

        try:
            from ai_assistant.rag.image_utils import image_to_data_uri, is_image_file
        except ImportError:
            is_image_file = getattr(rag_module, "_is_image_file", lambda _: False)
            image_to_data_uri = getattr(rag_module, "_image_to_data_uri", lambda _: "")

        for attachment in attachments:
            if is_image_file(attachment):
                try:
                    query_image_b64 = image_to_data_uri(attachment)
                    break
                except Exception as error:
                    logger.warning("Failed to read attachment for retrieval: %s", error)

        logger.info(
            "[MODE: CHAT] Query from %s: %s…",
            http_req.client.host,
            user_query[:60].replace("\n", " "),
        )
        from modules.multi_agent import Specialist

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
            estimated_tokens = sum(
                llm_module.estimate_tokens(part.get("text", ""))
                for message in messages
                for part in (
                    message.get("content")
                    if isinstance(message.get("content"), list)
                    else [{"text": message.get("content", "")}]
                )
            )
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
