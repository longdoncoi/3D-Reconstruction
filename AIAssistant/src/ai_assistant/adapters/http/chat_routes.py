"""Chat completion route (/v1/chat/completions).

Extracted from the composition root. Handles the chatbot conversational
endpoint — RAG retrieval, vision, context limit check, and streaming inference.
"""
from __future__ import annotations

import os
import stat
import time
from typing import TYPE_CHECKING, Any, Callable, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from ...config.paths import attachment_roots
from ...domain.context_budget import completion_token_budget, context_exhausted
from ...domain.governance import scrub_pii
from ...domain.specialists import Specialist
from ...path_policy import resolve_within
from ...ports import SEALED_BINARY_PREFIX
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
            msg = "role must be user, assistant, system, or assistant_agent"
            raise ValueError(msg)
        return value


class ChatRequest(BaseModel):
    """Incoming /v1/chat/completions request body."""

    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1, le=4096)
    language: str = Field(default="vi", pattern="^(vi|en)$")


def _scan_attachments_for_image(
    attachments: list[str], rag_module: "RAGModuleLike", logger: "LoggerLike",
    unseal: Callable[[bytes], bytes] | None = None,
) -> str | None:
    """Return the first attached image as a data URI, else ``None``.

    ``image_utils`` ships with the full runtime; fall back to the rag_module
    helpers when the RAG extras are not installed (offline CI).

    ``unseal`` opens an attachment that was sealed at rest (ADR 0029) — the
    chat attachments live encrypted on disk, so the route has to open them
    before the model can see them.
    """
    def fallback_is_image(_path: str) -> bool:
        return False

    def fallback_to_data_uri(_path: str) -> str:
        return ""

    image_check: Callable[[str], bool]
    data_uri_fn: Callable[[str], str]
    try:
        from ai_assistant.image_utils import image_to_data_uri, is_image_file

        image_check = is_image_file

        def _to_data_uri(path: str) -> str:
            return image_to_data_uri(path, unseal=unseal)

        data_uri_fn = _to_data_uri
    except ImportError:
        image_check = getattr(rag_module, "_is_image_file", fallback_is_image)
        data_uri_fn = getattr(rag_module, "_image_to_data_uri", fallback_to_data_uri)

    for attachment in attachments:
        if not image_check(attachment):
            continue
        if not _attachment_may_be_opened(attachment):
            logger.warning("Refusing to open a chat attachment outside the platform's upload roots: %r", attachment)
            continue
        try:
            return data_uri_fn(attachment)
        except Exception as error:
            logger.warning("Failed to read attachment for retrieval: %s", error)
    return None


def _attachment_may_be_opened(path: str) -> bool:
    """Whether the route may open a *client-named* file.

    The request body is untrusted and the route holds the at-rest key, so a
    path that arrives from a client is not a path the server may open. Two
    independent gates, either of which is enough:

    * the path resolves inside a directory the desktop client stores
      attachments in (the platform's upload roots) — the only place a
      legitimate attachment ever lives;
    * or the file is a **regular file** whose first bytes carry the platform's
      sealed-envelope marker. The marker is a *hint*, not a proof of
      provenance: anyone can write those six bytes. What keeps the content
      safe is the next step — the route *unseals* the file, and the AEAD
      refuses anything the platform did not seal, so a planted marker yields
      an error and never a document.

    Only a regular file is opened. A FIFO or a device would block the worker
    thread on a path the client chose, which is a denial of service the
    request body could trigger.
    """
    for root in attachment_roots():
        if resolve_within(root, path) is not None:
            return True
    try:
        if not stat.S_ISREG(os.stat(path).st_mode):
            return False
        with open(path, "rb") as handle:
            return handle.read(len(SEALED_BINARY_PREFIX)) == SEALED_BINARY_PREFIX
    except OSError:
        return False


def build_chat_router(
    llm_module: "LLMModuleLike",
    rag_module: "RAGModuleLike",
    get_chatbot_agent: Callable[[], "ChatbotAgentLike"],
    llm_n_ctx: int,
    logger: "LoggerLike",
    policy: TransportPolicy | None = None,
    unseal: Callable[[bytes], bytes] | None = None,
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
    unseal:            Opens an attachment sealed at rest (ADR 0029).
    """
    router = APIRouter(tags=["chat"], dependencies=chat_guards(policy))

    @router.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest, http_req: Request) -> dict[str, Any]:
        if llm_module.llm is None:
            raise HTTPException(status_code=503, detail="LLM is not initialized")

        req_start = time.monotonic()
        user_query = request.messages[-1].content
        attachments = request.messages[-1].attachments or []
        query_image_b64 = _scan_attachments_for_image(attachments, rag_module, logger, unseal)

        client_host = http_req.client.host if http_req.client else "unknown"
        # ADR 0032: the user's own words are scrubbed before they reach a
        # plaintext log file, exactly like the model output on the other side
        # of the same request.
        logger.info(
            "[MODE: CHAT] Query from %s: %s…",
            client_host,
            scrub_pii(user_query)[:60].replace("\n", " "),
        )
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
        # ADR 0040: one request sees one model. The lease spans prompt
        # assembly (which reads ``is_vision_model``), token estimation and the
        # whole generation; ``/admin/reload-model`` and ``/admin/release-vram``
        # wait for every lease before they swap or free the backend. The route
        # used to snapshot the handle under ``llm_lock`` and generate outside
        # it, so a reload that started mid-request freed the model under the
        # stream.
        with llm_module.model_lease() as backend:
            if backend is None:
                # The fast check above saw a model, but a reload may have won
                # the race for the lease. The backend the lease pins is the
                # authoritative read.
                raise HTTPException(status_code=503, detail="LLM is not initialized")
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

            # One domain policy with the ReAct loop (domain/context_budget.py):
            # same refusal threshold, same completion budget, one set of constants.
            if context_exhausted(estimated_tokens, llm_n_ctx):
                raise HTTPException(
                    status_code=400,
                    detail="Conversation exceeds the model context window",
                )
            max_tokens = min(
                request.max_tokens,
                completion_token_budget(llm_n_ctx, estimated_tokens),
            )

            try:
                # Generation is serialized inside the backend that owns the
                # single-threaded llama context; ``llm_lock`` only guards the
                # handle swap itself, and the lease already keeps the reload
                # from swapping it. A ``None`` backend was refused above.
                # ``generate`` is the port; the raw llama.cpp call is the
                # legacy fallback for a handle that predates it, so stepping
                # outside the protocol there is deliberate.
                stream_fn = (
                    backend.generate
                    if hasattr(backend, "generate")
                    else cast(Any, backend).create_chat_completion
                )
                started = time.monotonic()
                answer, finish_reason = "", "stop"
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
                    "reranker_active": rag_module.reranker_active,
                    "vision_model": llm_module.is_vision_model,
                },
            }

    return router
