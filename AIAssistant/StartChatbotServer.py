"""FastAPI router and entry point for the 3D-Reconstruction AI server."""

import gc
import importlib
import os
import sys
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from modules import agent_module, llm_module, mcp_server, rag_module
from modules.config import (
    _SERVER_START_TIME,
    BASE_DIR,
    CHARS_PER_TOKEN,
    EMBED_MODEL_NAME,
    ENABLE_RAG,
    LLM_N_CTX,
    LOG_FILE_PATH,
    MODEL_IDX,
    MODELS,
    _safe_relpath,
    logger,
)


class ChatMessage(BaseModel):
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
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1, le=4096)
    language: str = Field(default="vi", pattern="^(vi|en)$")


@asynccontextmanager
async def lifespan(_: FastAPI):
    total = time.monotonic() - _SERVER_START_TIME
    
    # Dọn dẹp checkpoint cũ nếu có
    try:
        from modules.checkpointing import cleanup_old_checkpoints
        cleanup_old_checkpoints()
    except Exception as e:
        logger.warning(f"Lỗi dọn dẹp checkpoint: {e}")

    logger.info("Server ready in %.1fs — http://127.0.0.1:8080", total)
    print(f"[SUCCESS] AI Server started successfully ({total:.1f}s)", flush=True)
    if mcp_server.MCP_AVAILABLE:
        async with mcp_server.lifespan():
            yield
    else:
        logger.warning("MCP SDK is not installed; MCP endpoint is unavailable")
        yield
    logger.info("Server shutdown after %.1fs", time.monotonic() - _SERVER_START_TIME)


app = FastAPI(title="3D-Reconstruction AI Server", version="2.4.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"],
)
app.include_router(agent_module.agent_router)
if mcp_server.MCP_AVAILABLE:
    app.mount("/mcp", mcp_server.asgi_app())


def _refresh_agent_routes() -> None:
    """Replace FastAPI's old Agent handlers after reloading agent_module."""
    agent_paths = {"/v1/agent/execute", "/v1/agent/approve", "/v1/agent/ui-action-result"}
    app.router.routes[:] = [route for route in app.router.routes
                            if getattr(route, "path", None) not in agent_paths]
    app.include_router(agent_module.agent_router)
    app.openapi_schema = None


@app.post("/admin/release-vram")
def release_vram():
    try:
        with llm_module.llm_lock:
            old_llm, llm_module.llm = llm_module.llm, None
            del old_llm
            gc.collect()
            rag_module._release_ml_memory()
        logger.info("VRAM released successfully")
        return {"status": "ok", "message": "VRAM and models unloaded"}
    except Exception as error:
        logger.error("Failed to release VRAM: %s", error)
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/admin/reload-model")
def reload_model():
    try:
        with llm_module.llm_lock:
            old_llm, llm_module.llm = llm_module.llm, None
            del old_llm
            gc.collect()
            rag_module._release_ml_memory()
        importlib.reload(llm_module)
        llm_module.load_model()
        if llm_module.is_vision_model:
            rag_module.release_embedding_for_vision()
        logger.info("Model reloaded successfully")
        return {"status": "ok", "model": llm_module.active_model_desc,
                "message": "Model reloaded successfully"}
    except Exception as error:
        logger.exception("LLM reload failed")
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/admin/reload-rag")
def reload_rag():
    if not ENABLE_RAG:
        return {"status": "skipped", "message": "RAG is disabled"}
    try:
        with rag_module.rag_lock:
            rag_module.knowledge_index = None
            rag_module.knowledge_chunks = []
            rag_module.bm25_index = None
            rag_module.embed_model_ref = None
            rag_module._reranker = None
        gc.collect()
        rag_module._release_ml_memory()
        importlib.reload(rag_module)
        chunks = rag_module.initialize_rag(
            force_rebuild=True, enable_reranker=not llm_module.is_vision_model,
        )
        if llm_module.is_vision_model:
            rag_module.release_embedding_for_vision()
        logger.info("RAG index rebuilt successfully")
        return {"status": "ok", "chunks": chunks, "message": "RAG index rebuilt successfully"}
    except Exception as error:
        logger.exception("RAG reload failed")
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/admin/reload-agent")
def reload_agent():
    try:
        agent_module.reset_agent_state()
        # agent_module imports LocalAgentGraph by name, so reload the graph
        # module first; otherwise a running server keeps the previous routing
        # implementation after an Agent reload.
        import LangGraphAgent
        importlib.reload(LangGraphAgent)
        importlib.reload(agent_module)
        _refresh_agent_routes()
        agent_module.reset_agent_state()
        logger.info("Agent code and state reloaded successfully")
        return {"status": "ok", "message": "Agent code and state reloaded successfully"}
    except Exception as error:
        logger.exception("Agent reload failed")
        raise HTTPException(status_code=500, detail=str(error)) from error

@app.post("/admin/shutdown")
def shutdown_server():
    """Terminate this AI Server process so ServerManager can spawn a fresh one.

    Runs the actual exit from a background thread with a tiny delay so the
    HTTP response reaches the Qt client before the process disappears.
    """
    def _terminate():
        time.sleep(0.3)
        logger.info("Shutdown requested via /admin/shutdown — exiting process")
        os._exit(0)

    threading.Thread(target=_terminate, daemon=True).start()
    return {"status": "ok", "message": "AI Server is shutting down"}        


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest, http_req: Request):
    if llm_module.llm is None:
        raise HTTPException(status_code=503, detail="LLM is not initialized")

    req_start = time.monotonic()
    user_query = request.messages[-1].content
    attachments = request.messages[-1].attachments or []
    query_image_b64 = None
    for attachment in attachments:
        if rag_module._is_image_file(attachment):
            try:
                query_image_b64 = rag_module._image_to_data_uri(attachment)
                break
            except Exception as error:
                logger.warning("Failed to read attachment for retrieval: %s", error)

    logger.info("[MODE: CHAT] Query from %s: %s…", http_req.client.host,
                user_query[:60].replace("\n", " "))
    suppress_citations = llm_module._is_character_query(user_query)
    rag_start = time.monotonic()
    doc_ctx, code_ctx, image_chunks = rag_module.get_context(user_query, query_image_b64=query_image_b64)
    if not query_image_b64:
        image_chunks = []
    rag_ms = (time.monotonic() - rag_start) * 1000

    # ``assistant_agent`` is stored for the Qt renderer only. It is structured
    # execution state, not natural-language conversation for the chat model.
    messages_raw = [message.model_dump() for message in request.messages
                    if message.role != "assistant_agent"]
    if not messages_raw:
        raise HTTPException(status_code=422, detail="No conversational messages were supplied")
    if llm_module.is_vision_model:
        messages = llm_module.build_vision_messages(messages_raw, doc_ctx, code_ctx, image_chunks,
                                                    suppress_citations=suppress_citations,
                                                    language=request.language)
        estimated_tokens = sum(llm_module.estimate_tokens(part.get("text", ""))
                               for message in messages
                               for part in (message.get("content") if isinstance(message.get("content"), list)
                                            else [{"text": message.get("content", "")}]))
    else:
        messages = llm_module.build_text_messages(messages_raw, doc_ctx, code_ctx,
                                                  suppress_citations=suppress_citations,
                                                  language=request.language)
        estimated_tokens = sum(llm_module.estimate_tokens(message.get("content", "")) for message in messages)

    if estimated_tokens >= LLM_N_CTX - 512:
        raise HTTPException(status_code=400, detail="Conversation exceeds the model context window")
    max_tokens = min(request.max_tokens, max(512, LLM_N_CTX - estimated_tokens - 400))

    try:
        with llm_module.llm_lock:
            started = time.monotonic()
            answer, finish_reason = "", "stop"
            for chunk in llm_module.llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=request.temperature,
                repeat_penalty=1.1, stream=True,
            ):
                choice = chunk["choices"][0]
                answer += choice.get("delta", {}).get("content", "")
                finish_reason = choice.get("finish_reason") or finish_reason
            llm_ms = (time.monotonic() - started) * 1000
    except Exception as error:
        logger.exception("LLM inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {error}") from error

    answer = answer.strip()
    if suppress_citations:
        answer = llm_module._strip_reference_citations_for_character_answer(answer)
    if finish_reason == "length":
        answer += "\n\n⚠️ Response may be incomplete because the token limit was reached."

    total_ms = (time.monotonic() - req_start) * 1000
    logger.info("Done | rag=%.0fms llm=%.0fms total=%.0fms", rag_ms, llm_ms, total_ms)
    return {
        "id": f"chatcmpl-{int(req_start * 1000)}", "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": answer},
                     "finish_reason": finish_reason}], "usage": {},
        "x_meta": {"rag_ms": round(rag_ms), "llm_ms": round(llm_ms),
                   "total_ms": round(total_ms), "estimated_tokens": estimated_tokens,
                   "max_tokens_used": max_tokens, "reranker_active": rag_module._reranker is not None,
                   "vision_model": llm_module.is_vision_model},
    }


@app.get("/health")
async def health():
    return {
        "status": "ok", "uptime_sec": round(time.monotonic() - _SERVER_START_TIME, 1),
        "llm_loaded": llm_module.llm is not None, "rag_chunks": len(rag_module.knowledge_chunks),
        "reranker": rag_module._reranker is not None, "embed_model": EMBED_MODEL_NAME,
        "chunk_chars": rag_module.CHUNK_CHARS, "chars_per_token": CHARS_PER_TOKEN,
        "max_context": rag_module.MAX_CONTEXT_CHARS, "model": llm_module.active_model_desc,
        "is_vision": llm_module.is_vision_model,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus exposition endpoint; enabled with AGENT_OBSERVABILITY=1."""
    from fastapi.responses import Response

    from modules.observability import prometheus_payload
    payload = prometheus_payload()
    if payload is None:
        raise HTTPException(status_code=404, detail="Observability is disabled")
    return Response(payload, media_type="text/plain; version=0.0.4")


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": llm_module.active_model_desc, "object": "model",
                      "desc": llm_module.active_model_desc}]}


def bootstrap_runtime() -> None:
    """Initialize RAG and LLM once before Uvicorn accepts requests."""
    logger.info("Starting runtime | model index=%d | log=%s", MODEL_IDX,
                _safe_relpath(LOG_FILE_PATH, BASE_DIR))
    rag_module.initialize_rag(enable_reranker=not MODELS[MODEL_IDX].get("is_vision", False))
    llm_module.load_model()
    if llm_module.is_vision_model:
        rag_module.release_embedding_for_vision()


if __name__ == "__main__":
    import uvicorn

    try:
        bootstrap_runtime()
    except Exception:
        logger.exception("AI server startup failed")
        sys.exit(1)
    try:
        # server_manager starts us with DETACHED_PROCESS, where stdout/stderr can
        # be None.  Uvicorn's default logging config calls sys.stdout.isatty()
        # and aborts in that case, so retain the application's file logger.
        uvicorn.run(app, host="127.0.0.1", port=8080, log_config=None, access_log=False)
    except BaseException:
        logger.exception("Uvicorn terminated during server startup")
        raise
