"""Health, metrics, and model listing routes.

Extracted from StartChatbotServer.py. These are read-only endpoints that
expose server state — no LLM inference is performed here.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

if TYPE_CHECKING:
    from ._types import LLMModuleLike, PlatformLike, RAGModuleLike


def build_health_router(
    llm_module: "LLMModuleLike",
    rag_module: "RAGModuleLike",
    platform: "PlatformLike",
    server_start_time: float,
    embed_model_name: str,
    chars_per_token: int,
    model_idx: int,
    app: object,
) -> APIRouter:
    """Create the health/metrics/models APIRouter.

    All dependencies are injected as arguments so the router never accesses
    module-level globals directly — making it testable and swappable.
    """
    router = APIRouter()

    @router.get("/health")
    async def health():  # noqa: ANN201
        return {
            "status": "ok",
            "uptime_sec": round(time.monotonic() - server_start_time, 1),
            "llm_loaded": llm_module.llm is not None,
            "rag_chunks": len(rag_module.knowledge_chunks),
            "reranker": rag_module._reranker is not None,
            "embed_model": embed_model_name,
            "chunk_chars": rag_module.CHUNK_CHARS,
            "chars_per_token": chars_per_token,
            "max_context": rag_module.MAX_CONTEXT_CHARS,
            "model": llm_module.active_model_desc,
            "is_vision": llm_module.is_vision_model,
            "model_idx": model_idx,
            "platform": {
                "version": getattr(app, "version", "3.0.0"),
                "registered_tools": len(platform.plugins.specs()),
                "mcp_enabled": platform.settings.enable_mcp,
                "a2a_enabled": platform.settings.enable_a2a,
                "remote_a2a_enabled": platform.settings.allow_remote_a2a,
            },
        }

    @router.get("/metrics")
    async def metrics():  # noqa: ANN201
        """Prometheus exposition endpoint; enabled with AGENT_OBSERVABILITY=1."""
        from ai_assistant.observability import prometheus_payload

        payload = prometheus_payload()
        if payload is None:
            raise HTTPException(status_code=404, detail="Observability is disabled")
        return Response(payload, media_type="text/plain; version=0.0.4")

    @router.get("/v1/models")
    async def list_models():  # noqa: ANN201
        return {
            "data": [
                {
                    "id": llm_module.active_model_desc,
                    "object": "model",
                    "desc": llm_module.active_model_desc,
                }
            ]
        }

    return router
