"""Health, metrics, and model listing routes.

Extracted from the composition root. These are read-only endpoints that
expose server state — no LLM inference is performed here.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response

if TYPE_CHECKING:
    from ._types import LLMModuleLike, PlatformLike, RAGModuleLike


def build_health_router(
    llm_module: "LLMModuleLike",
    rag_module: "RAGModuleLike",
    platform: "PlatformLike",
    server_start_time: float,
    embed_model_name: str,
    chars_per_token: float,
    model_idx: int,
    app: object,
) -> APIRouter:
    """Create the health/metrics/models APIRouter.

    All dependencies are injected as arguments so the router never accesses
    module-level globals directly — making it testable and swappable.
    """
    router = APIRouter()

    def _degraded_reasons() -> list[str]:
        """Why the process is not fully functional (empty ⇒ healthy)."""
        reasons: list[str] = []
        if llm_module.llm is None:
            reasons.append("llm_not_loaded")
        if not rag_module.knowledge_chunks:
            # P0 #1: an empty RAG index used to be invisible; surface it.
            reasons.append("rag_index_empty")
        return reasons

    @router.get("/livez")
    async def livez() -> dict[str, Any]:
        """Liveness: the event loop answered — nothing more is implied."""
        return {"status": "alive", "uptime_sec": round(time.monotonic() - server_start_time, 1)}

    @router.get("/readyz")
    async def readyz() -> JSONResponse:
        """Readiness: 503 while the LLM cannot serve, 200 ``degraded`` when RAG is empty.

        The distinction matters for orchestration: a missing model means agent
        requests cannot be answered at all (unready), whereas an empty corpus
        only costs retrieval quality (ready, but degraded).
        """
        reasons = _degraded_reasons()
        unready = [r for r in reasons if r == "llm_not_loaded"]
        status = "not_ready" if unready else ("degraded" if reasons else "ready")
        return JSONResponse(
            status_code=503 if unready else 200,
            content={"status": status, "reasons": reasons},
        )

    @router.get("/health")
    async def health() -> dict[str, Any]:
        reasons = _degraded_reasons()
        # One source of truth: the model description is only advertised while
        # a backend is actually held. After ``/admin/release-vram`` the old
        # route left ``active_model_desc`` populated, so ``/health`` said
        # "unloaded" while ``/v1/models`` kept advertising the released model.
        loaded = llm_module.llm is not None
        return {
            # Degraded (not "ok") whenever a subsystem is missing, so a probe
            # reading only ``status`` sees the truth.
            "status": "degraded" if reasons else "ok",
            "reasons": reasons,
            "uptime_sec": round(time.monotonic() - server_start_time, 1),
            "llm_loaded": loaded,
            "rag_chunks": len(rag_module.knowledge_chunks),
            "reranker": rag_module.reranker_active,
            "embed_model": embed_model_name,
            "chunk_chars": rag_module.CHUNK_CHARS,
            "chars_per_token": chars_per_token,
            "max_context": rag_module.MAX_CONTEXT_CHARS,
            "model": llm_module.active_model_desc if loaded else "",
            "is_vision": llm_module.is_vision_model if loaded else False,
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
    async def metrics() -> Response:
        """Prometheus exposition endpoint; enabled with AGENT_OBSERVABILITY=1."""
        from ai_assistant.observability import prometheus_payload

        payload = prometheus_payload()
        if payload is None:
            raise HTTPException(status_code=404, detail="Observability is disabled")
        return Response(payload, media_type="text/plain; version=0.0.4")

    @router.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        # No backend ⇒ no model to advertise. Returning an entry with an empty
        # id made a released model look served to OpenAI-compatible clients.
        if llm_module.llm is None:
            return {"data": []}
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
