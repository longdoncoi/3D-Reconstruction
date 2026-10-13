"""Admin routes: model reload, RAG rebuild, agent reset, VRAM release, shutdown.

Extracted from the composition root. These are privileged management endpoints
that MUST NOT be exposed on the public network interface.
"""
from __future__ import annotations

import gc
import os
import threading
import time
from typing import TYPE_CHECKING, Callable

from fastapi import APIRouter, HTTPException

from .security import TransportPolicy, admin_guards

if TYPE_CHECKING:
    from ._types import (
        ActionManifestLike,
        LLMModuleLike,
        LoggerLike,
        RAGModuleLike,
    )


def _rebuild_rag_index(
    rag_module: "RAGModuleLike",
    llm_module: "LLMModuleLike",
    rebuild_chatbot_agent_fn: Callable[[], None],
    logger: "LoggerLike",
) -> dict:
    """Rebuild the RAG index; raises ``HTTPException`` on failure."""
    from ai_assistant.config.rag_config import RAGSettings
    _rag = RAGSettings.from_env_and_dict({})
    if not _rag.enabled:
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
        # Delegate to canonical RAG pipeline (ADR 0018); no module reload needed.
        rebuild_chatbot_agent_fn()
        chunks = rag_module.initialize_rag(
            force_rebuild=True,
            enable_reranker=not llm_module.is_vision_model,
        )
        if llm_module.is_vision_model:
            rag_module.release_embedding_for_vision()
        logger.info("RAG index rebuilt successfully")
    except Exception as error:
        logger.exception("RAG reload failed")
        raise HTTPException(status_code=500, detail=str(error)) from error
    else:
        return {
            "status": "ok",
            "chunks": chunks,
            "message": "RAG index rebuilt successfully",
        }


def build_admin_router(
    llm_module: "LLMModuleLike",
    rag_module: "RAGModuleLike",
    action_manifest_module: "ActionManifestLike",
    reset_agent_state_fn: Callable[[], None],
    models: list,
    model_idx: int,
    refresh_agent_routes_fn: Callable[[], None],
    rebuild_chatbot_agent_fn: Callable[[], None],
    logger: "LoggerLike",
    policy: TransportPolicy | None = None,
) -> APIRouter:
    """Create the /admin/* APIRouter with all injected dependencies.

    Parameters
    ----------
    llm_module:             The live llm_module reference (module-level globals used).
    rag_module:             The live rag_module reference.
    action_manifest_module: Reference to action_manifest module for reload.
    reset_agent_state_fn:   Callback that clears the injected agent service state.
    models:                 MODELS list from config.
    model_idx:              Current MODEL_IDX from config.
    refresh_agent_routes_fn: Callback to re-attach agent router after reload.
    rebuild_chatbot_agent_fn: Callback to recreate ChatbotAgent after module reload.
    logger:                 Application logger.
    """
    router = APIRouter(
        prefix="/admin",
        tags=["admin"],
        # Destructive management surface: loopback peer, allow-listed origin and
        # (when deployed with AI_ADMIN_TOKEN) a bearer token — see ADR 0008.
        dependencies=admin_guards(policy),
    )

    @router.post("/release-vram")
    def release_vram():
        try:
            with llm_module.llm_lock:
                old_llm, llm_module.llm = llm_module.llm, None
                del old_llm
                gc.collect()
                rag_module._release_ml_memory()
            logger.info("VRAM released successfully")
        except Exception as error:
            logger.exception("Failed to release VRAM")
            raise HTTPException(status_code=500, detail=str(error)) from error
        else:
            return {"status": "ok", "message": "VRAM and models unloaded"}

    @router.post("/reload-model")
    def reload_model():
        try:
            logger.info(
                "[MODEL SWITCH] Reload requested — current index=%d (%s)",
                model_idx, llm_module.active_model_desc,
            )
            with llm_module.llm_lock:
                old_llm, llm_module.llm = llm_module.llm, None
                del old_llm
                gc.collect()
                rag_module._release_ml_memory()
            rebuild_chatbot_agent_fn()
            llm_module.load_model()
            if llm_module.is_vision_model:
                rag_module.release_embedding_for_vision()
            logger.info(
                "[MODEL SWITCH] Model reloaded successfully: %s (vision=%s)",
                llm_module.active_model_desc, llm_module.is_vision_model,
            )
        except Exception as error:
            logger.exception("LLM reload failed")
            raise HTTPException(status_code=500, detail=str(error)) from error
        else:
            return {
                "status": "ok",
                "model": llm_module.active_model_desc,
                "message": "Model reloaded successfully",
            }

    @router.post("/reload-rag")
    def reload_rag():
        return _rebuild_rag_index(rag_module, llm_module, rebuild_chatbot_agent_fn, logger)

    @router.post("/reload-agent")
    def reload_agent():
        try:
            reset_agent_state_fn()
            refresh_agent_routes_fn()
            reset_agent_state_fn()
            action_manifest_module.reload_manifest()
            logger.info("Agent code and state reloaded successfully")
        except Exception as error:
            logger.exception("Agent reload failed")
            raise HTTPException(status_code=500, detail=str(error)) from error
        else:
            return {"status": "ok", "message": "Agent code and state reloaded successfully"}

    @router.post("/shutdown")
    def shutdown_server():
        """Terminate this AI Server process.

        Runs the actual exit from a background thread with a tiny delay so
        the HTTP response reaches the Qt client before the process disappears.
        """
        def _terminate() -> None:
            time.sleep(0.3)
            logger.info("Shutdown requested via /admin/shutdown — exiting process")
            os._exit(0)

        threading.Thread(target=_terminate, daemon=True).start()
        return {"status": "ok", "message": "AI Server is shutting down"}

    return router
