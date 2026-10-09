"""Agent HTTP routes: /v1/agent/execute, /v1/agent/cancel, /v1/agent/ui-action-result, /v1/agent/approve.

Extracted from legacy agent_module.py into clean HTTP adapter. This adapter owns
transport negotiation (JSON vs Server-Sent Events); the agent service itself is
transport-agnostic and returns a plain result while optionally streaming steps
through an ``event_sink`` callback.
"""
from __future__ import annotations

import json
import queue
import threading
from typing import TYPE_CHECKING, Any, Callable

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ai_assistant.agents.models import (
    AgentApproveRequest,
    AgentCancelRequest,
    AgentExecuteRequest,
    AgentUiActionResultRequest,
)

if TYPE_CHECKING:
    from types import ModuleType


def _stream_langgraph_execution(run: Callable[[Callable[[dict], None]], dict]):
    """Stream agent node/tool steps as they happen over SSE."""
    events: queue.Queue[tuple[str, object]] = queue.Queue()

    def worker() -> None:
        try:
            events.put(("result", run(lambda step: events.put(("step", step)))))
        except Exception as error:  # noqa: BLE001
            events.put(("error", str(error)))

    threading.Thread(target=worker, daemon=True, name="agent-sse").start()

    def stream():
        yield 'event: status\ndata: {"status": "running"}\n\n'
        while True:
            kind, value = events.get()
            if kind == "step":
                yield f"event: step\ndata: {json.dumps(value, ensure_ascii=False)}\n\n"
            elif kind == "result":
                payload = value
                yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return
            else:
                yield f"event: error\ndata: {json.dumps({'detail': value}, ensure_ascii=False)}\n\n"
                return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def build_agent_router(agent_module: "ModuleType | Any" = None) -> APIRouter:
    """Create the /v1/agent/* APIRouter.

    Parameters
    ----------
    agent_module: The live agent service module providing execution services.
                  Defaults to ``ai_assistant.agents.service``.
    """
    if agent_module is None:
        from ai_assistant.agents import service as agent_module

    router = APIRouter(tags=["agent"])

    def _client_host(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    @router.post("/v1/agent/execute")
    def agent_execute(request: AgentExecuteRequest, http_req: Request):
        """Execute an agentic task with tool-calling loop (JSON or SSE)."""
        client_host = _client_host(http_req)
        if "text/event-stream" in http_req.headers.get("accept", ""):
            return _stream_langgraph_execution(
                lambda sink: agent_module.agent_execute(
                    request, event_sink=sink, client_host=client_host,
                )
            )
        return agent_module.agent_execute(request, client_host=client_host)

    @router.post("/v1/agent/cancel")
    def agent_cancel(request: AgentCancelRequest):
        """Request cooperative cancellation for a running session/request."""
        return agent_module.agent_cancel(request)

    @router.post("/v1/agent/ui-action-result")
    def agent_ui_action_result(request: AgentUiActionResultRequest):
        """Close the desktop-action loop after the Qt slot has run."""
        return agent_module.agent_ui_action_result(request)

    @router.post("/v1/agent/approve")
    def agent_approve(request: AgentApproveRequest):
        """Approve or reject a pending agent action (write_file, run_command)."""
        return agent_module.agent_approve(request)

    return router


__all__ = [
    "build_agent_router",
]
