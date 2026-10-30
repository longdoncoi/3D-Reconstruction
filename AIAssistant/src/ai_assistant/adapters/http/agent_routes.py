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
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ai_assistant.agents.models import (
    AgentApproveRequest,
    AgentCancelRequest,
    AgentExecuteRequest,
    AgentUiActionResultRequest,
)
from ai_assistant.domain.memory import DEFAULT_MEMORY_OWNER

from .security import TransportPolicy, agent_guards

if TYPE_CHECKING:
    from ai_assistant.agents.service import AgentService


def _cancel_agent_run(agent_service: "AgentService", session_id: str) -> None:
    """Best-effort cooperative cancel when the SSE client disconnects.

    The run is already on a worker thread; without this a dropped browser/Qt
    connection leaves it consuming the local model until completion.
    """
    try:
        agent_service.cancel(AgentCancelRequest(session_id=session_id or "agent_default"))
    except Exception:
        # The run may have finished between the disconnect and this callback.
        pass


def _sse_events(events: queue.Queue[tuple[str, object]],
                on_disconnect: Callable[[], None] | None = None) -> Iterable[str]:
    """Turn worker events into SSE frames and cancel a dropped run.

    Kept separate from the thread/response wiring so the disconnect path is
    testable without an HTTP transport: closing this generator before a
    ``result``/``error`` frame is exactly what Starlette does when the client
    drops the connection.
    """
    finished = False
    try:
        yield 'event: status\ndata: {"status": "running"}\n\n'
        while True:
            kind, value = events.get()
            if kind == "step":
                yield f"event: step\ndata: {json.dumps(value, ensure_ascii=False)}\n\n"
            elif kind == "result":
                finished = True
                yield f"event: done\ndata: {json.dumps(value, ensure_ascii=False)}\n\n"
                return
            else:
                finished = True
                yield f"event: error\ndata: {json.dumps({'detail': value}, ensure_ascii=False)}\n\n"
                return
    finally:
        if not finished and on_disconnect is not None:
            on_disconnect()


def _stream_langgraph_execution(
    run: Callable[[Callable[[dict[str, Any]], None]], dict[str, Any]],
    on_disconnect: Callable[[], None] | None = None,
) -> StreamingResponse:
    """Stream agent node/tool steps as they happen over SSE."""
    events: queue.Queue[tuple[str, object]] = queue.Queue()

    def worker() -> None:
        try:
            events.put(("result", run(lambda step: events.put(("step", step)))))
        except Exception as error:
            events.put(("error", str(error)))

    threading.Thread(target=worker, daemon=True, name="agent-sse").start()

    return StreamingResponse(
        _sse_events(events, on_disconnect),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def build_agent_router(agent_service: "AgentService | None" = None,
                       policy: TransportPolicy | None = None) -> APIRouter:
    """Create the /v1/agent/* APIRouter.

    Parameters
    ----------
    agent_service: The injected agent service instance. The composition root
                   builds one ``AgentService`` with the shared tool gateway and
                   passes it here; a bare default is only used by local tooling.
    policy:        Transport trust policy (ADR 0008) — loopback peer, allowed
                   origin and optional ``AI_AGENT_TOKEN`` bearer.
    """
    if agent_service is None:
        from ai_assistant.agents.service import AgentService
        agent_service = AgentService()

    router = APIRouter(tags=["agent"], dependencies=agent_guards(policy))

    def _client_host(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    @router.post("/v1/agent/execute", response_model=None)
    def agent_execute(request: AgentExecuteRequest, http_req: Request) -> dict[str, Any] | StreamingResponse:
        """Execute an agentic task with tool-calling loop (JSON or SSE).

        This surface is loopback-only (``agent_guards(allow_remote=False)``),
        so every run belongs to the local desktop install and therefore to
        ``DEFAULT_MEMORY_OWNER``. A multi-tenant caller enters through A2A,
        where the task's tenant scopes the memory instead (ADR 0043).
        """
        client_host = _client_host(http_req)
        if "text/event-stream" in http_req.headers.get("accept", ""):
            return _stream_langgraph_execution(
                lambda sink: agent_service.execute(
                    request, owner=DEFAULT_MEMORY_OWNER,
                    event_sink=sink, client_host=client_host,
                ),
                on_disconnect=partial(_cancel_agent_run, agent_service, request.session_id),
            )
        return agent_service.execute(
            request, owner=DEFAULT_MEMORY_OWNER, client_host=client_host,
        )

    @router.post("/v1/agent/cancel")
    def agent_cancel(request: AgentCancelRequest) -> dict[str, Any]:
        """Request cooperative cancellation for a running session/request."""
        return agent_service.cancel(request)

    @router.post("/v1/agent/ui-action-result")
    def agent_ui_action_result(request: AgentUiActionResultRequest) -> dict[str, Any]:
        """Close the desktop-action loop after the Qt slot has run."""
        return agent_service.ui_action_result(request)

    @router.post("/v1/agent/approve")
    def agent_approve(request: AgentApproveRequest) -> dict[str, Any]:
        """Approve or reject a pending agent action (write_file, run_command)."""
        return agent_service.approve(request)

    return router


__all__ = [
    "build_agent_router",
]
