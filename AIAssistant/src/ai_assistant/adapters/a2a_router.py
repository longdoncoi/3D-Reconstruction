"""A2A protocol router builder (ADR 0005).

Extracted from ``adapters/a2a.py`` to separate routing logic from payload construction.
Provides: standard A2A 0.3 endpoints, legacy compatibility endpoints, and JSON-RPC handler.

Route registration is grouped by surface (ADR 0014): one function per group —
agent card, message submission, standard 0.3 task routes, legacy ``/a2a/tasks``
routes and the JSON-RPC envelope. Each group owns its own handlers as closures
over the injected :class:`TaskService`, so ``build_a2a_router`` reads as a
composition of those groups instead of a single 47-branch function.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..application.tasks import TaskService
from ..domain.tasks import AgentTask
from .a2a_models import A2AMessage, ResumeTaskParams, SendMessageParams, SendTaskParams
from .a2a_payloads import APPROVAL_RESUME_EXTENSION, build_agent_card, task_payload
from .http.security import TransportPolicy, a2a_guards

# Terminal states that end an SSE stream. The subscription endpoint also ends
# when the task waits for human input so the client can switch to resume.
_TERMINAL_STATUSES = frozenset({"completed", "failed", "canceled", "rejected"})
_SUBSCRIBE_TERMINAL_STATUSES = _TERMINAL_STATUSES | {"input_required"}

_POLL_INTERVAL_SECONDS = 0.1

_TASK_NOT_FOUND = HTTPException(status_code=404, detail="A2A task not found")
_TASK_NOT_CANCELLABLE = HTTPException(status_code=409, detail="A2A task cannot be cancelled")
_TASK_NOT_RESUMABLE = HTTPException(status_code=409, detail="A2A task cannot be resumed")


async def _task_pages(service: TaskService, task_id: str,
                      terminal: frozenset[str]) -> AsyncIterator[tuple[list[dict[str, Any]], AgentTask]]:
    """Poll a task's event log; yield ``(new_events, task)`` until terminal.

    Shared by both SSE endpoints so the polling loop (cursor tracking, terminal
    check, bounded sleep) lives in exactly one place.
    """
    cursor = 0
    while True:
        events = list(service.events(task_id))
        task = service.get(task_id)
        if task is None:
            return
        yield events[cursor:], task
        cursor = len(events)
        if task.status.value in terminal:
            return
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)


def _sse_response(stream: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(stream, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


def _require_a2a_version(request: Request) -> None:
    """Reject anything that is not the declared protocol version (0.3)."""
    requested = request.headers.get("A2A-Version")
    if requested and requested != "0.3":
        raise HTTPException(
            status_code=400,
            detail={"type": "https://a2a-protocol.org/errors/version-not-supported",
                    "supportedVersions": ["0.3"]},
        )


def _text_message(message: A2AMessage) -> str:
    """Concatenate the text parts of an A2A message."""
    return "\n".join(part.text for part in message.parts if part.kind == "text").strip()


def _submit_message(service: TaskService, params: SendMessageParams) -> AgentTask:
    """Submit a new task, or refuse an in-band reply on an existing one.

    Approvals and desktop acknowledgements are single-use capabilities carried
    out-of-band, so a ``task_id`` on the message is rejected with the declared
    resume extension rather than silently accepted.
    """
    message = _text_message(params.message)
    if params.message.task_id:
        existing = service.get(params.message.task_id)
        if existing is None:
            raise _TASK_NOT_FOUND
        if params.message.context_id and params.message.context_id != existing.context_id:
            raise HTTPException(status_code=409, detail="Task context does not match message context")
        raise HTTPException(
            status_code=409,
            detail={"message": "Approval and desktop acknowledgements require the declared resume extension.",
                    "extension": APPROVAL_RESUME_EXTENSION},
        )
    return service.submit(params.capability, message, params.metadata,
                          params.message.context_id, params.classification)


def _dispatch_json_rpc(service: TaskService, method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Map one JSON-RPC method onto the task service, or raise for the envelope."""
    if method in {"SendMessage", "message/send"}:
        return {"task": task_payload(_submit_message(service, SendMessageParams.model_validate(params)))}
    if method == "tasks/send":
        typed = SendTaskParams.model_validate(params)
        submitted = service.submit(typed.capability, _text_message(typed.message),
                                   typed.metadata, typed.context_id, typed.classification)
        return {"task": task_payload(submitted)}
    if method == "tasks/get":
        fetched = service.get(str(params.get("id", "")))
        if fetched is None:
            msg = "A2A task not found"
            raise KeyError(msg)
        return {"task": task_payload(fetched)}
    if method == "tasks/list":
        return {"tasks": [task_payload(task) for task in service.list(
            context_id=params.get("context_id"), limit=int(params.get("limit", 100)))]}
    if method == "tasks/cancel":
        cancelled = service.cancel(str(params.get("id", "")))
        if cancelled is None:
            msg = "A2A task cannot be cancelled"
            raise ValueError(msg)
        return {"task": task_payload(cancelled)}
    if method == "tasks/resume":
        resumed = service.resume(str(params.get("id", "")), dict(params.get("input", {})))
        if resumed is None:
            msg = "A2A task cannot be resumed"
            raise ValueError(msg)
        return {"task": task_payload(resumed)}
    raise NotImplementedError(f"Unsupported A2A method: {method}")


# ── Route groups (ADR 0014) ──────────────────────────────────────────────────
# Each group closes over the injected service; the shared helpers above own
# everything that more than one endpoint needs.


def _add_agent_card_route(router: APIRouter, service: TaskService,
                          agent_name: str, agent_version: str) -> None:
    @router.get("/.well-known/agent.json")
    def agent_card(request: Request) -> dict[str, Any]:
        return build_agent_card(
            agent_name=agent_name,
            agent_version=agent_version,
            base_url=str(request.base_url).rstrip("/"),
            capabilities=list(service.capabilities),
        )


def _add_send_routes(router: APIRouter, service: TaskService) -> None:
    @router.post("/message:send")
    def send_message(params: SendMessageParams, request: Request) -> dict[str, Any]:
        _require_a2a_version(request)
        return {"task": task_payload(_submit_message(service, params))}

    @router.post("/a2a/tasks/send")
    def send_task(params: SendTaskParams) -> dict[str, Any]:
        submitted = service.submit(params.capability, _text_message(params.message),
                                   params.metadata, params.context_id, params.classification)
        return {"task": task_payload(submitted)}


def _add_standard_task_routes(router: APIRouter, service: TaskService) -> None:
    """A2A 0.3 endpoints; each requires the declared ``A2A-Version`` header."""

    @router.get("/tasks/{task_id}")
    def standard_get_task(task_id: str, request: Request) -> dict[str, Any]:
        _require_a2a_version(request)
        task = service.get(task_id)
        if task is None:
            raise _TASK_NOT_FOUND
        return {"task": task_payload(task)}

    @router.get("/tasks")
    # ``contextId`` is the A2A query-parameter name, not a local.
    def standard_list_tasks(request: Request, contextId: str | None = None, limit: int = 100) -> dict[str, Any]:  # noqa: N803
        _require_a2a_version(request)
        return {"tasks": [task_payload(task) for task in service.list(context_id=contextId, limit=limit)]}

    @router.post("/tasks/{task_id}:cancel")
    def standard_cancel_task(task_id: str, request: Request) -> dict[str, Any]:
        _require_a2a_version(request)
        task = service.cancel(task_id)
        if task is None:
            raise _TASK_NOT_CANCELLABLE
        return {"task": task_payload(task)}

    @router.post("/tasks/{task_id}:resume")
    def standard_resume_task(task_id: str, params: ResumeTaskParams, request: Request) -> dict[str, Any]:
        _require_a2a_version(request)
        task = service.resume(task_id, params.input)
        if task is None:
            raise _TASK_NOT_RESUMABLE
        return {"task": task_payload(task)}

    @router.post("/tasks/{task_id}:subscribe")
    def subscribe_task(task_id: str, request: Request):
        _require_a2a_version(request)
        if service.get(task_id) is None:
            raise _TASK_NOT_FOUND

        async def stream():
            previous = None
            async for _, task in _task_pages(service, task_id, terminal=_SUBSCRIBE_TERMINAL_STATUSES):
                snapshot = task_payload(task)
                if previous is None:
                    yield f"data: {json.dumps({'task': snapshot}, ensure_ascii=False)}\n\n"
                elif snapshot["status"] != previous["status"]:
                    yield f"data: {json.dumps({'statusUpdate': {'taskId': task.id, 'status': snapshot['status']}}, ensure_ascii=False)}\n\n"
                previous = snapshot

        return _sse_response(stream())


def _add_legacy_task_routes(router: APIRouter, service: TaskService) -> None:
    """The pre-0.3 ``/a2a/tasks/*`` endpoints kept for existing clients."""

    @router.get("/a2a/tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, Any]:
        task = service.get(task_id)
        if task is None:
            raise _TASK_NOT_FOUND
        return {"task": task_payload(task)}

    @router.get("/a2a/tasks")
    def list_tasks(context_id: str | None = None, limit: int = 100) -> dict[str, Any]:
        return {"tasks": [task_payload(task) for task in service.list(context_id=context_id, limit=limit)]}

    @router.post("/a2a/tasks/{task_id}:cancel")
    def cancel_task(task_id: str) -> dict[str, Any]:
        task = service.cancel(task_id)
        if task is None:
            raise _TASK_NOT_CANCELLABLE
        return {"task": task_payload(task)}

    @router.post("/a2a/tasks/{task_id}:resume")
    def resume_task(task_id: str, params: ResumeTaskParams) -> dict[str, Any]:
        task = service.resume(task_id, params.input)
        if task is None:
            raise _TASK_NOT_RESUMABLE
        return {"task": task_payload(task)}

    @router.get("/a2a/tasks/{task_id}/events")
    def stream_task(task_id: str):
        if service.get(task_id) is None:
            raise _TASK_NOT_FOUND

        async def stream():
            async for events, task in _task_pages(service, task_id, terminal=_TERMINAL_STATUSES):
                for event in events:
                    yield f"event: {event['kind']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                if task.status.value in _TERMINAL_STATUSES:
                    yield f"event: done\ndata: {json.dumps(task_payload(task), ensure_ascii=False)}\n\n"
                    return

        return _sse_response(stream())


def _add_json_rpc_route(router: APIRouter, service: TaskService) -> None:
    @router.post("/a2a")
    async def json_rpc(request: Request) -> dict[str, Any]:
        payload = await request.json()
        request_id = payload.get("id")
        try:
            result = _dispatch_json_rpc(service, payload.get("method"), payload.get("params", {}))
        except (KeyError, ValueError, NotImplementedError) as error:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": str(error)}}
        else:
            return {"jsonrpc": "2.0", "id": request_id, "result": result}


def build_a2a_router(service: TaskService, agent_name: str, agent_version: str,
                     policy: TransportPolicy | None = None) -> APIRouter:
    """Build the A2A 0.3 HTTP router with all standard and legacy endpoints."""
    router = APIRouter(tags=["a2a"], dependencies=a2a_guards(policy))
    _add_agent_card_route(router, service, agent_name, agent_version)
    _add_send_routes(router, service)
    _add_standard_task_routes(router, service)
    _add_legacy_task_routes(router, service)
    _add_json_rpc_route(router, service)
    return router


__all__ = ["build_a2a_router"]
