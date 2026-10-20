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
import threading
from time import monotonic
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..application.tasks import TaskService
from ..config.logging import get_agent_logger
from ..domain.tasks import AgentTask
from ..settings import DEFAULT_BASE_URL, a2a_max_subscribers
from .a2a_models import A2AMessage, ResumeTaskParams, SendMessageParams, SendTaskParams
from .a2a_payloads import APPROVAL_RESUME_EXTENSION, build_agent_card, task_payload
from .http.security import TransportPolicy, a2a_guards, resolve_a2a_tenant

# Terminal states that end an SSE stream. The subscription endpoint also ends
# when the task waits for human input so the client can switch to resume.
_TERMINAL_STATUSES = frozenset({"completed", "failed", "canceled", "rejected"})
_SUBSCRIBE_TERMINAL_STATUSES = _TERMINAL_STATUSES | {"input_required"}

_POLL_INTERVAL_SECONDS = 0.1
#: A subscriber must not pin a worker forever. The client can resubscribe with
#: the last event cursor; ending the stream is observable and bounded.
_SUBSCRIBE_MAX_SECONDS = 300.0

_TASK_NOT_FOUND = HTTPException(status_code=404, detail="A2A task not found")
_TASK_NOT_CANCELLABLE = HTTPException(status_code=409, detail="A2A task cannot be cancelled")
_TASK_NOT_RESUMABLE = HTTPException(status_code=409, detail="A2A task cannot be resumed")

#: Upper bound for a task-list page. SQLite treats ``LIMIT -1`` as "no
#: limit", so an unclamped caller could pull the whole task table.
_MAX_TASK_PAGE = 500

logger = get_agent_logger("a2a.router")


def _clamp_limit(value: int) -> int:
    """Bound task-list pagination to ``[1, _MAX_TASK_PAGE]``."""
    return max(1, min(int(value), _MAX_TASK_PAGE))


async def _task_pages(service: TaskService, task_id: str,
                      terminal: frozenset[str],
                      deadline_seconds: float = _SUBSCRIBE_MAX_SECONDS,
                      *, owner: str = "default",
                      ) -> AsyncIterator[tuple[list[dict[str, Any]], AgentTask]]:
    """Poll a task's event log; yield ``(new_events, task)`` until terminal.

    Shared by both SSE endpoints. The cursor is the store's native event
    sequence (SQLite ``rowid``), so each tick fetches only the events added
    since the last page instead of re-reading and re-slicing the whole log
    (O(n) per poll per subscriber). The deadline bounds a task stuck in a
    non-terminal state; the client can resubscribe from its cursor. Ownership
    is checked on every tick, so a stale stream ends if the row disappears.
    """
    cursor = 0
    deadline = monotonic() + max(0.0, deadline_seconds)
    while True:
        try:
            events = list(service.events(task_id, since=cursor, owner=owner))
        except TypeError:
            # Duck-typed service/store without the cursor parameter: replay
            # mode — read the log and slice the already-delivered prefix.
            events = list(service.events(task_id, owner=owner))[cursor:]
        task = service.get(task_id, owner=owner)
        if task is None:
            return
        yield events, task
        if events:
            # ``seq`` is present for cursor-capable stores; count-based stores
            # fall back to "cursor += number of events just delivered".
            cursor = int(events[-1].get("seq", cursor + len(events)))
        if task.status.value in terminal:
            return
        if monotonic() >= deadline:
            logger.warning("A2A subscription for %s hit its deadline; ending stream", task_id)
            return
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)


def _sse_response(stream: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(stream, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


class _SubscriberLimiter:
    """Count concurrent SSE subscribers and refuse admission past the cap.

    The deadline bounds one stream's lifetime; this bounds how many streams can
    exist at once. Without it a client can open enough holds streams to exhaust
    the worker's file descriptors and memory even though each one eventually
    times out.
    """

    def __init__(self, limit: int) -> None:
        self._limit = max(1, limit)
        self._active = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    def try_acquire(self) -> bool:
        with self._lock:
            if self._active >= self._limit:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1


def _admit_subscriber(limiter: _SubscriberLimiter) -> None:
    if not limiter.try_acquire():
        raise HTTPException(
            status_code=503,
            detail="A2A subscriber limit reached",
            headers={"Retry-After": "5"},
        )


async def _releasing(stream: AsyncIterator[str], limiter: _SubscriberLimiter) -> AsyncIterator[str]:
    """Release the subscriber slot however the SSE generator ends (incl. disconnect)."""
    try:
        async for chunk in stream:
            yield chunk
    finally:
        limiter.release()



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


def _submit_message(service: TaskService, params: SendMessageParams, owner: str) -> AgentTask:
    """Submit a new task, or refuse an in-band reply on an existing one.

    Approvals and desktop acknowledgements are single-use capabilities carried
    out-of-band, so a ``task_id`` on the message is rejected with the declared
    resume extension rather than silently accepted.
    """
    message = _text_message(params.message)
    if params.message.task_id:
        existing = service.get(params.message.task_id, owner=owner)
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
                          params.message.context_id, params.classification, owner=owner)


def _dispatch_json_rpc(service: TaskService, method: str, params: dict[str, Any], owner: str) -> dict[str, Any]:
    """Map one JSON-RPC method onto the task service, or raise for the envelope."""
    if method in {"SendMessage", "message/send"}:
        return {"task": task_payload(_submit_message(service, SendMessageParams.model_validate(params), owner))}
    if method == "tasks/send":
        typed = SendTaskParams.model_validate(params)
        submitted = service.submit(typed.capability, _text_message(typed.message),
                                   typed.metadata, typed.context_id, typed.classification, owner=owner)
        return {"task": task_payload(submitted)}
    if method == "tasks/get":
        fetched = service.get(str(params.get("id", "")), owner=owner)
        if fetched is None:
            msg = "A2A task not found"
            raise KeyError(msg)
        return {"task": task_payload(fetched)}
    if method == "tasks/list":
        return {"tasks": [task_payload(task) for task in service.list(
            context_id=params.get("context_id"), limit=_clamp_limit(int(params.get("limit", 100))), owner=owner)]}
    if method == "tasks/cancel":
        cancelled = service.cancel(str(params.get("id", "")), owner=owner)
        if cancelled is None:
            msg = "A2A task cannot be cancelled"
            raise ValueError(msg)
        return {"task": task_payload(cancelled)}
    if method == "tasks/resume":
        resumed = service.resume(str(params.get("id", "")), dict(params.get("input", {})), owner=owner)
        if resumed is None:
            msg = "A2A task cannot be resumed"
            raise ValueError(msg)
        return {"task": task_payload(resumed)}
    raise NotImplementedError(f"Unsupported A2A method: {method}")


# ── Route groups (ADR 0014) ──────────────────────────────────────────────────
# Each group closes over the injected service; the shared helpers above own
# everything that more than one endpoint needs.


def _add_agent_card_route(router: APIRouter, service: TaskService,
                          agent_name: str, agent_version: str,
                          card_base_url: str) -> None:
    @router.get("/.well-known/agent.json")
    def agent_card() -> dict[str, Any]:
        # ADR 0032: the advertised URL is the *configured* base URL, never
        # ``request.base_url`` — that one is derived from the Host header, so a
        # peer sending ``Host: evil.example`` would make this server publish a
        # card pointing at the attacker, and a peer that trusts the card would
        # dispatch its tasks there.
        return build_agent_card(
            agent_name=agent_name,
            agent_version=agent_version,
            base_url=card_base_url.rstrip("/"),
            capabilities=list(service.capabilities),
        )


def _add_send_routes(router: APIRouter, service: TaskService) -> None:
    @router.post("/message:send")
    def send_message(params: SendMessageParams, request: Request) -> dict[str, Any]:
        _require_a2a_version(request)
        owner = request.state.a2a_tenant
        return {"task": task_payload(_submit_message(service, params, owner))}

    @router.post("/a2a/tasks/send")
    def send_task(params: SendTaskParams, request: Request) -> dict[str, Any]:
        owner = request.state.a2a_tenant
        submitted = service.submit(params.capability, _text_message(params.message),
                                   params.metadata, params.context_id, params.classification, owner=owner)
        return {"task": task_payload(submitted)}


def _add_standard_task_routes(router: APIRouter, service: TaskService,
                              limiter: _SubscriberLimiter) -> None:
    """A2A 0.3 endpoints; each requires the declared ``A2A-Version`` header."""

    @router.get("/tasks/{task_id}")
    def standard_get_task(task_id: str, request: Request) -> dict[str, Any]:
        _require_a2a_version(request)
        task = service.get(task_id, owner=request.state.a2a_tenant)
        if task is None:
            raise _TASK_NOT_FOUND
        return {"task": task_payload(task)}

    @router.get("/tasks")
    # ``contextId`` is the A2A query-parameter name, not a local.
    def standard_list_tasks(request: Request, contextId: str | None = None, limit: int = 100) -> dict[str, Any]:  # noqa: N803
        _require_a2a_version(request)
        # ``limit`` is a caller value: clamp it, never hand it to SQL raw.
        return {"tasks": [task_payload(task) for task in service.list(
            context_id=contextId, limit=_clamp_limit(limit), owner=request.state.a2a_tenant)]}

    @router.post("/tasks/{task_id}:cancel")
    def standard_cancel_task(task_id: str, request: Request) -> dict[str, Any]:
        _require_a2a_version(request)
        task = service.cancel(task_id, owner=request.state.a2a_tenant)
        if task is None:
            raise _TASK_NOT_CANCELLABLE
        return {"task": task_payload(task)}

    @router.post("/tasks/{task_id}:resume")
    def standard_resume_task(task_id: str, params: ResumeTaskParams, request: Request) -> dict[str, Any]:
        _require_a2a_version(request)
        task = service.resume(task_id, params.input, owner=request.state.a2a_tenant)
        if task is None:
            raise _TASK_NOT_RESUMABLE
        return {"task": task_payload(task)}

    @router.post("/tasks/{task_id}:subscribe")
    def subscribe_task(task_id: str, request: Request) -> StreamingResponse:
        _require_a2a_version(request)
        owner = request.state.a2a_tenant
        if service.get(task_id, owner=owner) is None:
            raise _TASK_NOT_FOUND
        _admit_subscriber(limiter)

        async def stream() -> AsyncIterator[str]:
            previous = None
            async for _, task in _task_pages(service, task_id, terminal=_SUBSCRIBE_TERMINAL_STATUSES, owner=owner):
                snapshot = task_payload(task)
                if previous is None:
                    yield f"data: {json.dumps({'task': snapshot}, ensure_ascii=False)}\n\n"
                elif snapshot["status"] != previous["status"]:
                    yield f"data: {json.dumps({'statusUpdate': {'taskId': task.id, 'status': snapshot['status']}}, ensure_ascii=False)}\n\n"
                previous = snapshot

        return _sse_response(_releasing(stream(), limiter))


def _add_legacy_task_routes(router: APIRouter, service: TaskService,
                            limiter: _SubscriberLimiter) -> None:
    """The pre-0.3 ``/a2a/tasks/*`` endpoints kept for existing clients."""

    @router.get("/a2a/tasks/{task_id}")
    def get_task(task_id: str, request: Request) -> dict[str, Any]:
        task = service.get(task_id, owner=request.state.a2a_tenant)
        if task is None:
            raise _TASK_NOT_FOUND
        return {"task": task_payload(task)}

    @router.get("/a2a/tasks")
    def list_tasks(request: Request, context_id: str | None = None, limit: int = 100) -> dict[str, Any]:
        """List tasks (legacy surface); ``limit`` is clamped, never trusted."""
        return {"tasks": [task_payload(task) for task in service.list(
            context_id=context_id, limit=_clamp_limit(limit), owner=request.state.a2a_tenant)]}

    @router.post("/a2a/tasks/{task_id}:cancel")
    def cancel_task(task_id: str, request: Request) -> dict[str, Any]:
        task = service.cancel(task_id, owner=request.state.a2a_tenant)
        if task is None:
            raise _TASK_NOT_CANCELLABLE
        return {"task": task_payload(task)}

    @router.post("/a2a/tasks/{task_id}:resume")
    def resume_task(task_id: str, params: ResumeTaskParams, request: Request) -> dict[str, Any]:
        task = service.resume(task_id, params.input, owner=request.state.a2a_tenant)
        if task is None:
            raise _TASK_NOT_RESUMABLE
        return {"task": task_payload(task)}

    @router.get("/a2a/tasks/{task_id}/events")
    def stream_task(task_id: str, request: Request) -> StreamingResponse:
        owner = request.state.a2a_tenant
        if service.get(task_id, owner=owner) is None:
            raise _TASK_NOT_FOUND
        _admit_subscriber(limiter)

        async def stream() -> AsyncIterator[str]:
            async for events, task in _task_pages(service, task_id, terminal=_TERMINAL_STATUSES, owner=owner):
                for event in events:
                    yield f"event: {event['kind']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                if task.status.value in _TERMINAL_STATUSES:
                    yield f"event: done\ndata: {json.dumps(task_payload(task), ensure_ascii=False)}\n\n"
                    return

        return _sse_response(_releasing(stream(), limiter))


def _add_json_rpc_route(router: APIRouter, service: TaskService) -> None:
    @router.post("/a2a")
    async def json_rpc(request: Request) -> dict[str, Any]:
        payload = await request.json()
        request_id = payload.get("id")
        owner = request.state.a2a_tenant
        try:
            result = _dispatch_json_rpc(service, payload.get("method"), payload.get("params", {}), owner)
        except (KeyError, ValueError, NotImplementedError) as error:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": str(error)}}
        else:
            return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _a2a_tenant_dependency(policy: TransportPolicy) -> Any:
    """Build a FastAPI dependency that resolves the A2A tenant for a request."""
    def dependency(request: Request) -> str:
        return resolve_a2a_tenant(request, policy)
    return dependency


def build_a2a_router(service: TaskService, agent_name: str, agent_version: str,
                     policy: TransportPolicy | None = None,
                     max_subscribers: int | None = None,
                     card_base_url: str = DEFAULT_BASE_URL) -> APIRouter:
    """Build the A2A 0.3 HTTP router with all standard and legacy endpoints.

    ``card_base_url`` is the address this server advertises in its Agent Card.
    It is injected by the composition root so the published card never depends
    on the request's Host header (ADR 0032).
    """
    resolved = policy or TransportPolicy()
    limiter = _SubscriberLimiter(
        a2a_max_subscribers() if max_subscribers is None else max_subscribers,
    )
    router = APIRouter(
        tags=["a2a"],
        dependencies=[*a2a_guards(resolved), Depends(_a2a_tenant_dependency(resolved))],
    )
    # Exposed for health introspection and white-box admission tests; the
    # routes close over the same object, so there is exactly one counter.
    setattr(router, "_subscriber_limiter", limiter)
    _add_agent_card_route(router, service, agent_name, agent_version, card_base_url)
    _add_send_routes(router, service)
    _add_standard_task_routes(router, service, limiter)
    _add_legacy_task_routes(router, service, limiter)
    _add_json_rpc_route(router, service)
    return router


__all__ = ["build_a2a_router"]
