"""Unified Agent Service facade for the 3D-Reconstruction AI Assistant.

This module provides the main entry point for Agent execution, delegating to:
- ``ApprovalService`` — Human-in-the-Loop approval workflows
- ``UIActionService`` — Desktop UI action continuation

The facade pattern preserves the existing public interface while enabling
independent testing and evolution of each sub-service (ADR 0004).

The shared run wiring (LLM, tools, supervisor, observability) lives in
:class:`~ai_assistant.agents.context.AgentRunContext`, so this facade only
spells out the arguments that vary per execution (prompt, task, history).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from ..application.coordination import TaskCoordinator
from ..domain.errors import NotFoundError, ServiceUnavailableError
from ..orchestration.specialists.code import (
    CodingTaskContext,
    is_coding_task,
)
from ..orchestration.specialists.code import (
    instruction as coding_instruction,
)
from ..ports import LLMRuntime, PendingActionPort, ToolGateway
from ..tools.factory import create_tool_registry
from ..tools.registry import ToolRegistry
from .approval_service import ApprovalService
from .context import AgentRunContext, require_llm_runtime
from .models import (
    AgentApproveRequest,
    AgentCancelRequest,
    AgentExecuteRequest,
    AgentUiActionResultRequest,
)
from .pending_store import MemoryPendingStore
from .prompts import build_agent_system_prompt
from .ui_action_service import UIActionService

logger = logging.getLogger("ai_assistant.agents.service")


class AgentService:
    """Domain service for full Agent execution lifecycle.

    Facade that delegates to:
    - ``ApprovalService`` for HITL approval workflows
    - ``UIActionService`` for desktop UI action continuation
    """

    def __init__(self, *, llm_runtime: LLMRuntime | None = None,
                 pending_actions: PendingActionPort | None = None,
                 pending_lock: Any = None, tool_registry: ToolRegistry | None = None,
                 tool_gateway: ToolGateway | None = None,
                 checkpointer: Any = None, a2a_router: Any = None,
                 task_coordinator: TaskCoordinator | None = None,
                 max_concurrent_runs: int = 0,
                 delegate_fn: Callable[..., Any] | None = None) -> None:
        self.llm_runtime = llm_runtime
        # Durability is injected (ADR 0003): the composition root wires the
        # file-backed store; a bare facade (tests, embeds) keeps approvals in
        # memory for the life of the process.
        self.pending_actions = (
            pending_actions if pending_actions is not None else MemoryPendingStore()
        )
        self.pending_lock = pending_lock if pending_lock is not None else threading.Lock()
        # One re-entrant run lock per session id (review finding 3).  Entries
        # live for the process: a pruning cache could build a *second* lock for
        # a session while an older thread is still queued on the first, which
        # is exactly the overlap this lock exists to prevent.  Cost is one tiny
        # lock per distinct session the client has opened.
        self._session_locks: dict[str, threading.RLock] = {}
        # Global run budget (ADR 0024): one run per session is already
        # enforced below, but a deployment can also cap total concurrency so
        # a burst of sessions cannot exhaust the single local model. Zero
        # keeps the facade unlimited (desktop default).
        self._run_slots = (
            threading.BoundedSemaphore(max_concurrent_runs)
            if max_concurrent_runs > 0 else None
        )
        # The composition root owns one catalog and one coordinator; a bare
        # instance (tests, embeds) builds its own so the facade stays usable.
        self.tool_registry = tool_registry if tool_registry is not None else create_tool_registry()
        self.tool_gateway = tool_gateway
        self.task_coordinator = task_coordinator if task_coordinator is not None else TaskCoordinator()
        # LangGraph persistence and the A2A transport are adapter concerns
        # injected by the composition root; ``None`` keeps the facade runnable
        # without them (stateless graphs, local-only dispatch).
        self._checkpointer = checkpointer
        self._a2a_router = a2a_router
        # Durability (ADR 0003): rehydrate approvals/desktop acks persisted by a
        # previous process so cross-restart A2A continuation still resolves.
        self.pending_actions.load()

        # One wiring block for this facade's run, handed to the context and to
        # both sub-services. The A2A router is only used on the semantic-path
        # delegations of the main execute path, so the sub-services get the
        # shared block without it. Adding a dependency is one entry here
        # instead of three call sites (review 2026-10-16 duplication finding).
        shared: dict[str, Any] = {
            "llm_runtime": llm_runtime,
            "pending_actions": self.pending_actions,
            "pending_lock": self.pending_lock,
            "tool_gateway": tool_gateway,
            "tool_registry": self.tool_registry,
            "checkpointer": self._checkpointer,
            "task_coordinator": self.task_coordinator,
        }
        if delegate_fn is not None:
            # The composition root injects a delegate bound to its A2A
            # registry; a bare facade keeps the local-only default.
            shared["delegate_fn"] = delegate_fn
        self._ctx = AgentRunContext(**shared, a2a_router=a2a_router)
        self._approval_service = ApprovalService(**shared)
        self._ui_action_service = UIActionService(**shared)

    def _require_llm(self) -> None:
        require_llm_runtime(self.llm_runtime)

    def _save_pending(self) -> None:
        self.pending_actions.save()

    def _cleanup_pending(self) -> None:
        cutoff = time.time() - 600
        with self.pending_lock:
            if self.pending_actions.cleanup(cutoff):
                self._save_pending()
                logger.info("Cleaned up expired pending agent actions")

    def _session_lock(self, session_id: str) -> threading.RLock:
        """The session's run lock, created on first use.

        ``dict.setdefault`` is a single atomic operation under the GIL, so
        concurrent first runs of the same session can not build two locks.
        """
        return self._session_locks.setdefault(session_id, threading.RLock())

    @contextmanager
    def _model_lease(self) -> Iterator[None]:
        """Pin one model backend for the whole run when the runtime supports it.

        Duck-typed runtimes (test doubles and older adapters) may predate the
        drain contract; they keep the previous behaviour instead of failing.
        """
        lease = getattr(self.llm_runtime, "model_lease", None)
        if callable(lease):
            with lease():
                yield
        else:
            yield

    def execute(self, request: AgentExecuteRequest, *,
                event_sink: Callable[[dict[str, Any]], None] | None = None,
                client_host: str = "unknown") -> dict[str, Any]:
        self._cleanup_pending()
        self._require_llm()
        req_start = time.monotonic()
        task = request.task
        session_id = request.session_id or "agent_default"
        # One run at a time per session (review finding 3): concurrent executes
        # for the same session share one checkpointer thread_id and would
        # clobber the coordinator's running record, so they queue here while
        # other sessions keep running in parallel.  ``cancel``/``approve`` stay
        # outside this lock on purpose — they must work while a run holds it.
        #
        # The global budget is acquired *after* the session lock: taking it
        # first let a run queued behind its own session hold one of the few
        # global slots, so two same-session requests could consume the entire
        # budget and 503 unrelated sessions. Now only a run that can actually
        # execute spends a slot.
        with self._session_lock(session_id):
            slots = self._run_slots
            if slots is not None and not slots.acquire(blocking=False):
                # Backpressure (ADR 0024): refuse immediately (503) instead of
                # queueing behind the single local model; the caller retries.
                msg = ("The agent run budget is exhausted; retry after a running "
                       "run finishes (AGENT_MAX_CONCURRENT_RUNS).")
                raise ServiceUnavailableError(msg)
            try:
                with self._model_lease():
                    return self._execute_locked(
                        request, session_id=session_id, task=task, req_start=req_start,
                        event_sink=event_sink, client_host=client_host,
                    )
            finally:
                if slots is not None:
                    slots.release()

    def _execute_locked(
        self, request: AgentExecuteRequest, *, session_id: str, task: str,
        req_start: float, event_sink: Callable[[dict[str, Any]], None] | None,
        client_host: str,
    ) -> dict[str, Any]:
        """The ``execute`` body, already holding the session's run lock."""
        self.task_coordinator.start(session_id, task=request.task)
        retry_idx = request.retry_message_index

        logger.info("[MODE: AGENT] Task from %s: %s…", client_host,
                    task[:80].replace("\n", " "))

        system_prompt = build_agent_system_prompt(self.tool_registry, language=request.language)
        if is_coding_task(task):
            system_prompt += "\n\n" + coding_instruction(CodingTaskContext(
                task=task, language=request.language, project_root=".",
            ))

        history_messages: list[dict[str, str]] = []
        for entry in request.history:
            role = entry.get("role")
            content_msg = entry.get("content")
            if role in {"user", "assistant"} and isinstance(content_msg, str) and content_msg.strip():
                history_messages.append({"role": role, "content": content_msg[:32000]})

        task_with_attachments = task
        if request.attachments:
            names = [os.path.basename(path) for path in request.attachments]
            task_with_attachments += "\n\n[Attached files: " + ", ".join(names) + "]"

        initial_msgs = [
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": task_with_attachments},
        ]

        # Transport negotiation (JSON vs SSE) belongs to the HTTP adapter; the
        # service returns a plain result and optionally streams steps via event_sink.
        result = self._ctx.run(
            system_prompt=system_prompt,
            task=task_with_attachments,
            session_id=session_id,
            temperature=request.temperature,
            language=request.language,
            request_started=req_start,
            initial_messages=initial_msgs,
            event_sink=event_sink,
        )
        if retry_idx is not None:
            result["retry_message_index"] = retry_idx
        return result

    def cancel(self, request: AgentCancelRequest) -> dict[str, Any]:
        cancelled = self.task_coordinator.cancel(request.session_id, request.request_id)
        if cancelled is None:
            msg = "Unknown or already finished agent task"
            raise NotFoundError(msg)
        return {"status": "cancelled", **cancelled}

    def approve(self, request: AgentApproveRequest) -> dict[str, Any]:
        """Delegate to ApprovalService for HITL approval workflows.

        An approval *resumes the run*, so it re-enters the same per-session
        lock ``execute`` holds: the resumed graph would otherwise share the
        checkpointer ``thread_id`` and the coordinator's record with a
        concurrent execute of the same session — exactly the overlap the lock
        exists to prevent. ``cancel`` stays lock-free: it must answer while a
        run holds one.
        """
        self._cleanup_pending()
        with self._session_lock(request.session_id):
            return self._approval_service.approve(request)

    def ui_action_result(self, request: AgentUiActionResultRequest) -> dict[str, Any]:
        """Delegate to UIActionService for desktop UI action continuation.

        The desktop ack resumes the run too, so it takes the owning session's
        run lock. The request id alone does not carry the session, so the
        parked action is peeked (never popped) under the pending lock first;
        an unknown id still fails inside the sub-service as before.
        """
        self._cleanup_pending()
        with self.pending_lock:
            parked = self.pending_actions.get(request.request_id) or {}
        with self._session_lock(str(parked.get("session_id", ""))):
            return self._ui_action_service.ui_action_result(request)

    def reset_state(self) -> None:
        with self.pending_lock:
            self.pending_actions.clear()
            try:
                if os.path.exists(self.pending_actions.path):
                    os.remove(self.pending_actions.path)
            except OSError as error:
                logger.warning("Unable to remove pending action state: %s", error)


__all__ = [
    "AgentApproveRequest",
    "AgentCancelRequest",
    "AgentExecuteRequest",
    "AgentService",
    "AgentUiActionResultRequest",
]
