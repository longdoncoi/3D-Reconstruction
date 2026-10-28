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
from ..application.memory import MemoryService
from ..domain.errors import ModelNotLoadedError, NotFoundError, ServiceUnavailableError
from ..domain.memory import conversation_turns, format_memory_block, merge_history
from ..domain.parallel import PARALLEL_MAX_DEFAULT
from ..domain.steps import Step, has_step_type
from ..orchestration.specialists.code import (
    CodingTaskContext,
    is_coding_task,
)
from ..orchestration.specialists.code import (
    instruction as coding_instruction,
)
from ..ports import (
    AgentToolCatalog,
    LLMRuntime,
    LockLike,
    PendingActionPort,
    ShortTermMemoryPort,
    ToolGateway,
)
from .approval_service import ApprovalService
from .catalog import default_agent_catalog
from .context import AgentRunContext, require_llm_runtime
from .models import (
    AgentApproveRequest,
    AgentCancelRequest,
    AgentExecuteRequest,
    AgentUiActionResultRequest,
)
from .pending_store import MemoryPendingStore
from .prompts import build_agent_system_prompt
from .session_locks import SessionLockPool
from .ui_action_service import UIActionService

logger = logging.getLogger("ai_assistant.agents.service")


def _latest_final_answer(steps: object) -> str:
    """The run's last non-empty ``final_answer``, or ``""`` when it has none.

    The completed response does not ship the message list (payload size), so
    the short-term buffer rebuilds its assistant turn from the step log the
    response *does* carry. A parked run ends in ``pending_approval``/
    ``pending_ui_action`` and therefore records no answer.
    """
    if not isinstance(steps, list):
        return ""
    for step in reversed(steps):
        if isinstance(step, dict) and has_step_type(step, "final_answer"):
            content = step.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return ""


class AgentService:
    """Domain service for full Agent execution lifecycle.

    Facade that delegates to:
    - ``ApprovalService`` for HITL approval workflows
    - ``UIActionService`` for desktop UI action continuation
    """

    def __init__(self, *, llm_runtime: LLMRuntime | None = None,
                 pending_actions: PendingActionPort | None = None,
                 pending_lock: LockLike | None = None, tool_registry: AgentToolCatalog | None = None,
                 tool_gateway: ToolGateway | None = None,
                 checkpointer: Any = None, a2a_router: Any = None,
                 task_coordinator: TaskCoordinator | None = None,
                 max_concurrent_runs: int = 0,
                 delegate_fn: Callable[..., Any] | None = None,
                 authorise_delegation_fn: Callable[..., tuple[bool, str | None]] | None = None,
                 short_term_memory: ShortTermMemoryPort | None = None,
                 memory_service: MemoryService | None = None,
                 parallel_max: int = PARALLEL_MAX_DEFAULT) -> None:
        self.llm_runtime = llm_runtime
        # Durability is injected (ADR 0003): the composition root wires the
        # file-backed store; a bare facade (tests, embeds) keeps approvals in
        # memory for the life of the process.
        self.pending_actions = (
            pending_actions if pending_actions is not None else MemoryPendingStore()
        )
        self.pending_lock = pending_lock if pending_lock is not None else threading.Lock()
        # One re-entrant run lock per live session id (review finding 3).
        # The pool reference-counts holders *and* queued threads and drops the
        # entry when the last one leaves, so the map is bounded by current
        # concurrency rather than by every session id ever seen.
        self._session_locks = SessionLockPool()
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
        self.tool_registry: AgentToolCatalog = (
            tool_registry if tool_registry is not None else default_agent_catalog()
        )
        self.tool_gateway = tool_gateway
        self.task_coordinator = task_coordinator if task_coordinator is not None else TaskCoordinator()
        # LangGraph persistence and the A2A transport are adapter concerns
        # injected by the composition root; ``None`` keeps the facade runnable
        # without them (stateless graphs, local-only dispatch).
        self._checkpointer = checkpointer
        self._a2a_router = a2a_router
        # Short-term memory (ADR 0043): ``None`` is the off switch — a bare
        # facade and ``AGENT_MEMORY_ENABLED=0`` both keep the old "client
        # ships everything" behaviour with no server-side state at all.
        self._short_term = short_term_memory
        # Long-term memory (ADR 0043): the service already fails closed on a
        # missing store, so ``None`` here is simply the off switch.
        self._memory = memory_service
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
            # ADR 0044: the same fan-out bound applies to the approve/resume
            # paths, which build their own context from this block.
            "parallel_max": parallel_max,
        }
        if delegate_fn is not None:
            # The composition root injects a delegate bound to its A2A
            # registry; a bare facade keeps the local-only default.
            shared["delegate_fn"] = delegate_fn
        if authorise_delegation_fn is not None:
            # The composition root binds the live plugin tool row here; a bare
            # facade keeps the builtin-only fail-closed default.
            shared["authorise_delegation_fn"] = authorise_delegation_fn
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

    def _hold_session(self, session_id: str) -> Any:
        """Serialize work for one session through the reference-counted pool."""
        return self._session_locks.hold(session_id)

    @contextmanager
    def _model_lease(self) -> Iterator[None]:
        """Pin one model backend for the whole run.

        ADR 0032: the drain contract is part of the ``LLMRuntime`` port, so
        it is called directly. It used to be duck-typed with a ``getattr``
        probe: a runtime that forgot the method silently lost the guarantee
        — no type error, no log — and the admin reload could swap the model
        in the middle of a run.

        ADR 0040: every entry point that runs the graph takes this lease
        (``execute``, ``approve``, ``ui_action_result``), not only the fresh
        run. The two resume paths were unguarded, so a reload could free the
        backend under a graph that was already executing.
        """
        runtime = self.llm_runtime
        if runtime is None:
            # ``_require_llm`` refuses the run before this point; the guard
            # is here so the port's contract is not violated on the way out.
            msg = "The LLM runtime is not configured"
            raise ModelNotLoadedError(msg)
        with runtime.model_lease():
            yield

    def _acquire_run_slot(self) -> threading.BoundedSemaphore | None:
        """Take one global run slot or refuse with a retryable 503 (ADR 0024).

        The session lock is always acquired first: a run queued behind its own
        session must not hold one of the few global slots, or two same-session
        requests could consume the entire budget and 503 unrelated sessions.
        Only a run that can actually execute spends a slot.

        ADR 0040 pinned the model lease to every entry point that runs the
        graph; the budget follows the same rule. A resume is a run too, so
        ``approve`` and ``ui_action_result`` take a slot exactly like
        ``execute`` — leaving them outside the cap let a burst of resumes
        exceed ``AGENT_MAX_CONCURRENT_RUNS`` on the single local model.
        """
        slots = self._run_slots
        if slots is not None and not slots.acquire(blocking=False):
            # Backpressure: refuse immediately (503) instead of queueing behind
            # the single local model; the caller retries.
            msg = ("The agent run budget is exhausted; retry after a running "
                   "run finishes (AGENT_MAX_CONCURRENT_RUNS).")
            raise ServiceUnavailableError(msg)
        return slots

    def _release_run_slot(self, slots: threading.BoundedSemaphore | None) -> None:
        """Return a slot on every exit path, including a crashing run."""
        if slots is not None:
            slots.release()

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
        # other sessions keep running in parallel.  ``cancel`` stays outside
        # this lock on purpose — it must answer while a run holds it; approval
        # and the desktop ack resume the run, so they re-enter the same lock.
        #
        # The global budget is acquired *after* the session lock: taking it
        # first let a run queued behind its own session hold one of the few
        # global slots, so two same-session requests could consume the entire
        # budget and 503 unrelated sessions. Now only a run that can actually
        # execute spends a slot.
        with self._hold_session(session_id):
            slots = self._acquire_run_slot()
            try:
                with self._model_lease():
                    return self._execute_locked(
                        request, session_id=session_id, task=task, req_start=req_start,
                        event_sink=event_sink, client_host=client_host,
                    )
            finally:
                self._release_run_slot(slots)

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

        # Short-term memory tops the shipped history up from the buffer: the
        # client's turns are kept as sent, the buffer only adds what the
        # client no longer holds (ADR 0043). Off when nothing is injected.
        short_term = self._short_term
        if short_term is not None:
            history_messages = merge_history(short_term.recent(session_id), history_messages)

        task_with_attachments = task
        if request.attachments:
            names = [os.path.basename(path) for path in request.attachments]
            task_with_attachments += "\n\n[Attached files: " + ", ".join(names) + "]"

        # Long-term memory recall (ADR 0043 §3): the block is appended to the
        # system prompt and the recall is announced on the step log, so a run
        # that was shaped by remembered context is observable. The service
        # fails closed on every path — no store, a throwing store, an
        # unknown kind — so this can only ever add context, never delay or
        # change the outcome of a run.
        memory = self._memory
        initial_steps: list[Step] | None = None
        prior_plan: list[str] | None = None
        if memory is not None and memory.enabled:
            recalled = memory.recall(task_with_attachments)
            memory_block = format_memory_block(recalled)
            if memory_block:
                system_prompt = f"{system_prompt}\n\n{memory_block}"
                initial_steps = [{"type": "memory_recall", "count": len(recalled)}]
            # The procedural half (ADR 0043 §7): an exact-signature plan
            # lookup, independent of the keyword recall above — a task can
            # have a remembered recipe without any matching fact.
            prior_plan = memory.procedural_plan(task_with_attachments)

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
            initial_steps=initial_steps,
            prior_plan=prior_plan,
            event_sink=event_sink,
        )
        if retry_idx is not None:
            result["retry_message_index"] = retry_idx
        # Remember what this run saw — the merged history it started from plus
        # the turn it just produced — so the next request with a shorter
        # transcript still has its context (ADR 0043). The completed response
        # deliberately does not ship the full message list (payload size), so
        # the turns are rebuilt here from what this service already holds; a
        # run that parked or produced no final answer records no answer.
        if short_term is not None:
            turns = list(history_messages)
            turns.append({"role": "user", "content": task_with_attachments})
            answer = _latest_final_answer(result.get("steps"))
            if answer:
                turns.append({"role": "assistant", "content": answer})
            short_term.record(session_id, conversation_turns(turns))
        # Consolidate into long-term memory on success only (ADR 0043 §4): a
        # parked or cancelled run is not worth remembering, and the service
        # contains its own failures — a store problem is logged, never
        # raised back into the response of a run that already succeeded.
        if memory is not None and memory.enabled:
            if result.get("status") == "completed":
                memory.consolidate_run(
                    task_with_attachments, session_id, result.get("steps") or [],
                )
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

        It also takes the model lease and the global run slot for the same
        reason ``execute`` does. The resume *is* a run: leaving it unleased let
        ``/admin/reload-model`` wait only for fresh runs and free the backend
        under a graph that had already started (ADR 0040); leaving it outside
        the budget let a burst of resumes exceed
        ``AGENT_MAX_CONCURRENT_RUNS`` on the single local model (ADR 0024).
        """
        self._cleanup_pending()
        with self._hold_session(request.session_id):
            slots = self._acquire_run_slot()
            try:
                with self._model_lease():
                    return self._approval_service.approve(request)
            finally:
                self._release_run_slot(slots)

    def ui_action_result(self, request: AgentUiActionResultRequest) -> dict[str, Any]:
        """Delegate to UIActionService for desktop UI action continuation.

        The desktop ack resumes the run too, so it takes the owning session's
        run lock, one global run slot and the model lease. The request id alone
        does not carry the session, so the parked action is peeked (never
        popped) under the pending lock first; an unknown id still fails inside
        the sub-service as before.
        """
        self._cleanup_pending()
        with self.pending_lock:
            parked = self.pending_actions.get(request.request_id) or {}
        with self._hold_session(str(parked.get("session_id", ""))):
            slots = self._acquire_run_slot()
            try:
                with self._model_lease():
                    return self._ui_action_service.ui_action_result(request)
            finally:
                self._release_run_slot(slots)

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
