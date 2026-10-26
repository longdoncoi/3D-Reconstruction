"""Core LangGraph agent runner.

The run signature splits in two (ADR 0013): everything that does not vary
between invocations is a frozen :class:`~ai_assistant.agents.deps.RunDeps`
bundle, and this function only accepts the per-invocation arguments (prompt,
task, resume state, approval grant, event sink).

The graph wiring was extracted to :mod:`ai_assistant.agents.graph_wiring`
(ADR 0013 follow-up): ``build_run_graph`` owns the callbacks and the
``local_agent_graph`` call.  The response mapping was extracted to
:mod:`ai_assistant.agents.response`: ``_pending_response`` owns the parked
action shapes, ``_completed_response`` owns the terminal tail.  The public
function now reads as guard → wiring → trace → run → map with no nested
``def`` at all, and each step is nameable in tests.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, cast

from ai_assistant.domain.errors import ServiceUnavailableError
from ai_assistant.orchestration.state import Step, has_step_type

from .deps import RunDeps
from .graph_wiring import _action_id_factory, build_run_graph
from .response import (
    _close_trace,
    _completed_response,
    _finish_task,
    _pending_response,
    _pending_status,
)

logger = logging.getLogger("ai_assistant.agents.runner")


def _run_graph(
    graph: Any,
    *,
    deps: RunDeps,
    session_id: str,
    messages: list[dict[str, Any]],
    temperature: float,
    initial_steps: list[Step] | None,
    prior_plan: list[str] | None,
    initial_iteration: int,
    resume_with_reflection: bool,
    approval_granted: bool,
    approval_scope: str,
    granted_fingerprint: str,
    ls_ctx: dict[str, Any],
    ls_ctx_mgr: Any,
) -> dict[str, Any]:
    """Execute the graph, closing the trace and the task record on failure.

    A run resumed from an approval carries its grant as an ``approval_granted``
    step, so the boolean is widened by that step's presence and the resumed leg
    keeps the authority the human signed (ADR 0010).
    """
    try:
        return cast(dict[str, Any], graph.run(
            messages, session_id, temperature, initial_steps, initial_iteration,
            resume_with_reflection=resume_with_reflection,
            required_ui_actions=[],
            enforce_plan_completion=True,
            prior_plan=prior_plan,
            approval_granted=approval_granted or bool(
                (initial_steps or []) and
                any(has_step_type(step, "approval_granted") for step in (initial_steps or []))
            ),
            approval_scope=approval_scope,
            granted_fingerprint=granted_fingerprint,
        ))
    except BaseException as error:
        ls_ctx["outputs"] = {"status": "error", "error": str(error)}
        ls_ctx_mgr.__exit__(type(error), error, error.__traceback__)
        # Nothing below runs on this path, so without this call a crash left
        # the coordinator's record parked in ``running`` forever (review
        # finding 6).  ``finish`` still resolves cancellation first, so a run
        # aborted by cancel reports ``cancelled``, not ``failed``.
        deps.task_coordinator.finish(session_id, success=False, error=str(error))
        raise


def run_langgraph_agent(
    *,
    # Injected wiring that never varies between invocations (ADR 0013)
    deps: RunDeps,
    # Prompt & context
    system_prompt: str,
    task: str,
    session_id: str,
    temperature: float,
    language: str,
    request_started: float,
    # Optional state for resume
    initial_messages: list[dict[str, str]] | None = None,
    initial_steps: list[Step] | None = None,
    prior_plan: list[str] | None = None,
    initial_iteration: int = 0,
    resume_with_reflection: bool = False,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
    prior_step_count: int = 0,
    approval_granted: bool = False,
    approval_scope: str = "",
    approval_token: str = "",
    granted_fingerprint: str = "",
    # Per-invocation validators/hooks: resume continuations swap these for the
    # named no-op helpers in ``agents.context``.
    validate_tool_call_fn: Callable[..., tuple[dict[str, Any] | None, str | None] | None] | None = None,
    generate_action_id_fn: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Run the tool loop via LangGraph while keeping Qt API response shape."""
    if deps.local_agent_graph is None:
        msg = "LangGraph is required for Agent mode. Run: pip install -r AIAssistant/requirements.txt"
        raise ServiceUnavailableError(msg)

    from ..tools.tool_contract import validate_tool_call
    if validate_tool_call_fn is None:
        validate_tool_call_fn = validate_tool_call

    action_id_fn = _action_id_factory(generate_action_id_fn)
    graph = build_run_graph(
        deps=deps, task=task, session_id=session_id, approval_token=approval_token,
        action_id_fn=action_id_fn, validate_tool_call_fn=validate_tool_call_fn,
        event_sink=event_sink,
    )

    logger.info("Starting LangGraph session %s", session_id)

    # ── LangSmith tracing ────────────────────────────────────────────────────
    _ls_ctx_mgr = deps.langsmith_trace_fn(
        "agent.session", run_type="chain",
        inputs={"task": task[:200], "session_id": session_id,
                "supervisor_route": "supervisor"},
        metadata={"session_id": session_id, "temperature": temperature, "language": language},
    )
    _ls_ctx = _ls_ctx_mgr.__enter__()

    messages = initial_messages or [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    state = _run_graph(
        graph, deps=deps, session_id=session_id, messages=messages, temperature=temperature,
        initial_steps=initial_steps, prior_plan=prior_plan, initial_iteration=initial_iteration,
        resume_with_reflection=resume_with_reflection, approval_granted=approval_granted,
        approval_scope=approval_scope, granted_fingerprint=granted_fingerprint,
        ls_ctx=_ls_ctx, ls_ctx_mgr=_ls_ctx_mgr,
    )

    # ── State → response ─────────────────────────────────────────────────────
    pending_status = _pending_status(state)
    _finish_task(deps, session_id, state)
    total_ms = round((time.monotonic() - request_started) * 1000)
    _close_trace(_ls_ctx, _ls_ctx_mgr, pending_status, state, total_ms)

    parked = _pending_response(
        deps=deps, session_id=session_id, task=task, state=state,
        temperature=temperature, language=language, request_started=request_started,
        prior_step_count=prior_step_count, action_id_fn=action_id_fn,
        save_pending_fn=deps.save_pending_fn,
    )
    if parked is not None:
        return parked
    return _completed_response(
        pending_status=pending_status, session_id=session_id, state=state,
        steps=state["steps"], prior_step_count=prior_step_count, request_started=request_started,
    )


__all__ = ["run_langgraph_agent"]
