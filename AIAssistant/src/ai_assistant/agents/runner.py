"""Core LangGraph agent runner.

The run signature splits in two (ADR 0013): everything that does not vary
between invocations is a frozen :class:`~ai_assistant.agents.deps.RunDeps`
bundle, and this function only accepts the per-invocation arguments (prompt,
task, resume state, approval grant, event sink).

The body was still a single 30-branch function: the tool-execution path and
the state-to-response mapping each carried most of that branching (ADR 0014).
Both are module-level functions now, so ``run_langgraph_agent`` reads as
preparation → graph construction → mapping.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from ai_assistant.agents.completion import (
    CRITIC_JSON_SCHEMA,
    PLANNER_JSON_SCHEMA,
    constrained_completion,
    parse_tool_call,
    structured_completion,
)
from ai_assistant.domain.errors import ServiceUnavailableError
from ai_assistant.domain.governance import has_sensitive_keys
from ai_assistant.orchestration.state import CONTEXT_EXHAUSTED_MESSAGE, has_step_type
from ai_assistant.tools.action_manifest import validate_action_params
from ai_assistant.tools.tool_contract import RUNTIME_UNCONFIGURED_RESULT, validate_tool_call

from .deps import RunDeps
from .ids import generate_action_id

logger = logging.getLogger("ai_assistant.agents.runner")

_AGENT_MAX_ITERATIONS = 12


def _resolve_action_id(generate_action_id_fn: Callable[[], str] | None) -> str:
    """Use the injected id factory, else the canonical generator."""
    return generate_action_id_fn() if generate_action_id_fn else generate_action_id()


def _persist_pending(deps: RunDeps, save_pending_fn: Callable[[], None] | None) -> None:
    """Flush the pending store, via the injected hook when one is wired."""
    if save_pending_fn:
        save_pending_fn()
    else:
        deps.pending_actions.save()


def _remote_dispatch(
    *,
    deps: RunDeps,
    delegation: Any,
    task: str,
    session_id: str,
    tool_name: str,
    params: dict,
    approval_token: str,
) -> dict:
    """Egress policy, then the gateway gate, then the transport.

    The remote leg never reaches the local executor, so nothing about it is
    left to chance: a payload carrying a credential is refused outright —
    masking it would hand the peer a falsified payload to run, and sending it
    as-is would put the secret on the wire — and every other call clears the
    same validation, scope, classification and single-use approval grant the
    local path clears, spending that grant, before it leaves (ADR 0002/0010).
    """
    if has_sensitive_keys(params):
        return {
            "success": False,
            "error_code": "data_policy_denied",
            "error": "Tham số nhạy cảm không được gửi đến agent A2A từ xa.",
        }
    gate = (deps.tool_gateway.preflight(tool_name, params, approval_token)
            if deps.tool_gateway is not None
            else dict(RUNTIME_UNCONFIGURED_RESULT))
    if not gate.get("success"):
        return gate
    remote_payload = {"tool": tool_name, "parameters": params,
                      "session_id": session_id,
                      "idempotency_key": delegation.idempotency_key}
    with deps.span_fn("agent.a2a_delegate", tool=tool_name):
        result = deps.a2a_router.route(delegation.specialist.value, task, remote_payload)
    deps.audit_agent_fn("tool_transport", delegation, source=result.get("source", "local"))
    return result


def _execute_tool(
    *,
    deps: RunDeps,
    task: str,
    session_id: str,
    tool_name: str,
    params: dict,
    tools_requiring_approval: set[str],
    approval_token: str,
    action_id_fn: Callable[[], str],
    delegate_fn: Callable[[str, dict], Any],
) -> dict:
    """Supervisor authorisation, UI-action canonicalisation, then one gateway call.

    ADR 0002/0003: ``ToolExecutionService`` is the only execution path, reached
    through the injected gateway. When no gateway is wired we surface a
    structured error instead of calling the registry directly and bypassing
    policy checks. The remote A2A leg is no exception: it clears the same
    gateway through :meth:`~ai_assistant.ports.ToolGateway.preflight` before
    dispatching, so validation, scope, classification and the approval grant
    cannot be skipped by leaving the process.
    """
    delegation = delegate_fn(tool_name, params)
    allowed, reason = deps.authorise_delegation_fn(delegation, tool_name in tools_requiring_approval)
    if not allowed:
        deps.audit_agent_fn("tool_denied", delegation, reason=reason)
        return {"error": reason or "Tool call denied by supervisor policy."}
    if tool_name == "application_action":
        canonical_params, error = validate_action_params(params)
        if error is not None or canonical_params is None:
            return {"error": error or "Invalid UI action parameters"}
        canonical_params["request_id"] = action_id_fn() or generate_action_id()
        params.clear()
        params.update(canonical_params)
    if tool_name == "_validation_error":
        return {"error": f"Lỗi xác thực tham số tool '{params.get('tool')}': {params.get('error')}"}
    spec = deps.tool_registry.get(tool_name)
    if spec is None or spec.handler is None:
        return {"error": f"Tool không tồn tại hoặc không có handler: {tool_name}"}

    tool_started = time.monotonic()
    if delegation.remote_endpoint and deps.a2a_router is not None:
        result = _remote_dispatch(deps=deps, delegation=delegation, task=task,
                                  session_id=session_id, tool_name=tool_name,
                                  params=params, approval_token=approval_token)
    else:
        with deps.span_fn("agent.tool", tool=tool_name, session_id=session_id):
            if deps.tool_gateway is None:
                result = dict(RUNTIME_UNCONFIGURED_RESULT)
            else:
                # Only a single-use grant unlocks an approval-gated tool;
                # ``approval_granted`` alone no longer reaches execution.
                result = (deps.tool_gateway.execute_approved(tool_name, params, approval_token)
                          if approval_token
                          else deps.tool_gateway.execute(tool_name, params))
    deps.audit_agent_fn("tool_completed", delegation, success="error" not in result)
    deps.record_tool_fn(tool_name, "error" not in result, time.monotonic() - tool_started)
    return result


def _pending_status(state: dict) -> str:
    """Map the parked-action state onto the response status."""
    pending_state = state.get("pending_tool") or {}
    status = ("pending_ui_action" if pending_state.get("ui_ack")
              else "pending_approval" if pending_state
              else "completed")
    return "cancelled" if state.get("cancelled") else status


def _finish_task(deps: RunDeps, session_id: str, state: dict) -> None:
    """Tell the coordinator whether the run is parked waiting or finished."""
    pending_state = state.get("pending_tool") or {}
    if pending_state:
        deps.task_coordinator.update(
            session_id,
            status="waiting_ui_ack" if pending_state.get("ui_ack") else "waiting_approval",
            iterations=state.get("iteration", 0),
        )
        return
    deps.task_coordinator.finish(session_id, success=not state.get("cancelled"),
                                 iterations=state.get("iteration", 0))


def _close_trace(ctx: dict, ctx_mgr: Any, pending_status: str, state: dict, total_ms: int) -> None:
    """Record the run outcome on the LangSmith trace and close it."""
    ctx["outputs"] = {"status": pending_status, "iterations": state.get("iteration", 0),
                      "duration_ms": total_ms, "step_count": len(state.get("steps", []))}
    ctx_mgr.__exit__(None, None, None)


def _store_pending_action(
    *,
    deps: RunDeps,
    save_pending_fn: Callable[[], None] | None,
    session_id: str,
    task: str,
    state: dict,
    temperature: float,
    language: str,
    action_id: str,
    request_id: str,
    is_ui_ack: bool,
) -> None:
    """Persist the parked action so approval/UI-ACK resume survives a restart."""
    pending = state["pending_tool"]
    with deps.pending_lock:
        deps.pending_actions[request_id if is_ui_ack else action_id] = {
            "tool": pending["tool"],
            "params": pending["params"],
            "session_id": session_id,
            "task": task,
            # Snapshot the live state lists: the graph keeps appending to them
            # after this entry is parked, and the store re-serialises whatever it
            # holds on the next save — aliasing made the persisted entry change
            # after the fact (finding 9).
            "messages": list(state["messages"]),
            "steps": list(state["steps"]),
            "iteration": state["iteration"],
            "temperature": temperature,
            "language": language,
            "created_at": time.time(),
            "ui_ack": is_ui_ack,
            "approval_scope": pending.get("approval_scope", ""),
            "approval_preview": pending.get("approval_preview"),
        }
        _persist_pending(deps, save_pending_fn)


def _pending_response(
    *,
    deps: RunDeps,
    session_id: str,
    task: str,
    state: dict,
    temperature: float,
    language: str,
    request_started: float,
    prior_step_count: int,
    action_id_fn: Callable[[], str],
    save_pending_fn: Callable[[], None] | None,
) -> dict | None:
    """The two shapes a parked action returns, or ``None`` when nothing is parked."""
    pending = state.get("pending_tool")
    if not pending:
        return None
    steps = state["steps"]
    # A resumed run swaps in a no-op id generator (RESUME_ACTION_ID), but the
    # action parked *here* is new and has no id yet. Returning "" would hand the
    # client an id the approval boundary refuses (an empty action_id is
    # rejected on the way back in) and would key the store on "", stranding the
    # run — so mint one, the same way the application_action request id does.
    action_id = action_id_fn() or generate_action_id()
    is_ui_ack = bool(pending.get("ui_ack"))
    request_id = pending["params"].get("request_id", action_id)
    _store_pending_action(
        deps=deps, save_pending_fn=save_pending_fn, session_id=session_id, task=task,
        state=state, temperature=temperature, language=language,
        action_id=action_id, request_id=request_id, is_ui_ack=is_ui_ack,
    )
    total_ms = round((time.monotonic() - request_started) * 1000)
    if is_ui_ack:
        return {
            "status": "pending_ui_action", "session_id": session_id,
            "steps": steps, "request_id": request_id,
            "ui_action": {"request_id": request_id, "action": pending["params"]["action"],
                          "params": pending["params"]},
            "total_ms": total_ms,
        }
    steps.append({
        "type": "pending_approval",
        "action_id": action_id,
        "tool": pending["tool"],
        "params": pending["params"],
        "description": pending["params"].get("description", f"Thực thi {pending['tool']}"),
        "approval_scope": pending.get("approval_scope", ""),
        "preview": pending.get("approval_preview"),
    })
    return {
        "status": "pending_approval", "session_id": session_id,
        "steps": steps, "action_id": action_id,
        "prior_step_count": prior_step_count,
        "total_ms": total_ms,
    }


def _completed_response(*, pending_status: str, session_id: str, state: dict,
                        steps: list, prior_step_count: int, request_started: float) -> dict:
    """The terminal response, guaranteeing one ``final_answer`` step exists."""
    if not any(has_step_type(step, "final_answer") for step in steps):
        steps.append({"type": "final_answer", "content": "Agent đã kết thúc mà chưa có kết luận."})
    return {
        "status": pending_status, "session_id": session_id, "steps": steps,
        "prior_step_count": prior_step_count,
        "iterations": state["iteration"],
        "total_ms": round((time.monotonic() - request_started) * 1000),
    }


def _approval_cover_probe(gateway, approval_token: str):
    """Build the single-use grant probe for the graph, if one applies.

    The reason node asks again as soon as the grant for the approved
    invocation is spent, instead of looping on a tool the gateway will
    refuse. No token or no covering gateway means no probe.
    """
    if not approval_token or gateway is None or not hasattr(gateway, "approval_covers"):
        return None

    def _covers(tool_name: str, tool_params: dict) -> bool:
        return bool(gateway.approval_covers(tool_name, tool_params, approval_token))

    return _covers


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
    initial_steps: list[dict] | None = None,
    initial_iteration: int = 0,
    resume_with_reflection: bool = False,
    event_sink: Callable[[dict], None] | None = None,
    prior_step_count: int = 0,
    approval_granted: bool = False,
    approval_scope: str = "",
    approval_token: str = "",
    granted_fingerprint: str = "",
    # Per-invocation validators/hooks: resume continuations swap these for the
    # named no-op helpers in ``agents.context``.
    validate_tool_call_fn: Callable = validate_tool_call,
    generate_action_id_fn: Callable[[], str] | None = None,
) -> dict:
    """Run the tool loop via LangGraph while keeping Qt API response shape."""
    if deps.local_agent_graph is None:
        msg = "LangGraph is required for Agent mode. Run: pip install -r AIAssistant/requirements.txt"
        raise ServiceUnavailableError(msg)

    tool_registry = deps.tool_registry
    openai_tools = tool_registry.get_openai_tools()
    grammar_schema = tool_registry.grammar
    tools_requiring_approval = {spec.name for spec in tool_registry.get_all() if spec.requires_approval}

    def _action_id() -> str:
        return _resolve_action_id(generate_action_id_fn)

    # ── Inner callbacks for LangGraph ────────────────────────────────────────

    def complete(messages: list[dict], current_temp: float) -> str:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = int(total_chars / deps.chars_per_token)
        if estimated_tokens >= deps.llm_n_ctx - 512:
            return CONTEXT_EXHAUSTED_MESSAGE
        max_tokens = min(2048, max(512, deps.llm_n_ctx - estimated_tokens - 400))
        logger.info("LangGraph calls LLM (%d msgs, ~%d tokens)", len(messages), estimated_tokens)
        answer = constrained_completion(
            messages, max_tokens, current_temp,
            deps.llm_runtime, deps.backend_mode_fn, deps.openai_compatible_fn,
            openai_tools, grammar_schema, deps.record_token_usage_fn
        )
        logger.info("LangGraph LLM output (%d chars): %s", len(answer), answer[:120].replace("\n", " "))
        return answer

    def execute(tool_name: str, params: dict) -> dict:
        return _execute_tool(
            deps=deps, task=task, session_id=session_id, tool_name=tool_name, params=params,
            tools_requiring_approval=tools_requiring_approval,
            approval_token=approval_token, action_id_fn=_action_id,
            delegate_fn=_delegation_for,
        )

    _delegation_cache: dict[str, Any] = {}

    def _delegation_for(tool_name: str, params: dict) -> Any:
        """One supervisor classification per distinct call (review finding 14).

        Dispatch, execution, verification and reflection all ask for the same
        delegation of the same (tool, params); recomputing it cost a payload
        dump, a SHA-256, a remote-endpoint lookup and the ~50-term coding
        heuristic on every step of that pipeline."""
        key = json.dumps({"tool": tool_name, "params": params},
                         ensure_ascii=False, sort_keys=True, default=str)
        cached = _delegation_cache.get(key)
        if cached is None:
            cached = deps.delegate_fn(task, session_id, tool_name, params,
                                      prefer_code=deps.is_coding_task_fn(task))
            _delegation_cache[key] = cached
        return cached

    def select_specialist(tool_name: str, params: dict) -> dict:
        if tool_name == "_validation_error":
            return {}
        delegation = _delegation_for(tool_name, params)
        deps.audit_agent_fn("tool_delegated", delegation)
        return {
            "specialist": str(delegation.specialist),
            "idempotency_key": delegation.idempotency_key,
            "remote_endpoint": delegation.remote_endpoint,
        }

    def verify_tool_result(tool_name: str, params: dict, result: dict) -> dict:
        if tool_name == "_validation_error":
            return {"passed": False, "reason": result.get("error", "Validation error")}
        delegation = _delegation_for(tool_name, params)
        verification = deps.verify_result_fn(delegation, result)
        deps.audit_agent_fn("tool_verified", delegation, **verification)
        return verification

    def deterministic_reflection(tool_name: str, params: dict, result: dict, verification: dict) -> dict:
        delegation = _delegation_for(tool_name, params)
        reflection = deps.reflect_result_fn(delegation, result, verification)
        deps.audit_agent_fn("tool_reflected", delegation, **reflection)
        return reflection

    logger.info("Starting LangGraph session %s", session_id)

    # ── LangSmith tracing ────────────────────────────────────────────────────
    _ls_ctx_mgr = deps.langsmith_trace_fn(
        "agent.session", run_type="chain",
        inputs={"task": task[:200], "session_id": session_id,
                "supervisor_route": "supervisor"},
        metadata={"session_id": session_id, "temperature": temperature, "language": language},
    )
    _ls_ctx = _ls_ctx_mgr.__enter__()

    # Single-use approval grant: the reason node asks again as soon as the
    # grant for the approved invocation is spent, instead of looping on a
    # tool the gateway will refuse.
    approval_covers_fn = _approval_cover_probe(deps.tool_gateway, approval_token)

    graph = deps.local_agent_graph(
        complete=complete,
        parse=lambda text: parse_tool_call(text, tool_registry.models, validate_tool_call_fn,
                                           deps.record_schema_error_fn),
        execute=execute,
        needs_approval=lambda tn: tn in tools_requiring_approval,
        max_iterations=_AGENT_MAX_ITERATIONS,
        emit=event_sink,
        select_specialist=select_specialist,
        approval_covers=approval_covers_fn,
        verify_result=verify_tool_result,
        reflect_result=deterministic_reflection,
        plan_complete=lambda msgs, temp: structured_completion(
            msgs, 512, temp, PLANNER_JSON_SCHEMA, deps.llm_runtime,
            deps.backend_mode_fn, deps.openai_compatible_fn),
        reflect_complete=lambda msgs, temp: structured_completion(
            msgs, 512, temp, CRITIC_JSON_SCHEMA, deps.llm_runtime,
            deps.backend_mode_fn, deps.openai_compatible_fn),
        plan_reflect_complete=lambda msgs, temp: structured_completion(
            msgs, 512, temp, CRITIC_JSON_SCHEMA, deps.llm_runtime,
            deps.backend_mode_fn, deps.openai_compatible_fn),
        cancel_checker=lambda: deps.task_coordinator.is_cancelled(session_id),
        checkpointer=deps.checkpointer,
    )

    messages = initial_messages or [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    try:
        state = graph.run(
            messages, session_id, temperature, initial_steps, initial_iteration,
            resume_with_reflection=resume_with_reflection,
            required_ui_actions=[],
            enforce_plan_completion=True,
            approval_granted=approval_granted or bool(
                (initial_steps or []) and
                any(has_step_type(step, "approval_granted") for step in (initial_steps or []))
            ),
            approval_scope=approval_scope,
            granted_fingerprint=granted_fingerprint,
        )
    except BaseException as error:
        _ls_ctx["outputs"] = {"status": "error", "error": str(error)}
        _ls_ctx_mgr.__exit__(type(error), error, error.__traceback__)
        # Nothing below runs on this path, so without this call a crash left
        # the coordinator's record parked in ``running`` forever (review
        # finding 6).  ``finish`` still resolves cancellation first, so a run
        # aborted by cancel reports ``cancelled``, not ``failed``.
        deps.task_coordinator.finish(session_id, success=False, error=str(error))
        raise

    # ── State → response ─────────────────────────────────────────────────────
    pending_status = _pending_status(state)
    _finish_task(deps, session_id, state)
    total_ms = round((time.monotonic() - request_started) * 1000)
    _close_trace(_ls_ctx, _ls_ctx_mgr, pending_status, state, total_ms)

    parked = _pending_response(
        deps=deps, session_id=session_id, task=task, state=state,
        temperature=temperature, language=language, request_started=request_started,
        prior_step_count=prior_step_count, action_id_fn=_action_id,
        save_pending_fn=deps.save_pending_fn,
    )
    if parked is not None:
        return parked
    return _completed_response(
        pending_status=pending_status, session_id=session_id, state=state,
        steps=state["steps"], prior_step_count=prior_step_count, request_started=request_started,
    )


__all__ = ["run_langgraph_agent"]
