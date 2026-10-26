"""LangGraph wiring builder — extracted from ``runner.py`` (ADR 0013).

The graph wiring was a wall of ten inner closures around one constructor call
inside ``_build_run_graph``. This module owns that wiring as a pure function of
its arguments: same run, same graph. The public entry point is
:func:`build_run_graph`, which takes the per-run context and returns a ready
to-invoke graph.
"""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from typing import Any, cast

from ..domain.context_budget import (
    CONTEXT_COMPLETION_CAP,
    completion_token_budget,
    context_exhausted,
)
from ..domain.governance import has_sensitive_content
from ..orchestration.state import CONTEXT_EXHAUSTED_MESSAGE
from ..ports import LLMRuntime
from ..tools.action_manifest import validate_action_params
from ..tools.registry import ToolRegistry
from ..tools.tool_contract import RUNTIME_UNCONFIGURED_RESULT
from .deps import RunDeps

logger = logging.getLogger("ai_assistant.agents.graph_wiring")


def _resolve_action_id(generate_action_id_fn: Callable[[], str] | None) -> str:
    """Use the injected id factory, else the canonical generator."""
    from .ids import generate_action_id
    return generate_action_id_fn() if generate_action_id_fn else generate_action_id()


def _remote_dispatch(
    *,
    deps: RunDeps,
    delegation: Any,
    task: str,
    session_id: str,
    tool_name: str,
    params: dict[str, Any],
    approval_token: str,
) -> dict[str, Any]:
    """Egress policy, then the gateway gate, then the transport.

    The remote leg never reaches the local executor, so nothing about it is
    left to chance: a payload carrying a credential is refused outright —
    masking it would hand the peer a falsified payload to run, and sending it
    as-is would put the secret on the wire — and every other call clears the
    same validation, scope, classification and single-use approval grant the
    local path clears, spending that grant, before it leaves (ADR 0002/0010).
    """
    if has_sensitive_content(params):
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
        result: dict[str, Any] = deps.a2a_router.route(delegation.specialist.value, task, remote_payload)
    deps.audit_agent_fn("tool_transport", delegation, source=result.get("source", "local"))
    return result


def _execute_tool(
    *,
    deps: RunDeps,
    task: str,
    session_id: str,
    tool_name: str,
    params: dict[str, Any],
    tools_requiring_approval: set[str],
    approval_token: str,
    action_id_fn: Callable[[], str],
    delegate_fn: Callable[[str, dict[str, Any]], Any],
) -> dict[str, Any]:
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
        canonical_params["request_id"] = action_id_fn() or _resolve_action_id(None)
        params.clear()
        params.update(canonical_params)
    if tool_name == "_validation_error":
        return {"error": f"Lỗi xác thực tham số tool '{params.get('tool')}': {params.get('error')}"}
    spec = deps.tool_registry.get(tool_name)
    if spec is None or spec.handler is None:
        return {"error": f"Tool không tồn tại hoặc không có handler: {tool_name}"}

    import time
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
                result = (deps.tool_gateway.execute_approved(tool_name, params, approval_token)
                          if approval_token
                          else deps.tool_gateway.execute(tool_name, params))
    deps.audit_agent_fn("tool_completed", delegation, success="error" not in result)
    deps.record_tool_fn(tool_name, "error" not in result, time.monotonic() - tool_started)
    return result


def _approval_cover_probe(
    gateway: Any, approval_token: str
) -> Callable[[str, dict[str, Any]], bool] | None:
    """Build the single-use grant probe for the graph, if one applies.

    The reason node asks again as soon as the grant for the approved
    invocation is spent, instead of looping on a tool the gateway will
    refuse. No token or no covering gateway means no probe.
    """
    if not approval_token or gateway is None or not hasattr(gateway, "approval_covers"):
        return None

    def _covers(tool_name: str, tool_params: dict[str, Any]) -> bool:
        return bool(gateway.approval_covers(tool_name, tool_params, approval_token))

    return _covers


def _action_id_factory(generate_action_id_fn: Callable[[], str] | None) -> Callable[[], str]:
    """Bind the injected id factory (or the canonical generator) into a callable."""
    def _action_id() -> str:
        return _resolve_action_id(generate_action_id_fn)
    return _action_id


def build_run_graph(
    *,
    deps: RunDeps,
    task: str,
    session_id: str,
    approval_token: str,
    action_id_fn: Callable[[], str],
    validate_tool_call_fn: Callable[..., tuple[dict[str, Any] | None, str | None] | None],
    event_sink: Callable[[dict[str, Any]], None] | None,
) -> Any:
    """Wire the LangGraph callbacks onto the engine and build the graph.

    The ten callbacks close over per-run views of the tool catalog (OpenAI
    shape, grammar, approval-gated set) and over one supervisor-delegation memo
    per run, so the wiring is a pure function of its arguments: same run, same
    graph.  The single-use approval grant probe lives here too — the reason
    node asks again as soon as the grant for the approved invocation is spent,
    instead of looping on a tool the gateway will refuse (ADR 0010).
    """
    tool_registry = deps.tool_registry
    llm_runtime = cast(LLMRuntime, deps.llm_runtime)
    openai_tools = tool_registry.get_openai_tools()
    grammar_schema = tool_registry.grammar
    tools_requiring_approval = {spec.name for spec in tool_registry.get_all() if spec.requires_approval}

    def complete(messages: list[dict[str, Any]], current_temp: float) -> str:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = int(total_chars / deps.chars_per_token)
        # One domain policy with the HTTP chat route: refuse the same way,
        # budget the completion the same way (domain/context_budget.py).
        if context_exhausted(estimated_tokens, deps.llm_n_ctx):
            return CONTEXT_EXHAUSTED_MESSAGE
        max_tokens = min(
            CONTEXT_COMPLETION_CAP,
            completion_token_budget(deps.llm_n_ctx, estimated_tokens),
        )
        logger.info("LangGraph calls LLM (%d msgs, ~%d tokens)", len(messages), estimated_tokens)
        from .completion import constrained_completion
        answer = constrained_completion(
            messages, max_tokens, current_temp,
            llm_runtime, deps.backend_mode_fn, deps.openai_compatible_fn,
            openai_tools, grammar_schema, deps.record_token_usage_fn
        )
        logger.info("LangGraph LLM output (%d chars): %s", len(answer), answer[:120].replace("\n", " "))
        return answer

    def execute(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        return _execute_tool(
            deps=deps, task=task, session_id=session_id, tool_name=tool_name, params=params,
            tools_requiring_approval=tools_requiring_approval,
            approval_token=approval_token, action_id_fn=action_id_fn,
            delegate_fn=_delegation_for,
        )

    _delegation_cache: dict[str, Any] = {}
    # Parallel branches (ADR 0044 §6) reach this cache concurrently: without
    # the lock, two branches of one group can both miss, both pay for the same
    # classification and overwrite each other's entry.
    _delegation_lock = threading.Lock()

    def _delegation_for(tool_name: str, params: dict[str, Any]) -> Any:
        """One supervisor classification per distinct call (review finding 14).

        Dispatch, execution, verification and reflection all ask for the same
        delegation of the same (tool, params); recomputing it cost a payload
        dump, a SHA-256, a remote-endpoint lookup and the ~50-term coding
        heuristic on every step of that pipeline."""
        key = json.dumps({"tool": tool_name, "params": params},
                         ensure_ascii=False, sort_keys=True, default=str)
        with _delegation_lock:
            cached = _delegation_cache.get(key)
            if cached is None:
                cached = deps.delegate_fn(task, session_id, tool_name, params,
                                          prefer_code=deps.is_coding_task_fn(task))
                _delegation_cache[key] = cached
        return cached

    def warm_delegations(calls: list[tuple[str, dict[str, Any]]]) -> None:
        """Classify a whole batch before it fans out (ADR 0044 §6).

        Branches then only read a warm entry; the sequential precompute also
        keeps the cost model of a group identical to the sequential path it
        replaces.
        """
        for tool_name, params in calls:
            _delegation_for(tool_name, params)

    def select_specialist(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "_validation_error":
            return {}
        delegation = _delegation_for(tool_name, params)
        deps.audit_agent_fn("tool_delegated", delegation)
        return {
            "specialist": str(delegation.specialist),
            "idempotency_key": delegation.idempotency_key,
            "remote_endpoint": delegation.remote_endpoint,
        }

    def verify_tool_result(tool_name: str, params: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "_validation_error":
            return {"passed": False, "reason": result.get("error", "Validation error")}
        delegation = _delegation_for(tool_name, params)
        verification = deps.verify_result_fn(delegation, result)
        deps.audit_agent_fn("tool_verified", delegation, **verification)
        return verification

    def deterministic_reflection(
        tool_name: str, params: dict[str, Any], result: dict[str, Any], verification: dict[str, Any]
    ) -> dict[str, Any]:
        delegation = _delegation_for(tool_name, params)
        reflection = deps.reflect_result_fn(delegation, result, verification)
        deps.audit_agent_fn("tool_reflected", delegation, **reflection)
        return reflection

    from .completion import CRITIC_JSON_SCHEMA, PLANNER_JSON_SCHEMA, structured_completion

    return deps.local_agent_graph(
        complete=complete,
        parse=lambda text: _parse_tool_call(text, tool_registry, validate_tool_call_fn, deps),
        execute=execute,
        needs_approval=lambda tn: tn in tools_requiring_approval,
        max_iterations=_AGENT_MAX_ITERATIONS,
        max_parallel=deps.parallel_max,
        requires_approval=frozenset(tools_requiring_approval),
        warm_delegations=warm_delegations,
        emit=event_sink,
        select_specialist=select_specialist,
        approval_covers=_approval_cover_probe(deps.tool_gateway, approval_token),
        verify_result=verify_tool_result,
        reflect_result=deterministic_reflection,
        plan_complete=lambda msgs, temp: structured_completion(
            msgs, 512, temp, PLANNER_JSON_SCHEMA, llm_runtime,
            deps.backend_mode_fn, deps.openai_compatible_fn),
        reflect_complete=lambda msgs, temp: structured_completion(
            msgs, 512, temp, CRITIC_JSON_SCHEMA, llm_runtime,
            deps.backend_mode_fn, deps.openai_compatible_fn),
        plan_reflect_complete=lambda msgs, temp: structured_completion(
            msgs, 512, temp, CRITIC_JSON_SCHEMA, llm_runtime,
            deps.backend_mode_fn, deps.openai_compatible_fn),
        cancel_checker=lambda: deps.task_coordinator.is_cancelled(session_id),
        checkpointer=deps.checkpointer,
    )


def _parse_tool_call(
    text: str,
    tool_registry: ToolRegistry,
    validate_tool_call_fn: Callable[..., tuple[dict[str, Any] | None, str | None] | None],
    deps: RunDeps,
) -> tuple[str | None, dict[str, Any] | None]:
    from .completion import parse_tool_call
    return parse_tool_call(text, tool_registry.models, validate_tool_call_fn,
                           deps.record_schema_error_fn)


_AGENT_MAX_ITERATIONS = 12

__all__ = [
    "_action_id_factory",
    "_approval_cover_probe",
    "_execute_tool",
    "_remote_dispatch",
    "build_run_graph",
]
