"""Core LangGraph agent runner."""
from __future__ import annotations

import logging
import time
from typing import Callable

from ai_assistant.agents.completion import (
    CRITIC_JSON_SCHEMA,
    PLANNER_JSON_SCHEMA,
    constrained_completion,
    parse_tool_call,
    structured_completion,
)
from ai_assistant.agents.pending_store import PendingActionStore
from ai_assistant.domain.errors import ServiceUnavailableError
from ai_assistant.tools.action_manifest import validate_action_params
from ai_assistant.tools.registry import ToolRegistry

logger = logging.getLogger("ai_assistant.agents.runner")

_AGENT_MAX_ITERATIONS = 12


def run_langgraph_agent(
    *,
    # Prompt & context
    system_prompt: str,
    task: str,
    session_id: str,
    temperature: float,
    language: str,
    request_started: float,
    # Infra dependencies (injected)
    llm_runtime,
    backend_mode_fn: Callable[[], str],
    openai_compatible_fn: Callable,
    record_token_usage_fn: Callable[[int, int], None],
    tool_registry: ToolRegistry,
    pending_actions: PendingActionStore,
    pending_lock,
    task_coordinator,
    delegate_fn: Callable,
    authorise_delegation_fn: Callable,
    audit_agent_fn: Callable,
    verify_result_fn: Callable,
    reflect_result_fn: Callable,
    record_tool_fn: Callable,
    record_schema_error_fn: Callable,
    validate_tool_call_fn: Callable,
    langsmith_trace_fn: Callable,
    span_fn: Callable,
    specialist_instruction_fn: Callable,
    is_coding_task_fn: Callable,
    # Optional state for resume
    initial_messages: list[dict[str, str]] | None = None,
    initial_steps: list[dict] | None = None,
    initial_iteration: int = 0,
    resume_with_reflection: bool = False,
    event_sink: Callable[[dict], None] | None = None,
    prior_step_count: int = 0,
    approval_granted: bool = False,
    approval_scope: str = "",
    supervisor_route=None,
    # LangGraph class
    LocalAgentGraph=None,
    Specialist=None,
    A2ARouter=None,
    # Tokens
    llm_n_ctx: int = 8192,
    chars_per_token: int = 4,
    # Helpers
    generate_action_id_fn: Callable[[], str] | None = None,
    save_pending_fn: Callable[[], None] | None = None,
) -> dict:
    """Run the tool loop via LangGraph while keeping Qt API response shape."""
    LANGGRAPH_AVAILABLE = LocalAgentGraph is not None
    if not LANGGRAPH_AVAILABLE:
        raise ServiceUnavailableError(
            "LangGraph is required for Agent mode. Run: pip install -r AIAssistant/requirements.txt"
        )
    if supervisor_route is None and Specialist is not None:
        supervisor_route = Specialist.SUPERVISOR

    openai_tools = tool_registry.get_openai_tools()
    grammar_schema = tool_registry.grammar
    tool_models = tool_registry.models
    tools_requiring_approval = {spec.name for spec in tool_registry.get_all() if spec.requires_approval}

    def _action_id() -> str:
        if generate_action_id_fn:
            return generate_action_id_fn()
        import hashlib
        return hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:12]

    def _save_pending() -> None:
        if save_pending_fn:
            save_pending_fn()
        else:
            pending_actions.save()

    # ── Inner callbacks for LangGraph ────────────────────────────────────────

    def complete(messages: list[dict], current_temp: float) -> str:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = int(total_chars / chars_per_token)
        if estimated_tokens >= llm_n_ctx - 512:
            return "Context quá dài, dừng Agent."
        max_tokens = min(2048, max(512, llm_n_ctx - estimated_tokens - 400))
        logger.info("LangGraph calls LLM (%d msgs, ~%d tokens)", len(messages), estimated_tokens)
        answer = constrained_completion(
            messages, max_tokens, current_temp,
            llm_runtime, backend_mode_fn, openai_compatible_fn,
            openai_tools, grammar_schema, record_token_usage_fn
        )
        logger.info("LangGraph LLM output (%d chars): %s", len(answer), answer[:120].replace("\n", " "))
        return answer

    def execute(tool_name: str, params: dict) -> dict:
        delegation = delegate_fn(task, session_id, tool_name, params,
                                 prefer_code=is_coding_task_fn(task))
        allowed, reason = authorise_delegation_fn(delegation, tool_name in tools_requiring_approval)
        if not allowed:
            audit_agent_fn("tool_denied", delegation, reason=reason)
            return {"error": reason or "Tool call denied by supervisor policy."}
        if tool_name == "application_action":
            canonical_params, error = validate_action_params(params)
            if error:
                return {"error": error}
            canonical_params["request_id"] = _action_id()
            params.clear()
            params.update(canonical_params)
        if tool_name == "_validation_error":
            return {"error": f"Lỗi xác thực tham số tool '{params.get('tool')}': {params.get('error')}"}
        spec = tool_registry.get(tool_name)
        if spec is None or spec.handler is None:
            return {"error": f"Tool không tồn tại hoặc không có handler: {tool_name}"}
        tool_started = time.monotonic()
        if delegation.remote_endpoint and A2ARouter is not None:
            remote_payload = {"tool": tool_name, "parameters": params,
                              "session_id": session_id,
                              "idempotency_key": delegation.idempotency_key}
            with span_fn("agent.a2a_delegate", tool=tool_name):
                result = A2ARouter().route(delegation.specialist.value, task, remote_payload)
            audit_agent_fn("tool_transport", delegation, source=result.get("source", "local"))
        else:
            with span_fn("agent.tool", tool=tool_name, session_id=session_id):
                from ai_assistant.bootstrap.runtime import execute_approved_tool, execute_tool
                # ADR 0002: ToolExecutionService is the only execution path. When the
                # platform runtime is unavailable we surface a structured error instead
                # of calling ``spec.handler`` directly and bypassing policy checks.
                result = (execute_approved_tool(tool_name, params) if approval_granted
                          else execute_tool(tool_name, params))
        audit_agent_fn("tool_completed", delegation, success="error" not in result)
        record_tool_fn(tool_name, "error" not in result, time.monotonic() - tool_started)
        return result

    def select_specialist(tool_name: str, params: dict) -> dict:
        if tool_name == "_validation_error":
            return {}
        delegation = delegate_fn(task, session_id, tool_name, params,
                                 prefer_code=is_coding_task_fn(task))
        audit_agent_fn("tool_delegated", delegation)
        return {
            "specialist": str(delegation.specialist),
            "idempotency_key": delegation.idempotency_key,
            "instruction": specialist_instruction_fn(delegation),
            "remote_endpoint": delegation.remote_endpoint,
        }

    def verify_tool_result(tool_name: str, params: dict, result: dict) -> dict:
        if tool_name == "_validation_error":
            return {"passed": False, "reason": result.get("error", "Validation error")}
        delegation = delegate_fn(task, session_id, tool_name, params,
                                 prefer_code=is_coding_task_fn(task))
        verification = verify_result_fn(delegation, result)
        audit_agent_fn("tool_verified", delegation, **verification)
        return verification

    def deterministic_reflection(tool_name: str, params: dict, result: dict, verification: dict) -> dict:
        delegation = delegate_fn(task, session_id, tool_name, params,
                                 prefer_code=is_coding_task_fn(task))
        reflection = reflect_result_fn(delegation, result, verification)
        audit_agent_fn("tool_reflected", delegation, **reflection)
        return reflection

    logger.info("Starting LangGraph session %s", session_id)

    # ── LangSmith tracing ────────────────────────────────────────────────────
    _ls_ctx_mgr = langsmith_trace_fn(
        "agent.session", run_type="chain",
        inputs={"task": task[:200], "session_id": session_id,
                "supervisor_route": supervisor_route.value if supervisor_route else "none"},
        metadata={"session_id": session_id, "temperature": temperature, "language": language},
    )
    _ls_ctx = _ls_ctx_mgr.__enter__()

    graph = LocalAgentGraph(
        complete=complete,
        parse=lambda text: parse_tool_call(text, tool_models, validate_tool_call_fn,
                                           record_schema_error_fn),
        execute=execute,
        needs_approval=lambda tn: tn in tools_requiring_approval,
        max_iterations=_AGENT_MAX_ITERATIONS,
        emit=event_sink,
        select_specialist=select_specialist,
        verify_result=verify_tool_result,
        reflect_result=deterministic_reflection,
        plan_complete=lambda msgs, temp: structured_completion(
            msgs, 512, temp, PLANNER_JSON_SCHEMA, llm_runtime, backend_mode_fn, openai_compatible_fn),
        reflect_complete=lambda msgs, temp: structured_completion(
            msgs, 512, temp, CRITIC_JSON_SCHEMA, llm_runtime, backend_mode_fn, openai_compatible_fn),
        plan_reflect_complete=lambda msgs, temp: structured_completion(
            msgs, 512, temp, CRITIC_JSON_SCHEMA, llm_runtime, backend_mode_fn, openai_compatible_fn),
        cancel_checker=lambda: task_coordinator.is_cancelled(session_id),
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
            supervisor_route=supervisor_route.value if supervisor_route else "supervisor",
            enforce_plan_completion=True,
            approval_granted=approval_granted or bool(
                (initial_steps or []) and
                any(step.get("type") == "approval_granted" for step in (initial_steps or []))
            ),
            approval_scope=approval_scope,
        )
    except BaseException as error:
        _ls_ctx["outputs"] = {"status": "error", "error": str(error)}
        _ls_ctx_mgr.__exit__(type(error), error, error.__traceback__)
        raise

    # ── State → response ─────────────────────────────────────────────────────
    pending_state = state.get("pending_tool") or {}
    pending_status = (
        "pending_ui_action" if pending_state.get("ui_ack")
        else "pending_approval" if pending_state
        else "completed"
    )
    if state.get("cancelled"):
        pending_status = "cancelled"

    if pending_state:
        task_coordinator.update(
            session_id,
            status="waiting_approval" if not pending_state.get("ui_ack") else "waiting_ui_ack",
            iterations=state.get("iteration", 0),
        )
    else:
        task_coordinator.finish(session_id, success=not state.get("cancelled"),
                                iterations=state.get("iteration", 0))

    total_ms = round((time.monotonic() - request_started) * 1000)
    _ls_ctx["outputs"] = {"status": pending_status, "iterations": state.get("iteration", 0),
                          "duration_ms": total_ms, "step_count": len(state.get("steps", []))}
    _ls_ctx_mgr.__exit__(None, None, None)

    steps = state["steps"]
    pending = state.get("pending_tool")
    if pending:
        action_id = _action_id()
        is_ui_ack = bool(pending.get("ui_ack"))
        request_id = pending["params"].get("request_id", action_id)
        with pending_lock:
            pending_actions[request_id if is_ui_ack else action_id] = {
                "tool": pending["tool"],
                "params": pending["params"],
                "session_id": session_id,
                "task": task,
                "messages": state["messages"],
                "steps": steps,
                "iteration": state["iteration"],
                "temperature": temperature,
                "language": language,
                "created_at": time.time(),
                "ui_ack": is_ui_ack,
                "approval_scope": pending.get("approval_scope", ""),
                "approval_preview": pending.get("approval_preview"),
            }
            _save_pending()

        if is_ui_ack:
            return {
                "status": "pending_ui_action", "session_id": session_id,
                "steps": steps, "request_id": request_id,
                "ui_action": {"request_id": request_id, "action": pending["params"]["action"],
                              "params": pending["params"]},
                "total_ms": round((time.monotonic() - request_started) * 1000),
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
            "total_ms": round((time.monotonic() - request_started) * 1000),
        }

    if not any(step["type"] == "final_answer" for step in steps):
        steps.append({"type": "final_answer", "content": "Agent đã kết thúc mà chưa có kết luận."})
    return {
        "status": "completed", "session_id": session_id, "steps": steps,
        "prior_step_count": prior_step_count,
        "iterations": state["iteration"],
        "total_ms": round((time.monotonic() - request_started) * 1000),
    }
