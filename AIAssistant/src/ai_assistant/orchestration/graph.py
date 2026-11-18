"""LangGraph orchestration graph for 3D-Reconstruction AI Assistant.

Coordinates ReAct reasoning, planning, tool execution, and reflection.

This module owns the ReAct + Plan-and-Execute topology but does **not**
import LangGraph directly.  The framework dependency is confined to
``internal/langgraph_engine.py`` (ADR 0003/0012); this module
reaches the engine through :class:`LangGraphEngine`, so swapping
the framework or running a mock engine in tests never touches this file.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, cast

from ai_assistant.domain.governance import redact_value
from ai_assistant.domain.parallel import PARALLEL_MAX_DEFAULT
from ai_assistant.internal.langgraph_engine import (
    END,
    START,
    CompiledGraph,
    LangGraphEngine,
)
from ai_assistant.observability import span
from ai_assistant.orchestration.specialists.code import is_coding_task

from .helpers import completed_plan_steps
from .nodes import (
    ReasonContext,
    ReflectContext,
    ToolContext,
    after_plan_reflect,
    after_reason,
    after_reflect,
    after_tool,
    plan_node,
    plan_reflect_node,
    reason_node,
    reflect_node,
    tool_node,
)
from .nodes import (
    summarize_messages as _summarize_messages,
)
from .state import (
    AgentState,
    ApprovalCovers,
    Completion,
    Executor,
    NeedsApproval,
    Parser,
    ReflectResult,
    SelectSpecialist,
    Step,
    StepType,
    VerifyResult,
    has_step_type,
)

logger = logging.getLogger("ai_assistant.orchestration.graph")

_TRACE_TEXT_LIMIT = 4_000
_TRACE_COLLECTION_LIMIT = 50


def _bound_trace_value(value: Any) -> Any:
    """Keep trace payloads readable and bounded after governance redaction."""
    if isinstance(value, str):
        if len(value) <= _TRACE_TEXT_LIMIT:
            return value
        return value[:_TRACE_TEXT_LIMIT] + "... [truncated]"
    if isinstance(value, dict):
        items = list(value.items())
        bounded = {
            str(key): _bound_trace_value(item)
            for key, item in items[:_TRACE_COLLECTION_LIMIT]
        }
        if len(items) > _TRACE_COLLECTION_LIMIT:
            bounded["_trace_truncated_items"] = len(items) - _TRACE_COLLECTION_LIMIT
        return bounded
    if isinstance(value, list):
        bounded_list = [_bound_trace_value(item) for item in value[:_TRACE_COLLECTION_LIMIT]]
        if len(value) > _TRACE_COLLECTION_LIMIT:
            bounded_list.append({"_trace_truncated_items": len(value) - _TRACE_COLLECTION_LIMIT})
        return bounded_list
    return value


def _trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply the shared secret/PII policy and trace-specific size limits."""
    return cast(dict[str, Any], _bound_trace_value(redact_value(payload)))


def _first_user_request(state: AgentState) -> str:
    return next(
        (message.get("content", "") for message in state.get("messages", [])
         if message.get("role") == "user"),
        "",
    )


def _last_step(steps: list[Step], step_type: StepType, tool_name: str = "") -> Step | None:
    return next(
        (step for step in reversed(steps)
         if has_step_type(step, step_type)
         and (not tool_name or step.get("tool") == tool_name)),
        None,
    )


def _add_plan_trace_inputs(
    inputs: dict[str, Any], state: AgentState, messages: list[dict[str, str]], user_request: str,
) -> None:
    inputs["user_request"] = user_request
    inputs["message_count"] = len(messages)
    if state.get("prior_plan"):
        inputs["prior_plan"] = state.get("prior_plan")
    if state.get("plan_feedback"):
        inputs["plan_feedback"] = state.get("plan_feedback")


def _add_plan_reflect_trace_inputs(
    inputs: dict[str, Any], state: AgentState, plan: list[str], user_request: str,
) -> None:
    inputs["user_request"] = user_request
    inputs["plan_to_review"] = plan
    inputs["plan_attempt"] = state.get("plan_attempts", 0)
    if state.get("plan_spec"):
        inputs["plan_spec"] = state.get("plan_spec")


def _add_reason_trace_inputs(
    inputs: dict[str, Any], steps: list[Step], messages: list[dict[str, str]], user_request: str,
) -> None:
    inputs["user_request"] = user_request
    inputs["recent_messages"] = messages[-8:]
    inputs["message_count"] = len(messages)
    last_observation = _last_step(steps, "tool_result")
    last_reflection = _last_step(steps, "reflection")
    if last_observation:
        inputs["last_observation"] = last_observation
    if last_reflection:
        inputs["last_reflection"] = last_reflection


def _add_tool_trace_inputs(
    inputs: dict[str, Any], steps: list[Step], tool_name: str, tool_params: dict[str, Any],
) -> None:
    call_steps = [step for step in steps if has_step_type(step, "tool_call")]
    result_count = sum(1 for step in steps if has_step_type(step, "tool_result"))
    inputs["tool_calls"] = call_steps[result_count:]
    if tool_name:
        inputs["tool_name"] = tool_name
    if tool_params:
        inputs["tool_params"] = tool_params


def _add_reflect_trace_inputs(
    inputs: dict[str, Any], steps: list[Step], tool_name: str, user_request: str,
) -> None:
    inputs["user_request"] = user_request
    if tool_name:
        inputs["tool_name"] = tool_name
    last_call = _last_step(steps, "tool_call", tool_name)
    last_result = _last_step(steps, "tool_result", tool_name)
    last_verification = _last_step(steps, "verification", tool_name)
    if last_call:
        inputs["tool_call"] = last_call
    if last_result:
        inputs["tool_result"] = last_result
    if last_verification:
        inputs["verification"] = last_verification


def _build_node_trace_inputs(
    name: str,
    state: AgentState,
    plan: list[str],
    current_step: str | None,
    current_step_index: int | None,
    done_count: int,
    tool_name: str,
    tool_params: dict[str, Any],
) -> dict[str, Any]:
    steps = list(state.get("steps") or [])
    messages = list(state.get("messages") or [])
    user_request = _first_user_request(state)
    inputs: dict[str, Any] = {
        "node": name,
        "iteration": state.get("iteration", 0),
        "routing_plan": state.get("routing_plan", ""),
        "specialist": state.get("routing_plan") or "supervisor",
        "done": state.get("done", False),
    }
    if plan:
        inputs["plan"] = plan
        inputs["current_step"] = current_step
        inputs["current_step_index"] = current_step_index
        inputs["completed_plan_steps"] = done_count
        inputs["total_plan_steps"] = len(plan)

    if name == "plan":
        _add_plan_trace_inputs(inputs, state, messages, user_request)
    elif name == "plan_reflect":
        _add_plan_reflect_trace_inputs(inputs, state, plan, user_request)
    elif name == "reason":
        _add_reason_trace_inputs(inputs, steps, messages, user_request)
    elif name == "tool":
        _add_tool_trace_inputs(inputs, steps, tool_name, tool_params)
    elif name == "reflect":
        _add_reflect_trace_inputs(inputs, steps, tool_name, user_request)
    return _trace_payload(inputs)


def _trace_plan_outputs(result: dict[str, Any], outputs: dict[str, Any]) -> None:
    gen_plan = result.get("plan")
    outputs["plan"] = gen_plan
    outputs["plan_verified"] = result.get("plan_verified", False)
    plan_spec = result.get("plan_spec")
    if plan_spec:
        outputs["plan_spec"] = plan_spec
    if gen_plan:
        outputs["plan_step_count"] = len(gen_plan)


def _trace_plan_reflect_outputs(
    result: dict[str, Any],
    new_steps: list[Step],
    outputs: dict[str, Any],
) -> None:
    outputs["plan_verified"] = result.get("plan_verified", False)
    plan_reflect_step = next((s for s in reversed(new_steps) if has_step_type(s, "plan_reflection")), None)
    if plan_reflect_step:
        review = plan_reflect_step.get("result", {})
        outputs["review"] = review
        outputs["decision"] = review.get("decision", "")
    if result.get("plan_feedback"):
        outputs["plan_feedback"] = result.get("plan_feedback")


def _trace_reason_outputs(
    result: dict[str, Any],
    new_steps: list[Step],
    plan: list[str],
    outputs: dict[str, Any],
) -> None:
    tool_call_steps = [step for step in new_steps if has_step_type(step, "tool_call")]
    step_answer_step = next((s for s in reversed(new_steps) if has_step_type(s, "step_answer")), None)
    final_answer_step = next((s for s in reversed(new_steps) if has_step_type(s, "final_answer")), None)

    if tool_call_steps:
        calls: list[dict[str, Any]] = []
        for tool_call_step in tool_call_steps:
            tc_info: dict[str, Any] = {
                "tool": tool_call_step.get("tool"),
                "params": tool_call_step.get("params"),
            }
            if "plan_step_index" in tool_call_step:
                idx = tool_call_step.get("plan_step_index")
                tc_info["plan_step_index"] = idx
                if plan and isinstance(idx, int) and 0 <= idx < len(plan):
                    tc_info["step"] = plan[idx]
            calls.append(tc_info)
        outputs["tool_calls"] = calls
    if result.get("pending_tool"):
        outputs["decision"] = "pending_tool"
        outputs["pending_tool"] = result.get("pending_tool")
    elif tool_call_steps:
        outputs["decision"] = "tool_call"
    elif step_answer_step:
        outputs["decision"] = "step_answer"
        outputs["step_answer"] = step_answer_step.get("content")
    elif final_answer_step:
        outputs["decision"] = "final_answer"
        outputs["final_answer"] = final_answer_step.get("content")


def _trace_tool_outputs(
    result: dict[str, Any],
    new_steps: list[Step],
    outputs: dict[str, Any],
) -> None:
    tool_res_steps = [step for step in new_steps if has_step_type(step, "tool_result")]
    if tool_res_steps:
        tool_results: list[dict[str, Any]] = [
            {
                "tool": step.get("tool"),
                "result": step.get("result", {}),
                "success": (
                    "error" not in (step.get("result") or {})
                    and bool(step.get("success", True))
                ),
                "plan_step_index": step.get("plan_step_index"),
                "batch_id": step.get("batch_id"),
            }
            for step in tool_res_steps
        ]
        outputs["tool_results"] = tool_results
        outputs["success"] = all(item["success"] for item in tool_results)
        if len(tool_results) == 1:
            outputs["tool_name"] = tool_results[0]["tool"]
            outputs["tool_result"] = tool_results[0]["result"]
    elif result.get("pending_tool"):
        pending = result["pending_tool"]
        status = "pending_ui_action" if pending.get("ui_ack") else "pending_approval"
        outputs["pending_status"] = status
        outputs["pending_tool"] = pending


def _trace_reflect_outputs(
    new_steps: list[Step],
    outputs: dict[str, Any],
) -> None:
    reflect_step = next((s for s in reversed(new_steps) if has_step_type(s, "reflection")), None)
    if reflect_step:
        ref_res = reflect_step.get("result") or reflect_step.get("critique")
        outputs["reflection"] = ref_res


def _build_node_trace_outputs(
    name: str,
    result: dict[str, Any],
    state: AgentState,
    plan: list[str],
) -> dict[str, Any]:
    steps_before = list(state.get("steps") or [])
    messages_before = list(state.get("messages") or [])
    new_steps = [s for s in result.get("steps", [])[len(steps_before):]] if result.get("steps") else []
    new_messages = (
        result.get("messages", [])[len(messages_before):]
        if result.get("messages") else []
    )
    outputs: dict[str, Any] = {
        "node": name,
        "done": result.get("done", state.get("done", False)),
        "state_update": {
            key: value for key, value in result.items()
            if key not in {"steps", "messages"}
        },
        "new_steps": new_steps,
        "new_messages": new_messages,
        "new_step_count": len(new_steps),
    }
    if name == "plan":
        _trace_plan_outputs(result, outputs)
    elif name == "plan_reflect":
        _trace_plan_reflect_outputs(result, new_steps, outputs)
    elif name == "reason":
        _trace_reason_outputs(result, new_steps, plan, outputs)
    elif name == "tool":
        _trace_tool_outputs(result, new_steps, outputs)
    elif name == "reflect":
        _trace_reflect_outputs(new_steps, outputs)
    return _trace_payload(outputs)


class LocalAgentGraph:
    """ReAct + Plan-and-Execute graph for local AI agent workflows."""

    def __init__(
        self,
        complete: Completion,
        parse: Parser,
        execute: Executor,
        needs_approval: NeedsApproval,
        max_iterations: int,
        emit: Callable[[dict[str, Any]], None] | None = None,
        stream_event_sink: Callable[[dict[str, Any]], None] | None = None,
        select_specialist: SelectSpecialist | None = None,
        approval_covers: ApprovalCovers | None = None,
        verify_result: VerifyResult | None = None,
        reflect_result: ReflectResult | None = None,
        plan_complete: Completion | None = None,
        reflect_complete: Completion | None = None,
        plan_reflect_complete: Completion | None = None,
        response_stream_complete: Completion | None = None,
        cancel_checker: Callable[[], bool] | None = None,
        # Parallel tool batches (ADR 0044): the width, the group-wide approval
        # refusal set and the sequential delegation precompute. Defaulted so a
        # bare graph (tests, embeds) simply never fans out.
        max_parallel: int = PARALLEL_MAX_DEFAULT,
        requires_approval: frozenset[str] = frozenset(),
        warm_delegations: Callable[[list[tuple[str, dict[str, Any]]]], None] | None = None,
        # Persistence is an adapter concern: the composition root injects the
        # checkpointer (see bootstrap). ``None`` runs stateless, which is what
        # tests and single-shot runs want.
        checkpointer: Any | None = None,
    ) -> None:
        self._complete = complete
        self._parse = parse
        self._execute = execute
        self._needs_approval = needs_approval
        self._max_iterations = max_iterations
        self._emit = emit
        self._stream_event_sink = stream_event_sink
        self._select_specialist = select_specialist
        self._approval_covers = approval_covers
        self._verify_result = verify_result
        self._reflect_result = reflect_result
        self._plan_complete = plan_complete or complete
        self._reflect_complete = reflect_complete or complete
        self._plan_reflect_complete = plan_reflect_complete
        self._response_stream_complete = response_stream_complete
        self._cancel_checker = cancel_checker or (lambda: False)
        self._max_parallel = max_parallel
        self._requires_approval = requires_approval
        self._warm_delegations = warm_delegations
        self._emitted_steps = 0

        # Build the graph through the engine wrapper (no direct LangGraph
        # import). The state schema is *passed in*: the wrapper imports no
        # orchestration module, so the two layers stay one-directional.
        engine = LangGraphEngine(AgentState)
        engine.add_node("plan", cast(Any, self._traced("plan", self._plan)))
        engine.add_node("plan_reflect", cast(Any, self._traced("plan_reflect", self._plan_reflect)))
        engine.add_node("reason", cast(Any, self._traced("reason", self._reason)))
        engine.add_node("tool", cast(Any, self._traced("tool", self._tool)))
        engine.add_node("reflect", cast(Any, self._traced("reflect", self._reflect)))

        engine.add_conditional_edges(
            START,
            self._initial_node,
            {"plan": "plan", "reflect": "reflect"},
        )
        engine.add_conditional_edges(
            "plan",
            self._after_plan,
            {"plan_reflect": "plan_reflect", "reason": "reason"},
        )
        engine.add_conditional_edges(
            "plan_reflect",
            self._after_plan_reflect,
            {"plan": "plan", "reason": "reason", "end": END},
        )
        engine.add_conditional_edges(
            "reason",
            self._after_reason,
            {"tool": "tool", "reason": "reason", "end": END},
        )
        engine.add_conditional_edges(
            "tool",
            self._after_tool,
            {"reason": "reason", "reflect": "reflect", "end": END},
        )
        engine.add_conditional_edges(
            "reflect",
            self._after_reflect,
            {"reason": "reason", "plan": "plan", "end": END},
        )

        self._checkpointer = checkpointer
        self._graph: CompiledGraph[AgentState] = engine.compile(checkpointer=self._checkpointer)

    def _traced(self, name: str, handler: Callable[[AgentState], dict[str, Any]]) -> Callable[[AgentState], dict[str, Any]]:
        def invoke(state: AgentState) -> dict[str, Any]:
            iteration = state.get("iteration", 0)
            specialist = state.get("routing_plan") or "supervisor"
            plan = list(state.get("plan") or [])
            steps_before = list(state.get("steps") or [])

            done_count = completed_plan_steps(steps_before, plan) if plan else 0
            current_step = plan[done_count] if plan and done_count < len(plan) else None
            current_step_index = done_count if plan and done_count < len(plan) else None

            last_tool_call = next((s for s in reversed(steps_before) if has_step_type(s, "tool_call")), None)
            tool_name = last_tool_call.get("tool", "") if last_tool_call else ""
            tool_params = last_tool_call.get("params", {}) if last_tool_call else {}

            trace_metadata: dict[str, Any] = {
                "iteration": iteration,
                "routing_plan": state.get("routing_plan", ""),
                "specialist": specialist,
                "node": name,
            }
            if plan:
                trace_metadata["plan"] = json.dumps(plan, ensure_ascii=False)[:500]
                if current_step is not None:
                    trace_metadata["current_step"] = str(current_step)
                    trace_metadata["current_step_index"] = done_count
            if tool_name:
                trace_metadata["tool_name"] = tool_name
            trace_metadata = _trace_payload(trace_metadata)

            trace_inputs = _build_node_trace_inputs(
                name=name, state=state, plan=plan,
                current_step=current_step, current_step_index=current_step_index,
                done_count=done_count, tool_name=tool_name, tool_params=tool_params,
            )

            if self._stream_event_sink:
                self._stream_event_sink({
                    "_stream_event": "status",
                    "node": name,
                    "phase": "started",
                    "iteration": iteration,
                })

            with span(
                f"agent.{name}",
                trace_inputs=trace_inputs,
                trace_metadata=trace_metadata,
                **trace_metadata,
            ) as trace_context:
                result = handler(state)
                if trace_context is not None:
                    trace_context["outputs"] = _build_node_trace_outputs(
                        name=name, result=result, state=state, plan=plan,
                    )
            if self._stream_event_sink:
                self._stream_event_sink({
                    "_stream_event": "status",
                    "node": name,
                    "phase": "completed",
                    "iteration": result.get("iteration", iteration),
                })
            if self._emit and result.get("steps"):
                steps = result["steps"]
                for step in steps[self._emitted_steps:]:
                    self._emit(step)
                self._emitted_steps = len(steps)
            return result

        return invoke

    def run(
        self,
        messages: list[dict[str, str]],
        session_id: str,
        temperature: float,
        steps: list[dict[str, Any]] | None = None,
        iteration: int = 0,
        resume_with_reflection: bool = False,
        required_ui_actions: list[dict[str, Any]] | None = None,
        enforce_plan_completion: bool = False,
        approval_granted: bool = False,
        approval_scope: str = "",
        granted_fingerprint: str = "",
        prior_plan: list[str] | None = None,
        skip_plan: bool = False,
    ) -> AgentState:
        self._emitted_steps = len(steps or [])
        restored_plan = next(
            (step.get("steps") for step in reversed(steps or []) if has_step_type(step, "plan")),
            None,
        )
        prior_plan_review = next(
            (step.get("result", {}) for step in reversed(steps or []) if has_step_type(step, "plan_reflection")),
            None,
        )

        config = {"configurable": {"thread_id": session_id}}

        input_state: dict[str, Any] = {
            "messages": messages,
            "steps": steps or [],
            "iteration": iteration,
            "temperature": temperature,
            "done": False,
            "pending_tool": None,
            "plan": restored_plan,
            "plan_spec": next((step.get("spec") for step in reversed(steps or []) if has_step_type(step, "plan")), None),
            "approval_granted": approval_granted,
            "approval_scope": approval_scope,
            "granted_fingerprint": granted_fingerprint,
            "cancelled": False,
            "tool_call_count": 0,
            "error_count": 0,
            "resume_with_reflection": resume_with_reflection,
            "skip_reflect": False,
            "synthesize_after_rag": False,
            "plan_verified": bool(prior_plan_review and prior_plan_review.get("passed") is True),
            "plan_attempts": sum(1 for step in (steps or []) if has_step_type(step, "plan")),
            "plan_feedback": "",
            # Procedural memory seed (ADR 0043 §7): always present so the
            # schema channel is filled; ``None`` means "no remembered recipe".
            "prior_plan": prior_plan,
            # The route is a constant: no caller ever passed one (review 21).
            "routing_plan": "supervisor",
            "enforce_plan_completion": enforce_plan_completion,
            "enforce_coding_workflow": is_coding_task(next((m.get("content", "") for m in messages if m.get("role") == "user"), "")),
            "replan_requested": False,
            "replan_count": 0,
            "max_replans": 2,
            "max_tool_calls": max(4, self._max_iterations),
            "skip_plan": skip_plan,
        }

        if required_ui_actions is not None:
            input_state["required_ui_actions"] = required_ui_actions
        elif not resume_with_reflection or self._checkpointer is None:
            input_state["required_ui_actions"] = []
        else:
            try:
                prior = self._graph.get_state(config)
                prior_values = prior.values if prior else {}
            except Exception as error:
                logger.warning(
                    "[run] Không thể đọc checkpoint trước đó để khôi phục required_ui_actions (session=%s): %s",
                    session_id,
                    error,
                )
                prior_values = {}
            input_state["required_ui_actions"] = prior_values.get("required_ui_actions", [])

        return self._graph.invoke(input_state, config)

    @staticmethod
    def _initial_node(state: AgentState) -> str:
        return "reflect" if state.get("resume_with_reflection") else "plan"

    @staticmethod
    def _after_plan(state: AgentState) -> str:
        if state.get("plan_verified") or not state.get("plan"):
            return "reason"
        return "plan_reflect"

    def _plan(self, state: AgentState) -> dict[str, Any]:
        return plan_node(state, self._plan_complete)

    def _plan_reflect(self, state: AgentState) -> dict[str, Any]:
        return plan_reflect_node(state, self._plan_reflect_complete)

    @staticmethod
    def _after_plan_reflect(state: AgentState) -> str:
        return after_plan_reflect(state)

    def _reason(self, state: AgentState) -> dict[str, Any]:
        ctx = ReasonContext(
            complete=self._complete,
            parse=self._parse,
            needs_approval=self._needs_approval,
            max_iterations=self._max_iterations,
            cancel_checker=self._cancel_checker,
            select_specialist=self._select_specialist,
            approval_covers=self._approval_covers,
            stream_complete=self._response_stream_complete,
        )
        return reason_node(state, ctx)

    @staticmethod
    def _after_reason(state: AgentState) -> str:
        return after_reason(state)

    def _tool(self, state: AgentState) -> dict[str, Any]:
        ctx = ToolContext(
            execute=self._execute,
            verify_result=self._verify_result,
            cancel_checker=self._cancel_checker,
            requires_approval=self._requires_approval,
            warm_delegations=self._warm_delegations,
            max_parallel=self._max_parallel,
        )
        return tool_node(state, ctx)

    @staticmethod
    def _after_tool(state: AgentState) -> str:
        return after_tool(state)

    def _reflect(self, state: AgentState) -> dict[str, Any]:
        ctx = ReflectContext(
            reflect_complete=self._reflect_complete,
            reflect_result=self._reflect_result,
        )
        return reflect_node(state, ctx)

    @staticmethod
    def _after_reflect(state: AgentState) -> str:
        return after_reflect(state)


__all__ = [
    "AgentState",
    "LocalAgentGraph",
    "_summarize_messages",
]
