"""LangGraph orchestration graph for 3D-Reconstruction AI Assistant.

Coordinates ReAct reasoning, planning, tool execution, and reflection.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from ai_assistant.observability import langsmith_trace, span
from ai_assistant.orchestration.specialists.code import is_coding_task

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
    VerifyResult,
)

logger = logging.getLogger("ai_assistant.orchestration.graph")


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
        select_specialist: SelectSpecialist | None = None,
        approval_covers: ApprovalCovers | None = None,
        verify_result: VerifyResult | None = None,
        reflect_result: ReflectResult | None = None,
        plan_complete: Completion | None = None,
        reflect_complete: Completion | None = None,
        plan_reflect_complete: Completion | None = None,
        cancel_checker: Callable[[], bool] | None = None,
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
        self._select_specialist = select_specialist
        self._approval_covers = approval_covers
        self._verify_result = verify_result
        self._reflect_result = reflect_result
        self._plan_complete = plan_complete or complete
        self._reflect_complete = reflect_complete or complete
        self._plan_reflect_complete = plan_reflect_complete
        self._cancel_checker = cancel_checker or (lambda: False)
        self._emitted_steps = 0

        builder = StateGraph(AgentState)
        # LangGraph's typed overloads reject ReAct nodes returning plain
        # ``dict[str, Any]`` partials; cast through Any at this single boundary.
        # Runtime behavior is covered by the agent e2e/integration tests.
        builder.add_node("plan", cast(Any, self._traced("plan", self._plan)))
        builder.add_node("plan_reflect", cast(Any, self._traced("plan_reflect", self._plan_reflect)))
        builder.add_node("reason", cast(Any, self._traced("reason", self._reason)))
        builder.add_node("tool", cast(Any, self._traced("tool", self._tool)))
        builder.add_node("reflect", cast(Any, self._traced("reflect", self._reflect)))

        builder.add_conditional_edges(
            START,
            self._initial_node,
            {"plan": "plan", "reflect": "reflect"},
        )
        builder.add_conditional_edges(
            "plan",
            self._after_plan,
            {"plan_reflect": "plan_reflect", "reason": "reason"},
        )
        builder.add_conditional_edges(
            "plan_reflect",
            self._after_plan_reflect,
            {"plan": "plan", "reason": "reason", "end": END},
        )
        builder.add_conditional_edges(
            "reason",
            self._after_reason,
            {"tool": "tool", "reason": "reason", "end": END},
        )
        builder.add_conditional_edges(
            "tool",
            self._after_tool,
            {"reason": "reason", "reflect": "reflect", "end": END},
        )
        builder.add_conditional_edges(
            "reflect",
            self._after_reflect,
            {"reason": "reason", "end": END},
        )

        self._checkpointer = checkpointer
        self._graph = builder.compile(checkpointer=self._checkpointer)

    def _traced(self, name: str, handler: Callable[[AgentState], dict[str, Any]]) -> Callable[[AgentState], dict[str, Any]]:
        def invoke(state: AgentState) -> dict[str, Any]:
            iteration = state.get("iteration", 0)
            specialist = state.get("routing_plan", "supervisor")
            tool_name = ""
            for step in reversed(state.get("steps", [])):
                if step.get("type") in {"tool_call", "tool_result"}:
                    tool_name = step.get("tool", "")
                    break
            trace_metadata = {
                "iteration": iteration,
                "routing_plan": state.get("routing_plan", ""),
                "specialist": specialist,
                "tool_name": tool_name,
            }
            with span(f"agent.{name}", **trace_metadata):
                with langsmith_trace(
                    f"agent.{name}",
                    run_type="chain",
                    inputs={
                        "iteration": iteration,
                        "routing_plan": state.get("routing_plan", ""),
                        "done": state.get("done", False),
                        "specialist": specialist,
                        "tool_name": tool_name,
                    },
                    metadata=trace_metadata,
                ) as ls_ctx:
                    result = handler(state)
                    ls_ctx["outputs"] = {
                        "done": result.get("done", False),
                        "step_count": len(result.get("steps", [])),
                    }
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
        supervisor_route: str | None = None,
        enforce_plan_completion: bool = False,
        approval_granted: bool = False,
        approval_scope: str = "",
        granted_fingerprint: str = "",
    ) -> AgentState:
        self._emitted_steps = len(steps or [])
        restored_plan = next(
            (step.get("steps") for step in reversed(steps or []) if step.get("type") == "plan"),
            None,
        )
        prior_plan_review = next(
            (step.get("result", {}) for step in reversed(steps or []) if step.get("type") == "plan_reflection"),
            None,
        )

        config: RunnableConfig = {"configurable": {"thread_id": session_id}}

        input_state: dict[str, Any] = {
            "messages": messages,
            "steps": steps or [],
            "iteration": iteration,
            "temperature": temperature,
            "done": False,
            "pending_tool": None,
            "plan": restored_plan,
            "plan_spec": next((step.get("spec") for step in reversed(steps or []) if step.get("type") == "plan"), None),
            "approval_granted": approval_granted,
            "approval_scope": approval_scope,
            "granted_fingerprint": granted_fingerprint,
            "cancelled": False,
            "tool_call_count": 0,
            "last_reflection": None,
            "error_count": 0,
            "resume_with_reflection": resume_with_reflection,
            "skip_reflect": False,
            "synthesize_after_rag": False,
            "plan_verified": bool(prior_plan_review and prior_plan_review.get("passed") is True),
            "plan_attempts": sum(1 for step in (steps or []) if step.get("type") == "plan"),
            "plan_feedback": "",
            "routing_plan": supervisor_route,
            "enforce_plan_completion": enforce_plan_completion,
            "enforce_coding_workflow": is_coding_task(next((m.get("content", "") for m in messages if m.get("role") == "user"), "")),
        }

        if required_ui_actions is not None:
            input_state["required_ui_actions"] = required_ui_actions
        elif not resume_with_reflection or self._checkpointer is None:
            input_state["required_ui_actions"] = []
        else:
            try:
                prior = self._graph.get_state(config)
                prior_values = prior.values if prior else {}
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    "[run] Không thể đọc checkpoint trước đó để khôi phục required_ui_actions (session=%s): %s",
                    session_id,
                    error,
                )
                prior_values = {}
            input_state["required_ui_actions"] = prior_values.get("required_ui_actions", [])

        return cast(AgentState, self._graph.invoke(cast(Any, input_state), config=config))

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
