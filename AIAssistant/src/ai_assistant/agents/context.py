"""Shared LangGraph run wiring for the agent sub-services (ADR 0004/0012).

``AgentService``, ``ApprovalService`` and ``UIActionService`` each used to
spell out the same ~26 keyword arguments when calling ``run_langgraph_agent``:
the injected LLM runtime, the tool registry/gateway, the pending-action store,
and every supervisor/observability/LLM callable. That was ~78 lines of pure
duplication across the three services, and adding one dependency meant editing
three call sites.

:class:`AgentRunContext` owns that shared block once. It is a frozen, immutable
record of the wiring a run needs; each service builds one from its own injected
dependencies and calls :meth:`AgentRunContext.run` with only the arguments that
actually vary per invocation (prompt, task, resume state, event sink).
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..application.coordination import TaskCoordinator
from ..llm.defaults import CHARS_PER_TOKEN, LLM_N_CTX
from ..llm.inference import backend_mode, openai_compatible_completion
from ..observability import (
    langsmith_trace,
    record_schema_error,
    record_token_usage,
    record_tool,
    span,
)
from ..orchestration.specialists.code import is_coding_task
from ..orchestration.supervisor import Specialist, delegate, reflect_result, specialist_instruction, verify_result
from ..orchestration.supervisor import audit as audit_agent
from ..orchestration.supervisor import authorise as authorise_delegation
from ..ports import LLMRuntime, ToolGateway
from ..tools.tool_contract import validate_tool_call
from .deps import RunDeps
from .ids import generate_action_id
from .pending_store import PendingActionStore
from .runner import run_langgraph_agent

try:
    from ..orchestration.graph import LocalAgentGraph
except ImportError:  # pragma: no cover — graph framework optional in bare test envs
    LocalAgentGraph = None  # type: ignore[assignment,misc]


def _noop_validate_tool_call(
    tool_name: str, params: Any, tool_models: Any
) -> tuple[Any | None, str | None]:
    """Resume-path validator: parameters were already validated on the first pass."""
    return params, None


def _empty_action_id() -> str:
    """Resume-path action id: no new id is minted when continuing a graph."""
    return ""


# Resume continuations re-run the graph with the approved tool result already
# validated and the original action id, so they swap the first-pass validator
# and id generator for these no-ops.
RESUME_VALIDATE_TOOL_CALL = _noop_validate_tool_call
RESUME_ACTION_ID = _empty_action_id


@dataclass(frozen=True, slots=True)
class AgentRunContext:
    """Immutable bundle of the shared dependencies one agent run consumes."""

    llm_runtime: LLMRuntime | None = None
    pending_actions: PendingActionStore = field(default_factory=lambda: PendingActionStore(""))
    pending_lock: Any = field(default_factory=threading.Lock)
    tool_gateway: ToolGateway | None = None
    tool_registry: Any = None
    checkpointer: Any = None
    task_coordinator: TaskCoordinator = field(default_factory=TaskCoordinator)
    a2a_router: Any = None
    # The graph engine class is injected so the context stays framework-free at
    # import time; the composition root (or a service default) supplies it.
    local_agent_graph: Any = LocalAgentGraph

    def save_pending(self) -> None:
        """Persist the pending-action store (wired as the runner's save hook)."""
        self.pending_actions.save()

    def run(
        self,
        *,
        system_prompt: str,
        task: str,
        session_id: str,
        temperature: float,
        language: str,
        request_started: float,
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
        supervisor_route: Any = None,
        validate_tool_call_fn: Callable = validate_tool_call,
        generate_action_id_fn: Callable[[], str] = generate_action_id,
    ) -> dict:
        """Invoke the shared runner with this context's wiring.

        Only the per-invocation arguments are passed by the caller; every
        dependency (LLM, tools, supervisor, observability, persistence) is
        packed once into a frozen :class:`~ai_assistant.agents.deps.RunDeps`
        bundle, and the graph engine/``Specialist`` are supplied here so no
        service repeats them.
        """
        deps = RunDeps(
            # ── shared wiring owned by the context ──────────────────────────
            llm_runtime=self.llm_runtime,
            backend_mode_fn=backend_mode,
            openai_compatible_fn=openai_compatible_completion,
            record_token_usage_fn=record_token_usage,
            tool_registry=self.tool_registry,
            tool_gateway=self.tool_gateway,
            pending_actions=self.pending_actions,
            pending_lock=self.pending_lock,
            task_coordinator=self.task_coordinator,
            delegate_fn=delegate,
            authorise_delegation_fn=authorise_delegation,
            audit_agent_fn=audit_agent,
            verify_result_fn=verify_result,
            reflect_result_fn=reflect_result,
            record_tool_fn=record_tool,
            record_schema_error_fn=record_schema_error,
            langsmith_trace_fn=langsmith_trace,
            span_fn=span,
            specialist_instruction_fn=specialist_instruction,
            is_coding_task_fn=is_coding_task,
            local_agent_graph=self.local_agent_graph,
            specialist=Specialist,
            a2a_router=self.a2a_router,
            checkpointer=self.checkpointer,
            llm_n_ctx=LLM_N_CTX,
            chars_per_token=CHARS_PER_TOKEN,
            save_pending_fn=self.save_pending,
        )
        return run_langgraph_agent(
            deps=deps,
            system_prompt=system_prompt,
            task=task,
            session_id=session_id,
            temperature=temperature,
            language=language,
            request_started=request_started,
            initial_messages=initial_messages,
            initial_steps=initial_steps,
            initial_iteration=initial_iteration,
            resume_with_reflection=resume_with_reflection,
            event_sink=event_sink,
            prior_step_count=prior_step_count,
            approval_granted=approval_granted,
            approval_scope=approval_scope,
            approval_token=approval_token,
            granted_fingerprint=granted_fingerprint,
            supervisor_route=supervisor_route if supervisor_route is not None else Specialist.SUPERVISOR,
            validate_tool_call_fn=validate_tool_call_fn,
            generate_action_id_fn=generate_action_id_fn,
        )


__all__ = [
    "RESUME_ACTION_ID",
    "RESUME_VALIDATE_TOOL_CALL",
    "AgentRunContext",
]
