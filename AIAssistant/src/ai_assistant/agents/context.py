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

Every supervisor/observability/LLM callable is an overridable *field* whose
default is the production module function (review finding 10): the container is
a real seam, so a test can substitute policy without reaching past the context.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..application.coordination import TaskCoordinator
from ..domain.errors import ModelNotLoadedError
from ..domain.parallel import PARALLEL_MAX_DEFAULT
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
from ..orchestration.state import Step
from ..orchestration.supervisor import audit as audit_agent
from ..orchestration.supervisor import authorise as authorise_delegation
from ..orchestration.supervisor import delegate, reflect_result, verify_result
from ..ports import LLMRuntime, PendingActionPort, ToolGateway
from ..tools.tool_contract import validate_tool_call
from .deps import RunDeps
from .ids import generate_action_id
from .pending_store import MemoryPendingStore
from .runner import run_langgraph_agent

try:
    from ..orchestration.graph import LocalAgentGraph
except ImportError:  # pragma: no cover — graph framework optional in bare test envs
    LocalAgentGraph = None  # type: ignore[assignment,misc]


def _empty_action_id() -> str:
    """Resume-path action id: no new id is minted when continuing a graph."""
    return ""


# A resumed continuation re-runs the graph with the original action id, so it
# swaps the first-pass id generator for this no-op. Tool-call *validation* is
# deliberately NOT bypassed on resume: a resumed run may emit brand-new calls
# and every one of them is validated by the real validator (review finding 27).
RESUME_ACTION_ID = _empty_action_id


def require_llm_runtime(runtime: LLMRuntime | None) -> None:
    """Raise when no inference backend is loaded (shared by every entry point)."""
    if runtime is None or runtime.llm is None:
        msg = "LLM chưa khởi tạo"
        raise ModelNotLoadedError(msg)


class _ExecutionOnlyGateway:
    """Physical execution view of the gateway, handed to the decision loop.

    The composition root builds one object that both executes tools and mints
    approval grants (``ToolGateway``); passing it straight into
    :class:`~ai_assistant.agents.deps.RunDeps` left the least-privilege split
    a type-annotation story — every closure inside the graph held an object
    with ``issue_approval_grant`` on it. This view exposes only the four
    execution methods, so the runner physically cannot mint a grant
    (review finding 15): separation of duties by construction, not by mypy.
    """

    __slots__ = ("_gateway",)

    def __init__(self, gateway: ToolGateway) -> None:
        self._gateway = gateway

    def execute(self, tool_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return self._gateway.execute(tool_name, parameters)

    def execute_approved(
        self, tool_name: str, parameters: dict[str, Any], approval_token: str = ""
    ) -> dict[str, Any]:
        return self._gateway.execute_approved(tool_name, parameters, approval_token)

    def preflight(
        self, tool_name: str, parameters: dict[str, Any], approval_token: str = ""
    ) -> dict[str, Any]:
        return self._gateway.preflight(tool_name, parameters, approval_token)

    def approval_covers(
        self, tool_name: str, parameters: dict[str, Any], approval_token: str
    ) -> bool:
        return self._gateway.approval_covers(tool_name, parameters, approval_token)


def _execution_view(gateway: ToolGateway | None) -> _ExecutionOnlyGateway | None:
    """The runner's gateway: the execution half, or ``None`` when unwired."""
    return None if gateway is None else _ExecutionOnlyGateway(gateway)


#: Upper bound on a tool result embedded in a resume message, in characters.
RESULT_TEXT_LIMIT = 8000


def truncate_result_text(result: Any, limit: int = RESULT_TEXT_LIMIT) -> str:
    """JSON-encode a tool result, truncating to ``limit`` for prompt embedding."""
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if len(text) > limit:
        return text[:limit] + "\n... [truncated]"
    return text


def append_tool_result_turn(
    messages: list[dict[str, str]],
    tool_name: str,
    params: dict[str, Any],
    result: Any,
    *,
    lead: str,
    tail: str,
) -> None:
    """Append the executed tool call and its (truncated) result as two turns.

    Both resume paths — HITL approval and desktop UI-ack — synthesise the same
    assistant ``tool_call`` fence plus a user result turn; only the surrounding
    prose differs, so it is passed in as ``lead``/``tail`` (review finding 24).
    """
    call_text = json.dumps({"tool": tool_name, "params": params}, ensure_ascii=False)
    messages.append({"role": "assistant", "content": f"```tool_call\n{call_text}\n```"})
    result_text = truncate_result_text(result)
    messages.append({
        "role": "user",
        "content": f"{lead}\n```json\n{result_text}\n```\n\n{tail}",
    })


@dataclass(frozen=True, slots=True)
class AgentRunContext:
    """Immutable bundle of the shared dependencies one agent run consumes.

    The supervisor, observability and LLM callables are fields (not closed-over
    module functions) so the wiring is a genuine injection seam; the defaults
    are the production implementations.
    """

    llm_runtime: LLMRuntime | None = None
    pending_actions: PendingActionPort = field(default_factory=MemoryPendingStore)
    pending_lock: Any = field(default_factory=threading.Lock)
    tool_gateway: ToolGateway | None = None
    tool_registry: Any = None
    checkpointer: Any = None
    task_coordinator: TaskCoordinator = field(default_factory=TaskCoordinator)
    a2a_router: Any = None
    # The graph engine class is injected so the context stays framework-free at
    # import time; the composition root (or a service default) supplies it.
    local_agent_graph: Any = LocalAgentGraph

    # ── LLM access ────────────────────────────────────────────────────────
    backend_mode_fn: Callable[[], str] = backend_mode
    openai_compatible_fn: Callable[..., dict[str, Any]] = openai_compatible_completion
    record_token_usage_fn: Callable[[int, int], None] = record_token_usage

    # ── Supervisor policy ─────────────────────────────────────────────────
    delegate_fn: Callable[..., Any] = delegate
    authorise_delegation_fn: Callable[..., tuple[bool, str | None]] = authorise_delegation
    verify_result_fn: Callable[..., dict[str, Any]] = verify_result
    reflect_result_fn: Callable[..., dict[str, Any]] = reflect_result
    audit_agent_fn: Callable[..., None] = audit_agent
    is_coding_task_fn: Callable[[str], bool] = is_coding_task

    # ── Observability ─────────────────────────────────────────────────────
    record_tool_fn: Callable[..., None] = record_tool
    record_schema_error_fn: Callable[..., None] = record_schema_error
    langsmith_trace_fn: Callable[..., Any] = langsmith_trace
    span_fn: Callable[..., Any] = span

    #: Fan-out width for one parallel tool group (ADR 0044), clamped by the
    #: settings parser on the way in. The default keeps an unconfigured
    #: context bounded instead of unbounded.
    parallel_max: int = PARALLEL_MAX_DEFAULT

    def save_pending(self) -> None:
        """Persist the pending-action store (wired as the runner's save hook)."""
        self.pending_actions.save()

    def require_llm(self) -> None:
        """Raise :class:`ModelNotLoadedError` when no backend is loaded."""
        require_llm_runtime(self.llm_runtime)

    def resume_available(self, *, require_llm_ready: bool = False) -> bool:
        """Whether the LangGraph loop can continue a parked run.

        The loop is the only production engine; it is available when the graph
        class is wired. The desktop UI-ack path additionally requires a live
        backend because it resumes without the approval service's earlier
        ``require_llm`` guard.
        """
        if self.local_agent_graph is None:
            return False
        if require_llm_ready and (self.llm_runtime is None or self.llm_runtime.llm is None):
            return False
        return True

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
        validate_tool_call_fn: Callable[
            ..., tuple[dict[str, Any] | None, str | None] | None
        ] = validate_tool_call,
        generate_action_id_fn: Callable[[], str] = generate_action_id,
    ) -> dict[str, Any]:
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
            backend_mode_fn=self.backend_mode_fn,
            openai_compatible_fn=self.openai_compatible_fn,
            record_token_usage_fn=self.record_token_usage_fn,
            tool_registry=self.tool_registry,
            tool_gateway=_execution_view(self.tool_gateway),
            pending_actions=self.pending_actions,
            pending_lock=self.pending_lock,
            task_coordinator=self.task_coordinator,
            delegate_fn=self.delegate_fn,
            authorise_delegation_fn=self.authorise_delegation_fn,
            audit_agent_fn=self.audit_agent_fn,
            verify_result_fn=self.verify_result_fn,
            reflect_result_fn=self.reflect_result_fn,
            record_tool_fn=self.record_tool_fn,
            record_schema_error_fn=self.record_schema_error_fn,
            langsmith_trace_fn=self.langsmith_trace_fn,
            span_fn=self.span_fn,
            is_coding_task_fn=self.is_coding_task_fn,
            local_agent_graph=self.local_agent_graph,
            a2a_router=self.a2a_router,
            checkpointer=self.checkpointer,
            llm_n_ctx=LLM_N_CTX,
            chars_per_token=CHARS_PER_TOKEN,
            parallel_max=self.parallel_max,
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
            prior_plan=prior_plan,
            initial_iteration=initial_iteration,
            resume_with_reflection=resume_with_reflection,
            event_sink=event_sink,
            prior_step_count=prior_step_count,
            approval_granted=approval_granted,
            approval_scope=approval_scope,
            approval_token=approval_token,
            granted_fingerprint=granted_fingerprint,
            validate_tool_call_fn=validate_tool_call_fn,
            generate_action_id_fn=generate_action_id_fn,
        )


class ContextService:
    """Shared construction and accessor surface for the sub-services.

    ``ApprovalService`` and ``UIActionService`` each carried this *same*
    constructor — identical signature, identical defaults, 20 lines apiece —
    on top of the same six read-only properties plus
    ``_save_pending``/``_require_llm``; everything lives here once so a new
    dependency means one edit, not three (review finding 24, review 2026-10-16
    duplication finding). The concrete services declare no ``__init__`` at all.
    """

    #: Built once by this constructor from the caller's injected wiring.
    _ctx: AgentRunContext

    def __init__(
        self,
        *,
        llm_runtime: LLMRuntime | None = None,
        pending_actions: PendingActionPort | None = None,
        pending_lock: Any = None,
        tool_gateway: ToolGateway | None = None,
        tool_registry: Any = None,
        checkpointer: Any = None,
        task_coordinator: TaskCoordinator | None = None,
        delegate_fn: Callable[..., Any] | None = None,
        authorise_delegation_fn: Callable[..., tuple[bool, str | None]] | None = None,
        parallel_max: int = PARALLEL_MAX_DEFAULT,
    ) -> None:
        supervisor_policy: dict[str, Any] = {}
        if authorise_delegation_fn is not None:
            supervisor_policy["authorise_delegation_fn"] = authorise_delegation_fn
        self._ctx = AgentRunContext(
            llm_runtime=llm_runtime,
            pending_actions=pending_actions if pending_actions is not None else MemoryPendingStore(),
            pending_lock=threading.Lock() if pending_lock is None else pending_lock,
            tool_gateway=tool_gateway,
            tool_registry=tool_registry,
            checkpointer=checkpointer,
            task_coordinator=TaskCoordinator() if task_coordinator is None else task_coordinator,
            delegate_fn=delegate if delegate_fn is None else delegate_fn,
            parallel_max=parallel_max,
            **supervisor_policy,
        )

    @property
    def llm_runtime(self) -> LLMRuntime | None:
        return self._ctx.llm_runtime

    @property
    def pending_actions(self) -> PendingActionPort:
        return self._ctx.pending_actions

    @property
    def pending_lock(self) -> Any:
        return self._ctx.pending_lock

    @property
    def tool_gateway(self) -> ToolGateway | None:
        return self._ctx.tool_gateway

    @property
    def tool_registry(self) -> Any:
        return self._ctx.tool_registry

    @property
    def task_coordinator(self) -> TaskCoordinator:
        return self._ctx.task_coordinator

    def _save_pending(self) -> None:
        self._ctx.save_pending()

    def _require_llm(self) -> None:
        self._ctx.require_llm()


__all__ = [
    "RESULT_TEXT_LIMIT",
    "RESUME_ACTION_ID",
    "AgentRunContext",
    "ContextService",
    "append_tool_result_turn",
    "require_llm_runtime",
    "truncate_result_text",
]
