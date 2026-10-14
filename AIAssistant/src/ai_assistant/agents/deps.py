"""Injected dependency bundle for one agent run (ADR 0013).

``run_langgraph_agent`` used to accept 46 keyword-only parameters: the LLM
runtime and its completion adapters, the tool catalog and gateway, the pending
store and its lock, the task coordinator, the supervisor policy callables, the
observability hooks, the graph engine class and the A2A transport. Roughly
twenty of those never vary between invocations, so the signature was a
dependency-injection container spelled out as an argument list, and wiring one
new dependency meant editing every call site.

:class:`RunDeps` owns that block once as a frozen, immutable record. The
runner keeps only the arguments that actually vary per invocation (prompt,
task, resume state, approval grant, event sink). ``AgentRunContext`` builds
the bundle from its own injected dependencies, so the three agent sub-services
still see no wiring at all.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..llm.defaults import CHARS_PER_TOKEN, LLM_N_CTX
from ..ports import LLMRuntime, ToolExecutionGateway
from ..tools.registry import ToolRegistry
from .pending_store import PendingActionStore


@dataclass(frozen=True, slots=True)
class RunDeps:
    """Everything one agent run needs that does not vary per invocation."""

    # ── LLM access ────────────────────────────────────────────────────────
    llm_runtime: LLMRuntime | None
    backend_mode_fn: Callable[[], str]
    openai_compatible_fn: Callable[..., dict]
    record_token_usage_fn: Callable[[int, int], None]

    # ── Tool catalog, policy gateway and persistence ──────────────────────
    tool_registry: ToolRegistry
    pending_actions: PendingActionStore
    pending_lock: Any
    task_coordinator: Any

    # ── Supervisor policy (delegation, verification, reflection) ──────────
    delegate_fn: Callable[..., Any]
    authorise_delegation_fn: Callable[..., tuple[bool, str | None]]
    verify_result_fn: Callable[..., dict]
    reflect_result_fn: Callable[..., dict]
    is_coding_task_fn: Callable[[str], bool]

    # ── Observability hooks ───────────────────────────────────────────────
    audit_agent_fn: Callable[..., None]
    record_tool_fn: Callable[..., None]
    record_schema_error_fn: Callable[..., None]
    langsmith_trace_fn: Callable[..., Any]
    span_fn: Callable[..., Any]

    # ── Optional wiring, injected by the composition root ─────────────────
    # The runner reaches only the *execution* surface of the gateway: it can
    # execute, pre-flight and probe approval grants but can never mint one.
    # Grant minting is confined to the approval services behind
    # ``ApprovalGrantIssuer`` (review finding 15 — separation of duties).
    tool_gateway: ToolExecutionGateway | None = None
    #: The graph engine class; ``None`` means LangGraph is unavailable.
    local_agent_graph: Any = None
    #: Remote A2A transport; ``None`` keeps dispatch local.
    a2a_router: Any = None
    checkpointer: Any = None

    # ── Token budget ──────────────────────────────────────────────────────
    llm_n_ctx: int = LLM_N_CTX
    chars_per_token: float = CHARS_PER_TOKEN

    # ── Persistence hooks ─────────────────────────────────────────────────
    save_pending_fn: Callable[[], None] | None = None


__all__ = ["RunDeps"]
