"""Unified Agent Service facade for the 3D-Reconstruction AI Assistant.

This module provides the main entry point for Agent execution, delegating to:
- ``ApprovalService`` — Human-in-the-Loop approval workflows
- ``UIActionService`` — Desktop UI action continuation

The facade pattern preserves the existing public interface while enabling
independent testing and evolution of each sub-service (ADR 0004).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable

from ..application.coordination import TaskCoordinator
from ..config.paths import get_paths
from ..domain.errors import ModelNotLoadedError, NotFoundError
from ..llm.defaults import CHARS_PER_TOKEN, LLM_N_CTX
from ..llm.inference import backend_mode, openai_compatible_completion
from ..observability import (
    langsmith_trace,
    record_schema_error,
    record_token_usage,
    record_tool,
    span,
)
from ..orchestration.specialists.code import (
    CodingTaskContext,
    is_coding_task,
)
from ..orchestration.specialists.code import (
    instruction as coding_instruction,
)
from ..orchestration.supervisor import (
    Specialist,
    delegate,
    reflect_result,
    specialist_instruction,
    verify_result,
)
from ..orchestration.supervisor import (
    audit as audit_agent,
)
from ..orchestration.supervisor import (
    authorise as authorise_delegation,
)
from ..ports import LLMRuntime, ToolGateway
from ..tools.factory import create_tool_registry
from ..tools.tool_contract import validate_tool_call
from .approval_service import ApprovalService
from .ids import generate_action_id
from .models import (
    AgentApproveRequest,
    AgentCancelRequest,
    AgentExecuteRequest,
    AgentUiActionResultRequest,
)
from .pending_store import PendingActionStore
from .prompts import build_agent_system_prompt
from .runner import run_langgraph_agent
from .ui_action_service import UIActionService

try:
    from ..orchestration.graph import LocalAgentGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LocalAgentGraph = None  # type: ignore[assignment,misc]
    LANGGRAPH_AVAILABLE = False

logger = logging.getLogger("ai_assistant.agents.service")


def _default_pending_path() -> str:
    """Canonical on-disk location for pending approvals and UI actions."""
    paths = get_paths()
    data_root = paths.data_dir if paths.data_dir.name == "AIAssistant" else paths.data_dir / "AIAssistant"
    return str(data_root / "pending_agent_actions.json")


class AgentService:
    """Domain service for full Agent execution lifecycle.

    Facade that delegates to:
    - ``ApprovalService`` for HITL approval workflows
    - ``UIActionService`` for desktop UI action continuation
    """

    def __init__(self, *, llm_runtime: LLMRuntime | None = None,
                 pending_actions: PendingActionStore | None = None,
                 pending_lock=None, tool_registry=None,
                 tool_gateway: ToolGateway | None = None,
                 checkpointer=None, a2a_router=None,
                 task_coordinator: TaskCoordinator | None = None) -> None:
        self.llm_runtime = llm_runtime
        self.pending_actions = (
            pending_actions if pending_actions is not None else PendingActionStore(_default_pending_path())
        )
        self.pending_lock = pending_lock if pending_lock is not None else threading.Lock()
        # The composition root owns one catalog and one coordinator; a bare
        # instance (tests, embeds) builds its own so the facade stays usable.
        self.tool_registry = tool_registry if tool_registry is not None else create_tool_registry()
        self.tool_gateway = tool_gateway
        self.task_coordinator = task_coordinator if task_coordinator is not None else TaskCoordinator()
        # LangGraph persistence and the A2A transport are adapter concerns
        # injected by the composition root; ``None`` keeps the facade runnable
        # without them (stateless graphs, local-only dispatch).
        self._checkpointer = checkpointer
        self._a2a_router = a2a_router
        # Durability (ADR 0003): rehydrate approvals/desktop acks persisted by a
        # previous process so cross-restart A2A continuation still resolves.
        self.pending_actions.load()

        # Sub-services share the same pending state, tool gateway, coordinator
        # and graph persistence; the A2A router stays on the semantic-path
        # delegations.
        self._approval_service = ApprovalService(
            llm_runtime=llm_runtime,
            pending_actions=self.pending_actions,
            pending_lock=self.pending_lock,
            tool_gateway=tool_gateway,
            tool_registry=self.tool_registry,
            checkpointer=self._checkpointer,
            task_coordinator=self.task_coordinator,
        )
        self._ui_action_service = UIActionService(
            llm_runtime=llm_runtime,
            pending_actions=self.pending_actions,
            pending_lock=self.pending_lock,
            tool_gateway=tool_gateway,
            tool_registry=self.tool_registry,
            checkpointer=self._checkpointer,
            task_coordinator=self.task_coordinator,
        )

    def _require_llm(self) -> None:
        if self.llm_runtime is None or self.llm_runtime.llm is None:
            raise ModelNotLoadedError("LLM chưa khởi tạo")

    def _save_pending(self) -> None:
        self.pending_actions.save()

    def _cleanup_pending(self) -> None:
        cutoff = time.time() - 600
        with self.pending_lock:
            if self.pending_actions.cleanup(cutoff):
                self._save_pending()
                logger.info("Cleaned up expired pending agent actions")

    def execute(self, request: AgentExecuteRequest, *, event_sink: Callable[[dict], None] | None = None,
                client_host: str = "unknown") -> dict:
        self._cleanup_pending()
        self._require_llm()

        req_start = time.monotonic()
        task = request.task
        session_id = request.session_id or "agent_default"
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

        task_with_attachments = task
        if request.attachments:
            names = [os.path.basename(path) for path in request.attachments]
            task_with_attachments += "\n\n[Attached files: " + ", ".join(names) + "]"

        initial_msgs = [
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": task_with_attachments},
        ]

        def _run_agent(event_sink: Callable[[dict], None] | None = None) -> dict:
            return run_langgraph_agent(
                system_prompt=system_prompt,
                task=task_with_attachments,
                session_id=session_id,
                temperature=request.temperature,
                language=request.language,
                request_started=req_start,
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
                validate_tool_call_fn=validate_tool_call,
                langsmith_trace_fn=langsmith_trace,
                span_fn=span,
                specialist_instruction_fn=specialist_instruction,
                is_coding_task_fn=is_coding_task,
                initial_messages=initial_msgs,
                event_sink=event_sink,
                supervisor_route=Specialist.SUPERVISOR,
                LocalAgentGraph=LocalAgentGraph,
                Specialist=Specialist,
                a2a_router=self._a2a_router,
                checkpointer=self._checkpointer,
                llm_n_ctx=LLM_N_CTX,
                chars_per_token=CHARS_PER_TOKEN,
                generate_action_id_fn=generate_action_id,
                save_pending_fn=self._save_pending,
            )

        # Transport negotiation (JSON vs SSE) belongs to the HTTP adapter; the
        # service returns a plain result and optionally streams steps via event_sink.
        result = _run_agent(event_sink)
        if retry_idx is not None:
            result["retry_message_index"] = retry_idx
        return result

    def cancel(self, request: AgentCancelRequest) -> dict:
        cancelled = self.task_coordinator.cancel(request.session_id, request.request_id)
        if cancelled is None:
            raise NotFoundError("Unknown or already finished agent task")
        return {"status": "cancelled", **cancelled}

    def approve(self, request: AgentApproveRequest) -> dict:
        """Delegate to ApprovalService for HITL approval workflows."""
        self._cleanup_pending()
        return self._approval_service.approve(request)

    def ui_action_result(self, request: AgentUiActionResultRequest) -> dict:
        """Delegate to UIActionService for desktop UI action continuation."""
        self._cleanup_pending()
        return self._ui_action_service.ui_action_result(request)

    def reset_state(self) -> None:
        with self.pending_lock:
            self.pending_actions.clear()
            try:
                import os
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
