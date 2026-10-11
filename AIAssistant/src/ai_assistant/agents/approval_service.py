"""Approval workflow service for Human-in-the-Loop (HITL) tool execution.

Extracted from ``AgentService`` (ADR 0004) to separate the approval lifecycle
from the main execution path. Owns: session binding, rejection, execution,
and LangGraph resume after approval.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from ..application.coordination import TaskCoordinator
from ..domain.approvals import approval_fingerprint
from ..domain.errors import AuthorizationError, NotFoundError
from ..llm.defaults import CHARS_PER_TOKEN, LLM_N_CTX
from ..llm.inference import backend_mode, openai_compatible_completion
from ..observability import (
    langsmith_trace,
    record_approval,
    record_schema_error,
    record_token_usage,
    record_tool,
    span,
)
from ..orchestration.specialists.code import (
    is_coding_task,
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
from ..settings import use_langgraph_agent
from .models import AgentApproveRequest
from .pending_store import PendingActionStore
from .prompts import build_agent_system_prompt
from .runner import run_langgraph_agent

try:
    from ..orchestration.graph import LocalAgentGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LocalAgentGraph = None  # type: ignore[assignment,misc]
    LANGGRAPH_AVAILABLE = False

logger = logging.getLogger("ai_assistant.agents.approval_service")


class ApprovalService:
    """Domain service for the Human-in-the-Loop approval lifecycle.

    Handles: session binding validation, rejection, tool execution after
    approval, and LangGraph resume with the approved tool result injected
    into the message history.
    """

    def __init__(
        self,
        *,
        llm_runtime: LLMRuntime | None = None,
        pending_actions: PendingActionStore | None = None,
        pending_lock: threading.Lock | None = None,
        tool_gateway: ToolGateway | None = None,
        tool_registry=None,
        checkpointer=None,
        task_coordinator: TaskCoordinator | None = None,
    ) -> None:
        self.llm_runtime = llm_runtime
        self.pending_actions = pending_actions if pending_actions is not None else PendingActionStore("")
        self.pending_lock = pending_lock or threading.Lock()
        self.tool_gateway = tool_gateway
        self.tool_registry = tool_registry
        self._checkpointer = checkpointer
        self.task_coordinator = task_coordinator if task_coordinator is not None else TaskCoordinator()

    def approve(self, request: AgentApproveRequest) -> dict[str, Any]:
        """Process an approval or rejection decision for a pending tool call."""
        self._require_llm()

        action_id = request.action_id
        with self.pending_lock:
            action = self.pending_actions.pop(action_id, None)
            self._save_pending()

        if action is None:
            record_approval("missing")
            raise NotFoundError(f"Action not found: {action_id}")
        action["action_id"] = action_id

        if self.task_coordinator.is_cancelled(action.get("session_id", "")):
            self.task_coordinator.finish(action.get("session_id", ""), success=False)
            return {"status": "cancelled", "action_id": action_id,
                    "steps": [*action.get("steps", []), {
                        "type": "cancelled", "content": "Task cancelled before approval."}]}

        # Fail closed: a missing session id is treated exactly like a mismatch,
        # otherwise an unauthenticated caller could approve another session's
        # pending action by simply omitting the field.
        if not request.session_id or request.session_id != action.get("session_id"):
            with self.pending_lock:
                self.pending_actions[action_id] = action
                self._save_pending()
            record_approval("unauthorized")
            raise AuthorizationError("Action does not belong to this session")

        if not request.approved:
            record_approval("rejected")
            return self._rejected(action)

        record_approval("approved")
        return self._approved(action)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _require_llm(self) -> None:
        if self.llm_runtime is None or self.llm_runtime.llm is None:
            from ..domain.errors import ModelNotLoadedError
            raise ModelNotLoadedError("LLM chưa khởi tạo")

    def _save_pending(self) -> None:
        self.pending_actions.save()

    def _rejected(self, action: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "rejected",
            "action_id": action.get("action_id", ""),
            "approval_scope": action.get("approval_scope", ""),
            "approval_preview": action.get("approval_preview"),
            "prior_step_count": len(action["steps"]),
            "steps": action["steps"] + [{
                "type": "tool_result",
                "tool": action["tool"],
                "action_id": action.get("action_id", ""),
                "result": {"rejected": True, "message": "Người dùng từ chối thực thi action này."},
                "iteration": action["iteration"],
            }],
        }

    def _approved(self, action: dict[str, Any]) -> dict[str, Any]:
        tool_name = action["tool"]
        tool_params = action["params"]
        fingerprint = approval_fingerprint(tool_name, tool_params)
        # One approval authorises exactly two single-use grants, both bound to
        # this invocation: one for the direct execution performed here and one
        # for at most one re-dispatch of the identical call inside the resumed
        # graph. Any other tool or parameter set asks the user again.
        run_token = action.get("action_id") or f"approval-{fingerprint[:16]}"
        direct_token = f"{run_token}#direct"
        if self.tool_gateway is not None:
            self.tool_gateway.issue_approval_grant(tool_name, tool_params, run_token)
            self.tool_gateway.issue_approval_grant(tool_name, tool_params, direct_token)
        tool_result = self._execute_approved(tool_name, tool_params, direct_token)
        record_tool(tool_name, "error" not in tool_result, time.monotonic())

        prior_step_count = len(action["steps"])
        steps = action["steps"]
        steps.append({
            "type": "tool_result",
            "tool": tool_name,
            "action_id": action.get("action_id", ""),
            "result": tool_result,
            "iteration": action["iteration"],
        })
        delegation = delegate(action["task"], action["session_id"], tool_name, tool_params)
        verification = verify_result(delegation, tool_result)
        audit_agent("tool_verified", delegation, **verification)
        steps.append({
            "type": "verification",
            "tool": tool_name,
            "result": verification,
            "iteration": action["iteration"],
        })

        messages = action["messages"]
        tool_call_text = json.dumps({"tool": tool_name, "params": tool_params}, ensure_ascii=False)
        messages.append({"role": "assistant", "content": f"```tool_call\n{tool_call_text}\n```"})
        result_text = json.dumps(tool_result, ensure_ascii=False, indent=2)
        if len(result_text) > 8000:
            result_text = result_text[:8000] + "\n... [truncated]"
        messages.append({
            "role": "user",
            "content": f"Tool `{tool_name}` was approved and executed. Result:\n```json\n{result_text}\n```\n\nContinue with your analysis or provide final answer.",
        })

        if use_langgraph_agent() and LANGGRAPH_AVAILABLE:
            system_prompt = messages[0]["content"] if messages and messages[0].get("role") == "system" else build_agent_system_prompt(self.tool_registry, language=action.get("language", "vi"))
            steps.append({"type": "approval_granted", "scope_id": action.get("approval_scope", ""),
                          "tool": tool_name, "action_id": action.get("action_id", "")})

            return run_langgraph_agent(
                system_prompt=system_prompt,
                task=action["task"],
                session_id=action["session_id"],
                temperature=action["temperature"],
                language=action.get("language", "vi"),
                request_started=time.monotonic(),
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
                validate_tool_call_fn=lambda *a, **kw: None,
                langsmith_trace_fn=langsmith_trace,
                span_fn=span,
                specialist_instruction_fn=specialist_instruction,
                is_coding_task_fn=is_coding_task,
                initial_messages=messages,
                initial_steps=steps,
                initial_iteration=action["iteration"],
                resume_with_reflection=True,
                prior_step_count=prior_step_count,
                approval_granted=True,
                approval_scope=action.get("approval_scope", ""),
                approval_token=run_token,
                granted_fingerprint=fingerprint,
                LocalAgentGraph=LocalAgentGraph,
                Specialist=Specialist,
                a2a_router=None,
                checkpointer=self._checkpointer,
                llm_n_ctx=LLM_N_CTX,
                chars_per_token=CHARS_PER_TOKEN,
                generate_action_id_fn=lambda: "",
                save_pending_fn=self._save_pending,
            )

        return {
            "status": "completed",
            "session_id": action["session_id"],
            "prior_step_count": prior_step_count,
            "steps": steps,
        }

    def _execute_approved(self, tool_name: str, tool_params: dict, approval_token: str) -> dict[str, Any]:
        if self.tool_gateway is None:
            return {"success": False, "error_code": "runtime_unconfigured",
                    "error": "AI Agent Platform is not bootstrapped"}
        return self.tool_gateway.execute_approved(tool_name, tool_params, approval_token)


__all__ = ["ApprovalService"]
