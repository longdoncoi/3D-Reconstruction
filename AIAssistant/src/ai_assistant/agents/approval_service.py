"""Approval workflow service for Human-in-the-Loop (HITL) tool execution.

Extracted from ``AgentService`` (ADR 0004) to separate the approval lifecycle
from the main execution path. Owns: session binding, rejection, execution,
and LangGraph resume after approval.

All shared run wiring (LLM, tools, supervisor, observability) lives in
:class:`~ai_assistant.agents.context.AgentRunContext`; this service only
supplies the per-invocation arguments.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from ..application.coordination import TaskCoordinator
from ..domain.approvals import approval_fingerprint, issue_grant_pair
from ..domain.errors import AuthorizationError, NotFoundError
from ..observability import record_approval, record_tool
from ..orchestration.supervisor import audit as audit_agent
from ..orchestration.supervisor import delegate, verify_result
from ..ports import LLMRuntime, ToolGateway
from ..tools.tool_contract import RUNTIME_UNCONFIGURED_RESULT
from .context import RESUME_ACTION_ID, AgentRunContext, ContextService, append_tool_result_turn
from .models import AgentApproveRequest
from .pending_store import PendingActionStore

logger = logging.getLogger("ai_assistant.agents.approval_service")


class ApprovalService(ContextService):
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
        self._ctx = AgentRunContext(
            llm_runtime=llm_runtime,
            pending_actions=pending_actions if pending_actions is not None else PendingActionStore(""),
            pending_lock=pending_lock or threading.Lock(),
            tool_gateway=tool_gateway,
            tool_registry=tool_registry,
            checkpointer=checkpointer,
            task_coordinator=task_coordinator if task_coordinator is not None else TaskCoordinator(),
        )

    def approve(self, request: AgentApproveRequest) -> dict[str, Any]:
        """Process an approval or rejection decision for a pending tool call."""
        self._require_llm()

        action_id = request.action_id
        with self._ctx.pending_lock:
            action = self._ctx.pending_actions.pop(action_id, None)
            self._save_pending()

        if action is None:
            record_approval("missing")
            msg = f"Action not found: {action_id}"
            raise NotFoundError(msg)
        action["action_id"] = action_id

        if self._ctx.task_coordinator.is_cancelled(action.get("session_id", "")):
            self._ctx.task_coordinator.finish(action.get("session_id", ""), success=False)
            return {"status": "cancelled", "action_id": action_id,
                    "steps": [*action.get("steps", []), {
                        "type": "cancelled", "content": "Task cancelled before approval."}]}

        # Fail closed: a missing session id is treated exactly like a mismatch,
        # otherwise an unauthenticated caller could approve another session's
        # pending action by simply omitting the field.
        if not request.session_id or request.session_id != action.get("session_id"):
            with self._ctx.pending_lock:
                self._ctx.pending_actions[action_id] = action
                self._save_pending()
            record_approval("unauthorized")
            msg = "Action does not belong to this session"
            raise AuthorizationError(msg)

        if not request.approved:
            record_approval("rejected")
            return self._rejected(action)

        record_approval("approved")
        return self._approved(action)

    # ── Private helpers ─────────────────────────────────────────────────────

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
        # this invocation (minted by the shared domain policy): one for the
        # direct execution performed here and one for at most one re-dispatch
        # of the identical call inside the resumed graph.
        run_token, direct_token = issue_grant_pair(
            tool_name, tool_params, run_id=action.get("action_id", ""),
        )
        if self._ctx.tool_gateway is not None:
            self._ctx.tool_gateway.issue_approval_grant(tool_name, tool_params, run_token)
            self._ctx.tool_gateway.issue_approval_grant(tool_name, tool_params, direct_token)
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
        append_tool_result_turn(
            messages, tool_name, tool_params, tool_result,
            lead=f"Tool `{tool_name}` was approved and executed. Result:",
            tail="Continue with your analysis or provide final answer.",
        )

        if self._ctx.resume_available():
            from .prompts import build_agent_system_prompt

            system_prompt = (messages[0]["content"] if messages and messages[0].get("role") == "system"
                             else build_agent_system_prompt(self._ctx.tool_registry,
                                                            language=action.get("language", "vi")))
            steps.append({"type": "approval_granted", "scope_id": action.get("approval_scope", ""),
                          "tool": tool_name, "action_id": action.get("action_id", "")})

            return self._ctx.run(
                system_prompt=system_prompt,
                task=action["task"],
                session_id=action["session_id"],
                temperature=action["temperature"],
                language=action.get("language", "vi"),
                request_started=time.monotonic(),
                initial_messages=messages,
                initial_steps=steps,
                initial_iteration=action["iteration"],
                resume_with_reflection=True,
                prior_step_count=prior_step_count,
                approval_granted=True,
                approval_scope=action.get("approval_scope", ""),
                approval_token=run_token,
                granted_fingerprint=fingerprint,
                generate_action_id_fn=RESUME_ACTION_ID,
            )

        return {
            "status": "completed",
            "session_id": action["session_id"],
            "prior_step_count": prior_step_count,
            "steps": steps,
        }

    def _execute_approved(self, tool_name: str, tool_params: dict, approval_token: str) -> dict[str, Any]:
        if self._ctx.tool_gateway is None:
            return dict(RUNTIME_UNCONFIGURED_RESULT)
        return self._ctx.tool_gateway.execute_approved(tool_name, tool_params, approval_token)


__all__ = ["ApprovalService"]
