"""Canonical agent orchestrator adapter around the LangGraph engine.

`ADR 0003` fixes one execution core (`ToolExecutionService`) and one production
decision loop (the LangGraph runner in `ai_assistant.agents`). This module lets
the durable A2A task lifecycle reuse that decision loop without the application
layer depending on LangGraph: the composition root injects the engine entry
points and this adapter translates between the two public contracts.

No policy lives here. Tool calls still traverse the shared gateway, and the
least-privilege principal resolved by `AgentRunService` is bound for the whole
call so capability scopes are enforced even though the engine was written for
the local desktop caller.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...agents.models import (
    AgentApproveRequest,
    AgentExecuteRequest,
    AgentUiActionResultRequest,
)
from ...domain.security import Principal, bind_principal
from ...domain.tasks import AgentTask

_LANGUAGES = frozenset({"vi", "en"})


class LangGraphAgentOrchestrator:
    """Runs platform tasks on the shared LangGraph agent engine."""

    def __init__(self, execute: Callable[[AgentExecuteRequest], dict[str, Any]],
                 approve: Callable[[AgentApproveRequest], dict[str, Any]],
                 ui_action_result: Callable[[AgentUiActionResultRequest], dict[str, Any]] | None = None) -> None:
        self._execute = execute
        self._approve = approve
        self._ui_action_result = ui_action_result

    def run(self, task: AgentTask, principal: Principal) -> dict[str, Any]:
        with bind_principal(principal):
            continuation = task.metadata.get("agent_run.continuation")
            if isinstance(continuation, dict):
                return self._resume(task, continuation)
            return self._map(self._execute(self._execute_request(task)))

    # ── Fresh run ────────────────────────────────────────────────────────────

    def _execute_request(self, task: AgentTask) -> AgentExecuteRequest:
        language = str(task.metadata.get("language", "vi")).lower()
        if language not in _LANGUAGES:
            language = "vi"
        try:
            temperature = float(task.metadata.get("temperature", 0.2))
        except (TypeError, ValueError):
            temperature = 0.2
        temperature = min(1.5, max(0.0, temperature))
        history = task.metadata.get("history")
        return AgentExecuteRequest(
            task=task.message,
            session_id=task.id,
            language=language,
            temperature=temperature,
            history=history if isinstance(history, list) else [],
        )

    # ── Resume after approval / desktop acknowledgement ──────────────────────

    def _resume(self, task: AgentTask, continuation: dict[str, Any]) -> dict[str, Any]:
        resume = task.metadata.get("agent_run.resume")
        if not isinstance(resume, dict):
            return self._failed("Task continuation is invalid")
        kind = continuation.get("kind")
        if kind == "approval":
            action_id = continuation.get("action_id")
            if not isinstance(action_id, str) or not action_id:
                return self._failed("Task continuation cannot be restored")
            return self._map(self._approve(AgentApproveRequest(
                action_id=action_id,
                approved=resume.get("approved") is True,
                session_id=task.id,
            )))
        if kind == "desktop_ack":
            request_id = continuation.get("request_id")
            payload = resume.get("result")
            if (not isinstance(request_id, str) or not request_id or self._ui_action_result is None
                    or not isinstance(payload, dict)):
                return self._failed("Desktop acknowledgement result is required")
            return self._map(self._ui_action_result(AgentUiActionResultRequest(
                request_id=request_id,
                success=bool(resume.get("success", True)),
                result=payload,
            )))
        return self._failed("Task continuation cannot be restored")

    # ── Qt response → A2A task result ────────────────────────────────────────

    def _map(self, result: dict[str, Any]) -> dict[str, Any]:
        status = result.get("status")
        steps = list(result.get("steps") or [])
        if status == "pending_approval":
            pending = next((step for step in steps if step.get("type") == "pending_approval"), {})
            return {
                "status": "input_required",
                "content": "Approval is required before this tool can run",
                "steps": steps,
                "continuation": {
                    "kind": "approval",
                    "action_id": result.get("action_id", ""),
                    "tool": pending.get("tool"),
                    "params": pending.get("params") or {},
                },
            }
        if status == "pending_ui_action":
            ui_action = result.get("ui_action") or {}
            return {
                "status": "input_required",
                "content": "Waiting for desktop action acknowledgement",
                "steps": steps,
                "continuation": {
                    "kind": "desktop_ack",
                    "request_id": result.get("request_id", ""),
                    "tool": ui_action.get("action", "application_action"),
                    "params": ui_action.get("params") or {},
                },
            }
        if status == "completed":
            return {"status": "completed", "content": self._final_content(steps), "steps": steps}
        if status == "rejected":
            return {"status": "completed", "content": "The requested action was not approved", "steps": steps}
        if status == "cancelled":
            return {"status": "failed", "content": "Agent task was cancelled", "steps": steps}
        return {
            "status": "failed",
            "content": self._final_content(steps) or str(result.get("error") or "Agent execution failed"),
            "steps": steps,
        }

    @staticmethod
    def _final_content(steps: list[dict[str, Any]]) -> str:
        for step in reversed(steps):
            if step.get("type") == "final_answer" and isinstance(step.get("content"), str):
                return step["content"]
        return ""

    @staticmethod
    def _failed(content: str) -> dict[str, Any]:
        return {"status": "failed", "content": content, "steps": []}
