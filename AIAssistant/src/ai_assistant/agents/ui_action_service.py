"""UI action continuation service for desktop acknowledgement workflow.

Extracted from ``AgentService`` (ADR 0004) to separate the desktop UI action
continuation lifecycle from the main execution path. Owns: UI action result
processing, queued action chaining, and LangGraph resume after desktop ack.

All shared run wiring (LLM, tools, supervisor, observability) lives in
:class:`~ai_assistant.agents.context.AgentRunContext`; this service only
supplies the per-invocation arguments.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from ..application.coordination import TaskCoordinator
from ..domain.errors import NotFoundError, UnprocessableRequestError
from ..orchestration.supervisor import audit as audit_agent
from ..orchestration.supervisor import delegate, verify_result
from ..ports import LLMRuntime, ToolGateway
from ..settings import use_langgraph_agent
from ..tools.action_manifest import validate_action_params
from .context import RESUME_ACTION_ID, RESUME_VALIDATE_TOOL_CALL, AgentRunContext
from .ids import generate_action_id
from .models import AgentUiActionResultRequest
from .pending_store import PendingActionStore

logger = logging.getLogger("ai_assistant.agents.ui_action_service")


class UIActionService:
    """Domain service for desktop UI action continuation.

    Handles: UI action result processing, queued action chaining (multi-step
    desktop workflows), and LangGraph resume with the desktop ack injected
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

    # ── Shared wiring surfaced for introspection/tests ──────────────────────
    @property
    def llm_runtime(self) -> LLMRuntime | None:
        return self._ctx.llm_runtime

    @property
    def pending_actions(self) -> PendingActionStore:
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

    def ui_action_result(self, request: AgentUiActionResultRequest) -> dict[str, Any]:
        """Process a desktop UI action result and continue the agent workflow."""
        with self._ctx.pending_lock:
            action = self._ctx.pending_actions.pop(request.request_id, None)
            self._save_pending()
        if action is None or not action.get("ui_ack"):
            msg = "Unknown or expired UI action request"
            raise NotFoundError(msg)

        params = action["params"]
        result = {"success": request.success, "action": params["action"], **request.result}
        steps = [{
            "type": "tool_result",
            "tool": "application_action",
            "request_id": request.request_id,
            "result": result,
            "iteration": action.get("iteration", 0),
        }]

        delegation = delegate(action["task"], action["session_id"], "application_action", params)
        verification = verify_result(delegation, result)
        audit_agent("tool_verified", delegation, **verification)
        steps.append({
            "type": "verification",
            "tool": "application_action",
            "result": verification,
            "iteration": action.get("iteration", 0),
        })

        next_actions = action.get("next_actions") or []
        if request.success and isinstance(next_actions, list) and next_actions:
            return self._chain_next_action(action, steps, next_actions)

        return self._finalize(action, steps, request.success)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _chain_next_action(
        self,
        action: dict[str, Any],
        steps: list[dict[str, Any]],
        next_actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Queue the next UI action in a multi-step desktop workflow."""
        queued = next_actions[0]
        if not isinstance(queued, dict):
            msg = "Invalid queued UI action"
            raise UnprocessableRequestError(msg)
        next_params, error = validate_action_params(queued)
        if error is not None or next_params is None:
            raise UnprocessableRequestError(str(error or "Invalid queued UI action"))
        next_request_id = generate_action_id()
        next_params["request_id"] = next_request_id
        all_steps = action.get("steps", []) + steps + [{
            "type": "tool_call", "tool": "application_action", "params": next_params,
            "iteration": action.get("iteration", 0) + 1,
        }]
        with self._ctx.pending_lock:
            self._ctx.pending_actions[next_request_id] = {
                "ui_ack": True, "params": next_params, "task": action["task"],
                "session_id": action["session_id"], "steps": all_steps,
                "messages": action.get("messages", []), "next_actions": next_actions[1:],
                "iteration": action.get("iteration", 0) + 1,
                "temperature": action.get("temperature", 0.3),
                "language": action.get("language", "vi"), "created_at": time.time(),
            }
            self._save_pending()
        return {
            "status": "pending_ui_action", "session_id": action["session_id"],
            "request_id": next_request_id, "prior_step_count": len(action.get("steps", [])),
            "steps": all_steps,
            "ui_action": {"request_id": next_request_id, "action": next_params["action"], "params": next_params},
        }

    def _finalize(
        self,
        action: dict[str, Any],
        steps: list[dict[str, Any]],
        success: bool,
    ) -> dict[str, Any]:
        """Finalize the UI action workflow, optionally resuming LangGraph."""
        langgraph_available = self._ctx.local_agent_graph is not None
        if action.get("messages") and use_langgraph_agent() and langgraph_available:
            return self._resume_langgraph(action, steps, success)

        content = (f"Đã thực thi {action['params']['action']}." if success
                   else f"Không thể thực thi {action['params']['action']}: unknown error")
        steps.append({"type": "final_answer", "content": content})
        all_steps = action.get("steps", []) + steps
        self._ctx.task_coordinator.finish(action["session_id"], success=success)
        retry_idx_stored = action.get("retry_message_index")
        return {
            "status": "completed" if success else "failed",
            "session_id": action["session_id"],
            "request_id": action.get("request_id", ""),
            "prior_step_count": len(action.get("steps", [])),
            "steps": all_steps,
            **({"retry_message_index": retry_idx_stored} if retry_idx_stored is not None else {}),
        }

    def _resume_langgraph(
        self,
        action: dict[str, Any],
        steps: list[dict[str, Any]],
        success: bool,
    ) -> dict[str, Any]:
        """Resume LangGraph after a desktop UI action acknowledgement."""
        from .prompts import build_agent_system_prompt

        messages = action["messages"]
        tool_call_text = json.dumps({"tool": "application_action", "params": action["params"]}, ensure_ascii=False)
        messages.append({"role": "assistant", "content": f"```tool_call\n{tool_call_text}\n```"})
        result_text = json.dumps({"success": success, "action": action["params"]["action"]}, ensure_ascii=False, indent=2)
        if len(result_text) > 8000:
            result_text = result_text[:8000] + "\n... [truncated]"
        messages.append({
            "role": "user",
            "content": f"Tool `application_action` returned:\n```json\n{result_text}\n```\n\nPhân tích kết quả. NẾU kế hoạch của bạn CÒN bước tiếp theo, hãy bắt buộc GỌI TOOL cho bước đó ngay lập tức (KHÔNG HỎI LẠI NGƯỜI DÙNG). Nếu đã hoàn thành toàn bộ, đưa ra thông báo kết thúc.",
        })

        system_prompt = (messages[0]["content"] if messages and messages[0].get("role") == "system"
                         else build_agent_system_prompt(self._ctx.tool_registry, language=action.get("language", "vi")))
        return self._ctx.run(
            system_prompt=system_prompt,
            task=action["task"],
            session_id=action["session_id"],
            temperature=action["temperature"],
            language=action.get("language", "vi"),
            request_started=time.monotonic(),
            initial_messages=messages,
            initial_steps=action.get("steps", []) + steps,
            initial_iteration=action.get("iteration", 0),
            resume_with_reflection=True,
            prior_step_count=len(action.get("steps", [])),
            validate_tool_call_fn=RESUME_VALIDATE_TOOL_CALL,
            generate_action_id_fn=RESUME_ACTION_ID,
        )


__all__ = ["UIActionService"]
