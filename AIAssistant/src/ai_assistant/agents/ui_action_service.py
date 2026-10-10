"""UI action continuation service for desktop acknowledgement workflow.

Extracted from ``AgentService`` (ADR 0004) to separate the desktop UI action
continuation lifecycle from the main execution path. Owns: UI action result
processing, queued action chaining, and LangGraph resume after desktop ack.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from ..application.coordination import coordinator as task_coordinator
from ..domain.errors import NotFoundError, UnprocessableRequestError
from ..llm.defaults import CHARS_PER_TOKEN, LLM_N_CTX
from ..llm.inference import backend_mode, openai_compatible_completion
from ..observability import (
    langsmith_trace,
    record_schema_error,
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
from ..tools.action_manifest import validate_action_params
from .models import AgentUiActionResultRequest
from .pending_store import PendingActionStore
from .prompts import build_agent_system_prompt
from .runner import run_langgraph_agent

try:
    from ..orchestration.graph import LocalAgentGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LocalAgentGraph = None  # type: ignore[assignment,misc]
    LANGGRAPH_AVAILABLE = False

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
    ) -> None:
        self.llm_runtime = llm_runtime
        self.pending_actions = pending_actions or PendingActionStore("")
        self.pending_lock = pending_lock or threading.Lock()
        self.tool_gateway = tool_gateway
        self.tool_registry = tool_registry
        self._checkpointer = checkpointer

    def ui_action_result(self, request: AgentUiActionResultRequest) -> dict[str, Any]:
        """Process a desktop UI action result and continue the agent workflow."""
        with self.pending_lock:
            action = self.pending_actions.pop(request.request_id, None)
            self._save_pending()
        if action is None or not action.get("ui_ack"):
            raise NotFoundError("Unknown or expired UI action request")

        params = action["params"]
        result = {"success": request.success, "action": params["action"], **request.result}
        steps = []
        steps.append({
            "type": "tool_result",
            "tool": "application_action",
            "request_id": request.request_id,
            "result": result,
            "iteration": action.get("iteration", 0),
        })

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

    def _save_pending(self) -> None:
        self.pending_actions.save()

    def _chain_next_action(
        self,
        action: dict[str, Any],
        steps: list[dict[str, Any]],
        next_actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Queue the next UI action in a multi-step desktop workflow."""
        queued = next_actions[0]
        if not isinstance(queued, dict):
            raise UnprocessableRequestError("Invalid queued UI action")
        next_params, error = validate_action_params(queued)
        if error is not None or next_params is None:
            raise UnprocessableRequestError(str(error or "Invalid queued UI action"))
        next_request_id = _generate_action_id()
        next_params["request_id"] = next_request_id
        all_steps = action.get("steps", []) + steps + [{
            "type": "tool_call", "tool": "application_action", "params": next_params,
            "iteration": action.get("iteration", 0) + 1,
        }]
        with self.pending_lock:
            self.pending_actions[next_request_id] = {
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
        if action.get("messages") and use_langgraph_agent() and LANGGRAPH_AVAILABLE:
            return self._resume_langgraph(action, steps, success)

        content = (f"Đã thực thi {action['params']['action']}." if success
                   else f"Không thể thực thi {action['params']['action']}: unknown error")
        steps.append({"type": "final_answer", "content": content})
        all_steps = action.get("steps", []) + steps
        task_coordinator.finish(action["session_id"], success=success)
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

        def record_tokens(_in: int, _out: int) -> None:
            pass

        system_prompt = messages[0]["content"] if messages and messages[0].get("role") == "system" else build_agent_system_prompt(self.tool_registry, language=action.get("language", "vi"))
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
            record_token_usage_fn=record_tokens,
            tool_registry=self.tool_registry,
            tool_gateway=self.tool_gateway,
            pending_actions=self.pending_actions,
            pending_lock=self.pending_lock,
            task_coordinator=task_coordinator,
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
            initial_steps=action.get("steps", []) + steps,
            initial_iteration=action.get("iteration", 0),
            resume_with_reflection=True,
            prior_step_count=len(action.get("steps", [])),
            LocalAgentGraph=LocalAgentGraph,
            Specialist=Specialist,
            A2ARouter=None,
            checkpointer=self._checkpointer,
            llm_n_ctx=LLM_N_CTX,
            chars_per_token=CHARS_PER_TOKEN,
            generate_action_id_fn=lambda: "",
            save_pending_fn=self._save_pending,
        )


def _generate_action_id() -> str:
    """Generate a unique action ID for UI action requests."""
    import hashlib
    seed = f"{time.time()}:{threading.get_ident()}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()[:12]


__all__ = ["UIActionService"]
