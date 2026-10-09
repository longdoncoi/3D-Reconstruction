"""Unified Agent Service for the 3D-Reconstruction AI Assistant.

Encapsulates Agent execution, cancellation, UI desktop action continuation,
and human-in-the-loop (HITL) approval workflows using LangGraph and Clean Architecture.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request

from ai_assistant.adapters.persistence import PendingActionStore
from ai_assistant.application.coordination import coordinator as task_coordinator
from ai_assistant.bootstrap.runtime import (
    execute_approved_tool as execute_approved_platform_tool,
)

# Legacy/shim references
from ai_assistant.legacy import llm_module as llm_runtime
from ai_assistant.legacy.config import (
    APP_DATA_DIR,
    CHARS_PER_TOKEN,
    LLM_N_CTX,
    PROJECT_DIR,
    USE_LANGGRAPH_AGENT,
    _safe_relpath,
)
from ai_assistant.llm.inference import backend_mode, openai_compatible_completion
from ai_assistant.observability import (
    langsmith_trace,
    record_approval,
    record_schema_error,
    record_tool,
    span,
)
from ai_assistant.orchestration.specialists.code import (
    CodingTaskContext,
    is_coding_task,
)
from ai_assistant.orchestration.specialists.code import (
    instruction as coding_instruction,
)
from ai_assistant.orchestration.supervisor import (
    Specialist,
    delegate,
    reflect_result,
    specialist_instruction,
    verify_result,
)
from ai_assistant.orchestration.supervisor import (
    audit as audit_agent,
)
from ai_assistant.orchestration.supervisor import (
    authorise as authorise_delegation,
)
from ai_assistant.tools.action_manifest import validate_action_params
from ai_assistant.tools.factory import create_tool_registry
from ai_assistant.tools.tool_contract import validate_tool_call

from .completion import (
    constrained_completion,
    parse_tool_call,
)
from .models import (
    AgentApproveRequest,
    AgentCancelRequest,
    AgentExecuteRequest,
    AgentUiActionResultRequest,
)
from .prompts import build_agent_system_prompt
from .runner import run_langgraph_agent

try:
    from ai_assistant.adapters.a2a_protocol import A2ARouter
except ImportError:
    A2ARouter = None  # type: ignore[assignment,misc]

try:
    from ai_assistant.orchestration.graph import LocalAgentGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LocalAgentGraph = None
    LANGGRAPH_AVAILABLE = False

logger = logging.getLogger("ai_assistant.agents.service")

TOOL_REGISTRY = create_tool_registry()
_TOOL_DEFINITIONS = {spec.name: spec.json_schema for spec in TOOL_REGISTRY.get_all()}
_TOOL_PARAM_MODELS = TOOL_REGISTRY.models
_LLAMA_CPP_TOOLS = TOOL_REGISTRY.get_openai_tools()
_TOOL_GRAMMAR_SCHEMA = TOOL_REGISTRY.grammar
_TOOLS_REQUIRING_APPROVAL = {spec.name for spec in TOOL_REGISTRY.get_all() if spec.requires_approval}
_AGENT_MAX_ITERATIONS = 12
AGENT_TOOLS = list(TOOL_REGISTRY.get_all())


def platform_executors() -> dict[str, Callable]:
    executors = {}
    for spec in TOOL_REGISTRY.get_all():
        if spec.handler:
            executors[spec.name] = spec.handler
    return executors


# Pending action store
_pending_lock = threading.Lock()
_PENDING_ACTIONS_FILE = os.path.join(APP_DATA_DIR, "AIAssistant", "pending_agent_actions.json")
_pending_actions = PendingActionStore(_PENDING_ACTIONS_FILE)


def _save_pending_actions() -> None:
    _pending_actions.save()


def _load_pending_actions() -> None:
    _pending_actions.load()


def _generate_action_id() -> str:
    seed = f"{time.time()}:{threading.get_ident()}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()[:12]


def _cleanup_pending_actions() -> None:
    cutoff = time.time() - 600
    with _pending_lock:
        had_expired = _pending_actions.cleanup(cutoff)
        if had_expired:
            _save_pending_actions()
            logger.info("Cleaned up expired pending agent actions")


def _constrained_agent_completion(messages: list[dict], max_tokens: int, temperature: float) -> str:
    """Compatibility entry point for constrained completion."""
    def dummy_record(_in: int, _out: int) -> None:
        pass

    return constrained_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        llm_runtime=llm_runtime,
        backend_mode_fn=backend_mode,
        openai_compatible_fn=openai_compatible_completion,
        openai_tools=_LLAMA_CPP_TOOLS,
        grammar_schema=_TOOL_GRAMMAR_SCHEMA,
        record_token_usage_fn=dummy_record,
    )


def _parse_tool_call(response_text: str) -> tuple[str | None, dict | None]:
    return parse_tool_call(response_text, _TOOL_PARAM_MODELS, validate_tool_call, record_schema_error)


class AgentService:
    """Domain service for full Agent execution lifecycle."""

    def __init__(self) -> None:
        self.pending_actions = _pending_actions
        self.pending_lock = _pending_lock
        self.tool_registry = TOOL_REGISTRY

    def execute(self, request: AgentExecuteRequest, http_req: Request) -> Any:
        _cleanup_pending_actions()
        if llm_runtime.llm is None:
            raise HTTPException(status_code=503, detail="LLM chưa khởi tạo")

        req_start = time.monotonic()
        task = request.task
        session_id = request.session_id or "agent_default"
        task_coordinator.start(session_id, task=request.task)
        retry_idx = request.retry_message_index

        logger.info("[MODE: AGENT] Task from %s: %s…", http_req.client.host if http_req.client else "unknown",
                    task[:80].replace("\n", " "))

        system_prompt = build_agent_system_prompt(self.tool_registry, language=request.language)
        if is_coding_task(task):
            system_prompt += "\n\n" + coding_instruction(CodingTaskContext(
                task=task, language=request.language, project_root=_safe_relpath(PROJECT_DIR, PROJECT_DIR),
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

        def record_tokens(in_tok: int, out_tok: int) -> None:
            pass

        def _run_agent(event_sink: Callable[[dict], None] | None = None) -> dict:
            return run_langgraph_agent(
                system_prompt=system_prompt,
                task=task_with_attachments,
                session_id=session_id,
                temperature=request.temperature,
                language=request.language,
                request_started=req_start,
                llm_runtime=llm_runtime,
                backend_mode_fn=backend_mode,
                openai_compatible_fn=openai_compatible_completion,
                record_token_usage_fn=record_tokens,
                tool_registry=self.tool_registry,
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
                A2ARouter=A2ARouter,
                llm_n_ctx=LLM_N_CTX,
                chars_per_token=CHARS_PER_TOKEN,
                generate_action_id_fn=_generate_action_id,
                save_pending_fn=_save_pending_actions,
            )

        from ai_assistant.adapters.http.agent_routes import _agent_response, _stream_langgraph_execution
        if "text/event-stream" in http_req.headers.get("accept", ""):
            return _stream_langgraph_execution(_run_agent)

        result = _run_agent()
        if retry_idx is not None:
            result["retry_message_index"] = retry_idx
        return _agent_response(result, http_req)

    def cancel(self, request: AgentCancelRequest) -> dict:
        cancelled = task_coordinator.cancel(request.session_id, request.request_id)
        if cancelled is None:
            raise HTTPException(status_code=404, detail="Unknown or already finished agent task")
        return {"status": "cancelled", **cancelled}

    def ui_action_result(self, request: AgentUiActionResultRequest) -> dict:
        _cleanup_pending_actions()
        with self.pending_lock:
            action = self.pending_actions.pop(request.request_id, None)
            _save_pending_actions()
        if action is None or not action.get("ui_ack"):
            raise HTTPException(status_code=404, detail="Unknown or expired UI action request")

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
            queued = next_actions[0]
            if not isinstance(queued, dict):
                raise HTTPException(status_code=422, detail="Invalid queued UI action")
            next_params, error = validate_action_params(queued)
            if error:
                raise HTTPException(status_code=422, detail=error)
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
                _save_pending_actions()
            return {
                "status": "pending_ui_action", "session_id": action["session_id"],
                "request_id": next_request_id, "prior_step_count": len(action.get("steps", [])),
                "steps": all_steps,
                "ui_action": {"request_id": next_request_id, "action": next_params["action"], "params": next_params},
            }

        if action.get("messages") and USE_LANGGRAPH_AGENT and LANGGRAPH_AVAILABLE:
            messages = action["messages"]
            tool_call_text = json.dumps({"tool": "application_action", "params": params}, ensure_ascii=False)
            messages.append({"role": "assistant", "content": f"```tool_call\n{tool_call_text}\n```"})
            result_text = json.dumps(result, ensure_ascii=False, indent=2)
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
                llm_runtime=llm_runtime,
                backend_mode_fn=backend_mode,
                openai_compatible_fn=openai_compatible_completion,
                record_token_usage_fn=record_tokens,
                tool_registry=self.tool_registry,
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
                validate_tool_call_fn=validate_tool_call,
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
                A2ARouter=A2ARouter,
                llm_n_ctx=LLM_N_CTX,
                chars_per_token=CHARS_PER_TOKEN,
                generate_action_id_fn=_generate_action_id,
                save_pending_fn=_save_pending_actions,
            )

        content = (f"Đã thực thi {params['action']}." if request.success
                   else f"Không thể thực thi {params['action']}: {result.get('error', 'unknown error')}")
        steps.append({"type": "final_answer", "content": content})
        all_steps = action.get("steps", []) + steps
        task_coordinator.finish(action["session_id"], success=request.success)
        retry_idx_stored = action.get("retry_message_index")
        return {
            "status": "completed" if request.success else "failed",
            "session_id": action["session_id"],
            "request_id": request.request_id,
            "prior_step_count": len(action.get("steps", [])),
            "steps": all_steps,
            **({"retry_message_index": retry_idx_stored} if retry_idx_stored is not None else {}),
        }

    def approve(self, request: AgentApproveRequest, http_req: Request) -> dict:
        _cleanup_pending_actions()
        if llm_runtime.llm is None:
            raise HTTPException(status_code=503, detail="LLM chưa khởi tạo")

        action_id = request.action_id
        with self.pending_lock:
            action = self.pending_actions.pop(action_id, None)
            _save_pending_actions()

        if action is None:
            record_approval("missing")
            raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")

        if task_coordinator.is_cancelled(action.get("session_id", "")):
            task_coordinator.finish(action.get("session_id", ""), success=False)
            return {"status": "cancelled", "action_id": action_id,
                    "steps": [*action.get("steps", []), {
                        "type": "cancelled", "content": "Task cancelled before approval."}]}

        if request.session_id and request.session_id != action.get("session_id"):
            with self.pending_lock:
                self.pending_actions[action_id] = action
                _save_pending_actions()
            record_approval("unauthorized")
            raise HTTPException(status_code=403, detail="Action does not belong to this session")

        if not request.approved:
            record_approval("rejected")
            return {
                "status": "rejected",
                "action_id": action_id,
                "approval_scope": action.get("approval_scope", ""),
                "approval_preview": action.get("approval_preview"),
                "prior_step_count": len(action["steps"]),
                "steps": action["steps"] + [{
                    "type": "tool_result",
                    "tool": action["tool"],
                    "action_id": action_id,
                    "result": {"rejected": True, "message": "Người dùng từ chối thực thi action này."},
                    "iteration": action["iteration"],
                }],
            }

        record_approval("approved")
        tool_name = action["tool"]
        tool_params = action["params"]
        approved_tool_started = time.monotonic()
        tool_result = execute_approved_platform_tool(tool_name, tool_params)
        record_tool(tool_name, "error" not in tool_result, time.monotonic() - approved_tool_started)

        prior_step_count = len(action["steps"])
        steps = action["steps"]
        steps.append({
            "type": "tool_result",
            "tool": tool_name,
            "action_id": action_id,
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

        if USE_LANGGRAPH_AGENT and LANGGRAPH_AVAILABLE:
            system_prompt = messages[0]["content"] if messages and messages[0].get("role") == "system" else build_agent_system_prompt(self.tool_registry, language=action.get("language", "vi"))
            steps.append({"type": "approval_granted", "scope_id": action.get("approval_scope", ""),
                          "tool": tool_name, "action_id": action_id})

            def record_tokens(_in: int, _out: int) -> None:
                pass

            return run_langgraph_agent(
                system_prompt=system_prompt,
                task=action["task"],
                session_id=action["session_id"],
                temperature=action["temperature"],
                language=action.get("language", "vi"),
                request_started=time.monotonic(),
                llm_runtime=llm_runtime,
                backend_mode_fn=backend_mode,
                openai_compatible_fn=openai_compatible_completion,
                record_token_usage_fn=record_tokens,
                tool_registry=self.tool_registry,
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
                validate_tool_call_fn=validate_tool_call,
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
                LocalAgentGraph=LocalAgentGraph,
                Specialist=Specialist,
                A2ARouter=A2ARouter,
                llm_n_ctx=LLM_N_CTX,
                chars_per_token=CHARS_PER_TOKEN,
                generate_action_id_fn=_generate_action_id,
                save_pending_fn=_save_pending_actions,
            )

        # Fallback if LangGraph unavailable
        return {
            "status": "completed",
            "session_id": action["session_id"],
            "prior_step_count": prior_step_count,
            "steps": steps,
        }

    def reset_state(self) -> None:
        with self.pending_lock:
            self.pending_actions.clear()
            try:
                if os.path.exists(_PENDING_ACTIONS_FILE):
                    os.remove(_PENDING_ACTIONS_FILE)
            except OSError as error:
                logger.warning("Unable to remove pending action state: %s", error)


# Default singleton instance and top-level function bridges
_default_service = AgentService()


def agent_execute(request: AgentExecuteRequest, http_req: Request) -> Any:
    return _default_service.execute(request, http_req)


def agent_cancel(request: AgentCancelRequest) -> dict:
    return _default_service.cancel(request)


def agent_ui_action_result(request: AgentUiActionResultRequest) -> dict:
    return _default_service.ui_action_result(request)


def agent_approve(request: AgentApproveRequest, http_req: Request) -> dict:
    return _default_service.approve(request, http_req)


def reset_agent_state() -> None:
    _default_service.reset_state()


__all__ = [
    "AGENT_TOOLS",
    "TOOL_REGISTRY",
    "AgentApproveRequest",
    "AgentCancelRequest",
    "AgentExecuteRequest",
    "AgentService",
    "AgentUiActionResultRequest",
    "_constrained_agent_completion",
    "_load_pending_actions",
    "_parse_tool_call",
    "_pending_actions",
    "_pending_lock",
    "_save_pending_actions",
    "agent_approve",
    "agent_cancel",
    "agent_execute",
    "agent_ui_action_result",
    "backend_mode",
    "llm_runtime",
    "platform_executors",
    "reset_agent_state",
]
