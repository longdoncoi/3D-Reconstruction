"""Tool execution node for the LangGraph orchestration.

This node executes the tool call selected by the Reasoner and updates
the state with the result, including observation summarization for RAG.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ai_assistant.config.logging import get_agent_logger
from ai_assistant.domain.governance import redact_value

from ..state import (
    COMPLETION_TOOLS,
    LOW_RISK_TOOLS,
    SEMANTIC_REFLECTION_TOOLS,
    AgentState,
    Executor,
    VerifyResult,
    has_step_type,
)
from .summarize import summarize_messages

logger = get_agent_logger("tool")


@dataclass
class ToolContext:
    """Dependencies for the tool node."""
    execute: Executor
    verify_result: VerifyResult | None
    cancel_checker: Callable[[], bool]


def _rag_synthesis_update(state: AgentState, steps: list[dict[str, Any]],
                          tool_name: str, result: dict[str, Any],
                          last_step: dict[str, Any], error_count: int) -> dict[str, Any] | None:
    """Build the step_answer synthesis request for verified RAG evidence.

    Returns the state update when the executed tool was ``rag_search``,
    else ``None`` so the caller continues with the generic routing.
    """
    if tool_name != "rag_search":
        return None
    evidence = json.dumps(result, ensure_ascii=False, indent=2)
    evidence_limit = 6000
    evidence_snippet = evidence[:evidence_limit] + ("\n... [truncated]" if len(evidence) > evidence_limit else "")

    rag_plan_step = ""
    if isinstance(last_step.get("plan_step_index"), int):
        rag_step_idx = last_step["plan_step_index"]
        plan_now = state.get("plan") or []
        if 0 <= rag_step_idx < len(plan_now):
            rag_plan_step = plan_now[rag_step_idx]
    step_directive = (
        f' Trả lời ngắn gọn ĐÚNG bước: "{rag_plan_step}"'
        " — không lặp lại ý đã nêu ở các bước trước."
        if rag_plan_step else ""
    )
    role_answer_directive = (
        " Chỉ nêu thông tin trong phạm vi bước hiện tại; KHÔNG lặp lại nội dung"
        " đã trả lời ở các bước trước. Trình bày tự nhiên, mạch lạc như đang giải"
        " thích cho người dùng; KHÔNG ép câu trả lời vào khung mục"
        " 'Thông tin / Vai trò / Nhiệm vụ-Trách nhiệm', và nếu thiếu dữ liệu thì"
        " nói rõ dữ liệu chưa có thay vì bịa đặt."
    )

    messages = summarize_messages(list(state["messages"]))
    messages.append({
        "role": "user",
        "content": (
            "RAG evidence đã được xác minh." + step_directive + role_answer_directive
            + " Dùng bằng chứng dưới đây để trả lời ngắn gọn, tự nhiên."
            " KHÔNG đề cập tiêu đề nguồn, tên file, số trích dẫn,"
            " 'TÀI LIỆU THAM KHẢO' hay 'MÃ NGUỒN LIÊN QUAN'."
            ' Trả về {"kind":"step_answer","content":"..."} — KHÔNG dùng final.\n\n'
            f"Bằng chứng:\n```json\n{evidence_snippet}\n```"
        ),
    })
    logger.info("[NODE: tool] Project RAG evidence ready; requesting step_answer synthesis.")
    return {"steps": steps, "messages": messages,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "error_count": error_count, "synthesize_after_rag": True}


def _completion_announcement_update(state: AgentState, steps: list[dict[str, Any]],
                                      tool_name: str, result: dict[str, Any],
                                      error_count: int) -> dict[str, Any] | None:
    """Announce a dispatched desktop action instead of reflecting on it.

    Returns the state update when the tool completed a user-visible action,
    else ``None`` so the caller continues with the generic routing.
    """
    if tool_name not in COMPLETION_TOOLS or not result.get("success"):
        return None
    messages = list(state["messages"])
    messages.append({
        "role":    "user",
        "content": (
            f"Tool `{tool_name}` đã thực thi thành công (action: {result.get('action', tool_name)}). "
            f"Hãy thông báo kết quả ngắn gọn cho người dùng."
        ),
    })
    return {
        "steps":            steps,
        "messages":         messages,
        "tool_call_count":  state.get("tool_call_count", 0) + 1,
        "error_count":      error_count,
    }


def _verified_reflection_update(state: AgentState, steps: list[dict[str, Any]],
                                  messages: list[dict[str, str]], tool_name: str,
                                  verification: dict[str, Any], result: dict[str, Any],
                                  last_step: dict[str, Any], error_count: int,
                                  allowed_tools: Any, reason: str,
                                  require_membership: bool = True) -> dict[str, Any] | None:
    """Shortcut a verified result with a contract reflection, when eligible.

    The low-risk shortcut fires for members of its tool set; the
    semantic-not-required shortcut fires for everyone *outside* its set.
    Returns the state update on eligibility, else ``None``.
    """
    member = (tool_name in allowed_tools) if require_membership else (tool_name not in allowed_tools)
    if not member or not verification.get("passed") or "error" in result:
        return None
    steps.append({
        "type": "reflection", "tool": tool_name,
        "result": {"passed": True, "decision": "continue", "reason": reason},
        "iteration": state["iteration"],
    })
    if isinstance(last_step.get("plan_step_index"), int):
        steps[-1]["plan_step_index"] = last_step["plan_step_index"]
    return {
        "steps":           steps,
        "messages":        messages,
        "tool_call_count": state.get("tool_call_count", 0) + 1,
        "error_count":     error_count,
        "skip_reflect":    True,
    }


@dataclass
class _ToolDispatch:
    """One pending tool call after pairing, execution and error counting."""

    tool_name: str
    params: dict[str, Any]
    result: dict[str, Any]
    last_step: dict[str, Any]
    error_count: int


def _dispatch(state: AgentState, ctx: ToolContext) -> _ToolDispatch | dict[str, Any]:
    """Cancel gate, pending-call pairing, payload validation and execution.

    Phase 1 of the tool node (review #18). Returns a complete state update
    when the run must stop before anything is recorded — cooperative cancel,
    a tool node reached without a pending call, an invalid payload, or a Qt
    acknowledgement the UI still owes us; otherwise the record handed to
    :func:`_record`.
    """
    if ctx.cancel_checker():
        steps = list(state["steps"])
        steps.append({"type": "cancelled", "content": "Task cancelled cooperatively.",
                      "iteration": state.get("iteration", 0)})
        return {"steps": steps, "done": True, "cancelled": True}

    call_steps = [s for s in state["steps"] if has_step_type(s, "tool_call")]
    result_steps = [s for s in state["steps"] if has_step_type(s, "tool_result")]
    if len(call_steps) <= len(result_steps):
        logger.error("[NODE: tool] Reached without a pending tool_call.")
        steps = list(state["steps"])
        steps.append({
            "type": "error",
            "content": "Tool node reached without a pending tool call.",
            "iteration": state.get("iteration", 0),
        })
        return {"steps": steps, "done": True}

    last_step = call_steps[len(result_steps)]

    tool_name = last_step.get("tool")
    params = last_step.get("params")
    if not tool_name or not isinstance(params, dict):
        logger.error("[NODE: tool] Pending tool_call has invalid payload: %s", last_step)
        steps = list(state["steps"])
        steps.append({
            "type": "error",
            "content": "Pending tool call has an invalid payload.",
            "iteration": state.get("iteration", 0),
        })
        return {"steps": steps, "done": True}

    logger.info("[NODE: tool] Thực thi: %s, params: %s", tool_name, str(redact_value(params))[:200])
    logger.debug(f"\n--- [LOG: TOOL NODE] Thuc thi: {tool_name} ---")
    try:
        result = ctx.execute(tool_name, params)
        logger.info("[NODE: tool] Kết quả: %s", str(result)[:500])
        logger.debug(f"[LOG: TOOL NODE] Ket qua: {result}")
    except Exception as error:
        result = {"error": f"Tool exception: {error}"}
        logger.exception("[NODE: tool] Exception khi thực thi '%s'", tool_name)
        logger.debug(f"[LOG: TOOL NODE] Loi: {result}")

    error_count = state.get("error_count", 0)
    if "error" in result:
        error_count += 1
    else:
        error_count = 0

    if result.get("pending_ui_ack"):
        return {
            "steps": state.get("steps", []),
            "pending_tool": {"tool": tool_name, "params": params, "ui_ack": True},
            "done": True,
        }
    return _ToolDispatch(tool_name, params, result, last_step, error_count)


def _record(state: AgentState, ctx: ToolContext, dispatched: _ToolDispatch) -> dict[str, Any]:
    """Write the observable outcome of an executed tool (review #18, phase 2).

    Result steps, the circuit breaker, optional verification, RAG/completion
    shortcuts, the tool-result message and the verified-reflection fast paths.
    """
    tool_name = dispatched.tool_name
    params = dispatched.params
    result = dispatched.result
    last_step = dispatched.last_step
    error_count = dispatched.error_count

    steps = list(state["steps"])
    tool_result_entry: dict[str, Any] = {
        "type": "tool_result", "tool": tool_name,
        "result": result, "iteration": state["iteration"],
    }
    if isinstance(last_step.get("plan_step_index"), int):
        tool_result_entry["plan_step_index"] = last_step["plan_step_index"]
    steps.append(tool_result_entry)

    if error_count >= 3:
        steps.append({"type": "final_answer", "content": f"Ngắt mạch (Circuit Breaker): Tool '{tool_name}' gặp lỗi 3 lần liên tiếp. Dừng tác vụ để tránh vòng lặp."})
        return {"steps": steps, "done": True, "error_count": error_count}

    verification = {"passed": "error" not in result, "reason": "No verifier configured."}
    if ctx.verify_result:
        verification = ctx.verify_result(tool_name, params, result)
        steps.append({"type": "verification", "tool": tool_name, "result": verification,
                      "iteration": state["iteration"]})

    rag_synthesis = _rag_synthesis_update(state, steps, tool_name, result, last_step, error_count)
    if rag_synthesis is not None:
        return rag_synthesis

    result_text = json.dumps(result, ensure_ascii=False, indent=2)
    if len(result_text) > 8000:
        result_text = result_text[:8000] + "\n... [truncated]"

    completion = _completion_announcement_update(state, steps, tool_name, result, error_count)
    if completion is not None:
        return completion

    messages = list(state["messages"])
    messages.append({
        "role":    "user",
        "content": (
            f"Tool `{tool_name}` tra ve:\n```json\n{result_text}\n```\n\n"
            "Use this result as evidence. If it answers the original question, "
            "return a final answer now; do not delegate or call another tool. "
            "Only continue when a specific missing fact is necessary."
        ),
    })

    low_risk = _verified_reflection_update(
        state, steps, messages, tool_name, verification, result, last_step,
        error_count, LOW_RISK_TOOLS, "Verified low-risk read-only tool result.")
    if low_risk is not None:
        return low_risk

    semantic_free = _verified_reflection_update(
        state, steps, messages, tool_name, verification, result, last_step,
        error_count, SEMANTIC_REFLECTION_TOOLS,
        "Verified result contract; semantic critic not required.",
        require_membership=False)
    if semantic_free is not None:
        return semantic_free

    return {
        "steps":           steps,
        "messages":        messages,
        "tool_call_count": state.get("tool_call_count", 0) + 1,
        "error_count":     error_count,
    }


def tool_node(state: AgentState, ctx: ToolContext) -> dict[str, Any]:
    """Execute the chosen tool and verify its result.

    Thin coordinator (review #18): :func:`_dispatch` pairs and runs the
    pending call, :func:`_record` writes everything the UI and routers then
    observe.
    """
    logger.info("[NODE: tool] Bắt đầu thực thi tool.")
    dispatched = _dispatch(state, ctx)
    if isinstance(dispatched, dict):
        return dispatched
    return _record(state, ctx, dispatched)


def after_tool(state: AgentState) -> str:
    """End after an asynchronous or approval-gated tool result."""
    if state["done"] or state["pending_tool"] is not None:
        logger.info("[ROUTER: after_tool] → END (done=%s, pending=%s)", state["done"], state["pending_tool"] is not None)
        logger.debug("--- [LOG: TOOL ROUTER] Ket thuc (cho ACK/phe duyet) ---")
        return "end"
    if state.get("synthesize_after_rag"):
        logger.info("[ROUTER: after_tool] → REASON (synthesize verified RAG evidence)")
        return "reason"
    if (state.get("skip_reflect") and state.get("steps")
            and has_step_type(state["steps"][-1], "reflection")):
        logger.info("[ROUTER: after_tool] → REASON (verified low-risk tool)")
        return "reason"
    logger.info("[ROUTER: after_tool] → REFLECT")
    logger.debug("--- [LOG: TOOL ROUTER] -> Reflect Node ---")
    return "reflect"
