"""Tool execution node for the LangGraph orchestration.

This node executes the tool call selected by the Reasoner and updates
the state with the result, including observation summarization for RAG.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ai_assistant.config.logging import get_agent_logger

from ..state import (
    COMPLETION_TOOLS,
    LOW_RISK_TOOLS,
    SEMANTIC_REFLECTION_TOOLS,
    AgentState,
    Executor,
    VerifyResult,
)
from .summarize import summarize_messages

logger = get_agent_logger("tool")


@dataclass
class ToolContext:
    """Dependencies for the tool node."""
    execute: Executor
    verify_result: VerifyResult | None
    cancel_checker: Callable[[], bool]


def tool_node(state: AgentState, ctx: ToolContext) -> dict[str, Any]:
    """Execute the chosen tool and verify its result."""
    logger.info("[NODE: tool] Bắt đầu thực thi tool.")
    if ctx.cancel_checker():
        steps = list(state["steps"])
        steps.append({"type": "cancelled", "content": "Task cancelled cooperatively.",
                      "iteration": state.get("iteration", 0)})
        return {"steps": steps, "done": True, "cancelled": True}
        
    call_steps = [s for s in state["steps"] if s.get("type") == "tool_call"]
    result_steps = [s for s in state["steps"] if s.get("type") == "tool_result"]
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

    is_project_research = (tool_name == "rag_search")

    logger.info("[NODE: tool] Thực thi: %s, params: %s", tool_name, str(params)[:200])
    logger.debug(f"\n--- [LOG: TOOL NODE] Thuc thi: {tool_name} ---")
    try:
        result = ctx.execute(tool_name, params)
        logger.info("[NODE: tool] Kết quả: %s", str(result)[:500])
        logger.debug(f"[LOG: TOOL NODE] Ket qua: {result}")
    except Exception as error:  # noqa: BLE001
        result = {"error": f"Tool exception: {error}"}
        logger.error("[NODE: tool] Exception khi thực thi '%s': %s", tool_name, error)
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

    if is_project_research:
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

    result_text = json.dumps(result, ensure_ascii=False, indent=2)
    if len(result_text) > 8000:
        result_text = result_text[:8000] + "\n... [truncated]"

    if tool_name in COMPLETION_TOOLS and result.get("success"):
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
    
    if tool_name in LOW_RISK_TOOLS and verification.get("passed") and "error" not in result:
        steps.append({
            "type": "reflection", "tool": tool_name,
            "result": {"passed": True, "decision": "continue",
                        "reason": "Verified low-risk read-only tool result."},
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
        
    if (tool_name not in SEMANTIC_REFLECTION_TOOLS
            and verification.get("passed") and "error" not in result):
        steps.append({
            "type": "reflection", "tool": tool_name,
            "result": {"passed": True, "decision": "continue",
                        "reason": "Verified result contract; semantic critic not required."},
            "iteration": state["iteration"],
        })
        if isinstance(last_step.get("plan_step_index"), int):
            steps[-1]["plan_step_index"] = last_step["plan_step_index"]
        return {"steps": steps, "messages": messages,
                "tool_call_count": state.get("tool_call_count", 0) + 1,
                "error_count": error_count, "skip_reflect": True}
                
    return {
        "steps":           steps,
        "messages":        messages,
        "tool_call_count": state.get("tool_call_count", 0) + 1,
        "error_count":     error_count,
    }


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
            and state["steps"][-1].get("type") == "reflection"):
        logger.info("[ROUTER: after_tool] → REASON (verified low-risk tool)")
        return "reason"
    logger.info("[ROUTER: after_tool] → REFLECT")
    logger.debug("--- [LOG: TOOL ROUTER] -> Reflect Node ---")
    return "reflect"
