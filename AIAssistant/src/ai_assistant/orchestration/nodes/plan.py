"""Plan node for the LangGraph orchestration.

This node is responsible for generating a step-by-step plan before the agent starts executing tools.
It delegates the actual LLM call to a provided ``plan_complete`` callback.
"""
from __future__ import annotations

import json
from typing import Any

from ai_assistant.config.logging import get_agent_logger
from ai_assistant.llm.inference import strip_think_tags
from ai_assistant.tools.action_manifest import normalize_text, rank_actions_for_step

from ..helpers import normalise_plan_payload, plan_step_execution_contract
from ..prompts import PLANNER_PROMPT
from ..state import AgentState, Completion

logger = get_agent_logger("reasoning")


def plan_node(state: AgentState, plan_complete: Completion) -> dict[str, Any]:
    """Sinh ke hoach cac buoc. Skip neu cau ngan (likely UI action)."""
    logger.info("[NODE: plan] Bắt đầu sinh kế hoạch.")
    logger.debug(f"[NODE: plan] State hiện tại (iteration: {state.get('iteration')}): messages={len(state.get('messages', []))} steps={len(state.get('steps', []))}")
    messages = state["messages"]

    previous_review = next(
        (step.get("result", {}) for step in reversed(state.get("steps", []))
         if step.get("type") == "plan_reflection"), None)
    
    if previous_review and previous_review.get("passed") is True:
        for step in reversed(state.get("steps", [])):
            if step.get("type") == "plan":
                logger.info("[NODE: plan] Đã có kế hoạch đã xác thực: %s", step.get("steps"))
                return {"plan": step.get("steps"), "plan_spec": step.get("spec"), "plan_verified": True}

    # Preserve an existing plan while entering the graph from a fresh
    # request; only a failed plan_reflection authorises regeneration.
    if previous_review is None:
        for step in reversed(state.get("steps", [])):
            if step.get("type") == "plan":
                logger.info("[NODE: plan] Đã có kế hoạch từ trước: %s", step.get("steps"))
                return {"plan": step.get("steps"), "plan_spec": step.get("spec")}

    # Skip planning nếu đang ở giữa task (đã có tool_call từ vòng lặp trước)
    if any(step.get("type") == "tool_call" for step in state.get("steps", [])) and previous_review is None:
        return {"plan": None}

    # Tìm tin nhắn thực sự của người dùng (tin nhắn 'user' đầu tiên)
    # để tránh nhầm lẫn với các tool result được đóng giả thành 'user' ở cuối.
    user_msg = ""
    for m in messages:
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    planning_msgs = [
        {"role": "system", "content": PLANNER_PROMPT},
        {
            "role": "user",
            "content": (
                f"Yêu cầu hiện tại: {user_msg}\n"
                f"Phản hồi kiểm tra kế hoạch trước (nếu có): {state.get('plan_feedback', '')}\n"
                "Nếu đây là yêu cầu giao diện, chỉ đưa vào plan các kết quả giao diện "
                "mà người dùng yêu cầu; không tự thêm bước chuẩn bị không cần thiết."
                + (
                    "\nKế hoạch trước bị từ chối. BẮT BUỘC phải sửa theo phản hồi trên: "
                    "tách các bước quá chung chung thành nhiều bước đơn lẻ, mỗi bước một hành động, "
                    "không lặp lại nguyên văn kế hoạch trước đó."
                    if state.get("plan_feedback") else ""
                )
            ),
        },
    ]
    
    logger.debug("[AGENT TRACE] >> Plan node: generating plan...")
    raw = plan_complete(planning_msgs, max(0.1, state["temperature"] - 0.1)).strip()
    raw = strip_think_tags(raw)
    logger.debug(f"[AGENT TRACE] ── Plan: kế hoạch thô: {raw[:200]}")

    plan: list[str] | None = None
    plan_spec: dict[str, Any] = {}
    parsed = False
    try:
        payload = json.loads(raw)
        parsed = isinstance(payload, dict)
        if parsed:
            plan, plan_spec = normalise_plan_payload(payload)
            if plan:
                plan_spec["steps"] = plan
    except (ValueError, json.JSONDecodeError):
        plan = None

    if plan:
        tool_hints = []
        kinds = plan_spec.get("step_kinds") or []
        for idx, step_text in enumerate(plan):
            contract = plan_step_execution_contract(
                step_text,
                rank_actions_for_step,
                normalize_text,
                kind=kinds[idx] if idx < len(kinds) else None,
            )
            tool_hints.append({
                "step": step_text,
                "tool": contract["mode"],
                "action": contract.get("action"),
            })
        plan_spec["tool_hints"] = tool_hints
        
        steps = list(state["steps"])
        steps.append({"type": "plan", "steps": plan, "spec": plan_spec, "tool_hints": tool_hints})
        logger.debug(f"[AGENT TRACE] ── Plan: {plan}")
        logger.info(f"[NODE: plan] Đã sinh kế hoạch: {plan}")
        return {"plan": plan, "steps": steps,
                "plan_verified": False,
                "plan_attempts": state.get("plan_attempts", 0) + 1,
                "plan_spec": plan_spec,
                "plan_feedback": ""}

    if not parsed:
        logger.debug("[AGENT TRACE] ── Plan: không parse được, tiếp tục không có plan.")
        logger.info("[NODE: plan] Không sinh được kế hoạch.")
    elif plan_spec.get("requires_plan"):
        logger.debug("[AGENT TRACE] ── Plan: planner yêu cầu kế hoạch nhưng không có bước hợp lệ.")
        logger.info("[NODE: plan] Planner không trả về bước kế hoạch hợp lệ.")
    else:
        logger.debug("[AGENT TRACE] ── Plan: planner xác nhận không cần kế hoạch.")
        logger.info("[NODE: plan] Planner xác nhận tác vụ một bước.")
    # Một yêu cầu duy nhất (requires_plan=false) không có "plan" nhưng vẫn cần
    # "plan_spec.step_kinds" để Reason biết dùng rag_search/direct/tool.
    return {"plan": None, "plan_spec": plan_spec}
