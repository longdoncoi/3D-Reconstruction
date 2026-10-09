"""Reflect node for the LangGraph orchestration.

This node uses the LLM to review the outcome of the most recent tool
execution before proceeding to the next reasoning turn.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ai_assistant.config.logging import get_agent_logger
from ai_assistant.tools.action_manifest import canonical_action, step_matches_action

from ..helpers import completed_plan_steps
from ..prompts import CRITIC_PROMPT
from ..state import AgentState, Completion, ReflectResult

logger = get_agent_logger("reflect")


@dataclass
class ReflectContext:
    """Dependencies for the reflect node."""
    reflect_complete: Completion
    reflect_result: ReflectResult | None


def reflect_node(state: AgentState, ctx: ReflectContext) -> dict[str, Any]:
    """Run an LLM-based review before the next ReAct turn."""
    logger.info("[NODE: reflect] Bắt đầu phản ánh kết quả tool bằng LLM.")
    calls = [step for step in state["steps"] if step.get("type") == "tool_call"]
    results = [step for step in state["steps"] if step.get("type") == "tool_result"]
    if not calls or not results:
        logger.info("[NODE: reflect] Không có tool call/result để phản ánh.")
        return {}
        
    call, result_step = calls[-1], results[-1]
    logger.info("[NODE: reflect] Đánh giá tool='%s'", call["tool"])

    verification = next((step.get("result", {}) for step in reversed(state["steps"])
                         if step.get("type") == "verification" and step.get("tool") == call["tool"]),
                        {"passed": "error" not in result_step.get("result", {})})

    tool_name = call["tool"]
    tool_params = json.dumps(call["params"], ensure_ascii=False)
    tool_result = json.dumps(result_step["result"], ensure_ascii=False)
    if len(tool_result) > 2000:
        tool_result = tool_result[:2000] + "... [truncated]"

    user_msg = next((m["content"] for m in state.get("messages", []) if m.get("role") == "user"), "")
    plan = state.get("plan")
    plan_text = f"Kế hoạch hiện tại: {plan}\n" if plan else ""

    scope_note = ""
    completed_steps = completed_plan_steps(state["steps"], plan)
    if plan and completed_steps < len(plan):
        current_step_text = plan[completed_steps]
        scope_note = (
            f"LƯU Ý QUAN TRỌNG: Đây là MỘT bước trong một kế hoạch nhiều bước. "
            f"Bước đang cần đánh giá là: \"{current_step_text}\". "
            f"CHỈ đánh giá xem tool call này có hoàn thành ĐÚNG bước hiện tại hay không. "
            f"KHÔNG đánh giá là thất bại chỉ vì các bước KHÁC trong kế hoạch chưa được thực hiện — "
            f"những bước đó sẽ được xử lý ở các lượt tiếp theo.\n"
        )

    critic_msgs = [
        {"role": "system", "content": CRITIC_PROMPT},
        {
            "role": "user",
            "content": f"Yêu cầu ban đầu của người dùng: {user_msg}\n{plan_text}{scope_note}Tool đã gọi: {tool_name}\nTham số: {tool_params}\nKết quả: {tool_result}\n\nDựa vào yêu cầu và kết quả này, tool đã gọi có hoàn thành đúng mục tiêu của bước hiện tại không?"
        }
    ]

    logger.debug("[AGENT TRACE] >> Reflect node: LLM evaluating tool result...")
    raw = ctx.reflect_complete(critic_msgs, max(0.1, state["temperature"] - 0.1)).strip()
    logger.debug(f"[AGENT TRACE] ── Reflect node LLM output: {raw[:150]}")

    reflection = None
    try:
        reflection = json.loads(raw)
        if not isinstance(reflection, dict):
            reflection = None
    except (ValueError, json.JSONDecodeError):
        pass

    if reflection and (
        not isinstance(reflection.get("passed"), bool)
        or reflection.get("decision") not in {"continue", "revise"}
        or not isinstance(reflection.get("reason"), str)
    ):
        reflection = None

    if not verification.get("passed"):
        reflection = ctx.reflect_result(call["tool"], call["params"], result_step["result"], verification) \
            if ctx.reflect_result else {
                "passed": False, "decision": "revise",
                "reason": verification.get("reason", "Verification failed."),
            }
    elif not reflection or "passed" not in reflection:
        logger.warning("[NODE: reflect] LLM parse lỗi, dùng verification fallback.")
        if ctx.reflect_result:
            reflection = ctx.reflect_result(call["tool"], call["params"], result_step["result"], verification)
        else:
            reflection = {"passed": bool(verification.get("passed")), "decision": "continue",
                          "reason": verification.get("reason", "No critic configured.")}

    if plan and completed_steps < len(plan) and call["tool"] == "application_action":
        current_step_text = plan[completed_steps]
        selected_action = result_step.get("result", {}).get("action") or call.get("params", {}).get("action", "")
        canonical_selected = canonical_action(str(selected_action))
        match = step_matches_action(current_step_text, str(selected_action))
        logger.info("[NODE: reflect] Semantic check | plan_step=%s | action=%s | canonical=%s | match=%s",
                    current_step_text, selected_action, canonical_selected or "none", match)
        if match is False:
            reflection = {
                "passed": False,
                "decision": "revise",
                "reason": (
                    f"Action '{selected_action}' does not match current plan step "
                    f"'{current_step_text}'."
                ),
            }

    logger.info("[NODE: reflect] Kết quả phản ánh | plan_step=%s | passed=%s | reason=%s",
                plan[completed_steps] if plan and completed_steps < len(plan) else "none",
                reflection.get("passed"), reflection.get("reason", "")[:200])

    return _record_reflection(state, call, reflection)


def _record_reflection(state: AgentState, call: dict[str, Any],
                       reflection: dict[str, Any]) -> dict[str, Any]:
    """Persist critic output and feed a failed review back to the reasoner."""
    steps = list(state["steps"])
    reflection_step = {"type": "reflection", "tool": call["tool"], "result": reflection,
                       "iteration": state["iteration"]}
    if isinstance(call.get("plan_step_index"), int):
        reflection_step["plan_step_index"] = call["plan_step_index"]
    steps.append(reflection_step)
    messages = list(state["messages"])
    
    if not reflection.get("passed"):
        logger.warning("[NODE: reflect] Review FAILED cho tool '%s': %s",
                       call["tool"], reflection.get("reason", ""))
        recovery = (
            " If this was a discovery call with no evidence, change the query or scope and use a complementary "
            "discovery tool (file listing, symbol analysis, focused read, or diff) instead of repeating the "
            "same fingerprint. Preserve the failure as evidence and continue the plan."
        )
        messages.append({"role": "system", "content": (
            "[Independent review failed] " + str(reflection.get("reason", "Revise the approach.")) +
            " Do not repeat the same failing call; inspect evidence, follow the current plan step, "
            "and choose a different valid tool or parameters when needed." + recovery)})
            
    return {"steps": steps, "messages": messages, "last_reflection": reflection}


def after_reflect(state: AgentState) -> str:
    """End after reflection or loop back to reason node."""
    if state["done"] or state["pending_tool"] is not None:
        logger.info("[ROUTER: after_reflect] → END")
        return "end"
    logger.info("[ROUTER: after_reflect] → REASON")
    return "reason"
