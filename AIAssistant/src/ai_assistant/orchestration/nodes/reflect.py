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
from ..state import AgentState, Completion, ReflectResult, Step, has_step_type

logger = get_agent_logger("reflect")

_REFLECT_DECISIONS = {
    "continue", "revise", "retry", "replan", "ask_user", "reconcile", "stop",
}


@dataclass
class ReflectContext:
    """Dependencies for the reflect node."""
    reflect_complete: Completion
    reflect_result: ReflectResult | None


def reflect_node(state: AgentState, ctx: ReflectContext) -> dict[str, Any]:
    """Run an LLM-based review before the next ReAct turn."""
    logger.info("[NODE: reflect] Bắt đầu phản ánh kết quả tool bằng LLM.")
    calls = [step for step in state["steps"] if has_step_type(step, "tool_call")]
    results = [step for step in state["steps"] if has_step_type(step, "tool_result")]
    if not calls or not results:
        logger.info("[NODE: reflect] Không có tool call/result để phản ánh.")
        return {}
        
    call, result_step = calls[-1], results[-1]
    logger.info("[NODE: reflect] Đánh giá tool='%s'", call["tool"])

    verification = next((step.get("result", {}) for step in reversed(state["steps"])
                         if has_step_type(step, "verification") and step.get("tool") == call["tool"]),
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

    # Deterministic fast-path: application_action is executed by Qt desktop client
    # with explicit UI acknowledgement. When the action matches the plan step and verification passed,
    # do not waste 20+ seconds calling LLM critic.
    if tool_name == "application_action":
        fast_params: dict[str, Any] = call.get("params") or {}
        selected_action = (result_step.get("result") or {}).get("action") or fast_params.get("action", "")
        current_step_text = plan[completed_steps] if (plan and completed_steps < len(plan)) else ""
        match = step_matches_action(current_step_text, str(selected_action)) if current_step_text else True
        success = bool((result_step.get("result") or {}).get("success", False))
        error = (result_step.get("result") or {}).get("error", "")

        if success and not error and verification.get("passed") and match is True:
            fast_reflection: dict[str, Any] = {
                "passed": True,
                "decision": "continue",
                "reason": f"Tool 'application_action' ({selected_action}) executed successfully via desktop client.",
            }
            logger.info("[NODE: reflect] Deterministic desktop action fast-path | plan_step=%s | action=%s | passed=True",
                        current_step_text or "none", selected_action)
            return _record_reflection(state, call, fast_reflection)
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

    parsed_reflection: dict[str, Any] | None = None
    try:
        raw_obj = json.loads(raw)
        if isinstance(raw_obj, dict):
            parsed_reflection = raw_obj
    except (ValueError, json.JSONDecodeError):
        pass

    if parsed_reflection and (
        not isinstance(parsed_reflection.get("passed"), bool)
        or parsed_reflection.get("decision") not in _REFLECT_DECISIONS
        or not isinstance(parsed_reflection.get("reason"), str)
    ):
        parsed_reflection = None

    reflection: dict[str, Any]
    if not verification.get("passed"):
        reflection = ctx.reflect_result(call["tool"], call["params"], result_step["result"], verification) \
            if ctx.reflect_result else {
                "passed": False, "decision": "revise",
                "reason": str(verification.get("reason", "Verification failed.")),
            }
    elif not parsed_reflection or "passed" not in parsed_reflection:
        logger.warning("[NODE: reflect] LLM parse lỗi, dùng verification fallback.")
        if ctx.reflect_result:
            reflection = ctx.reflect_result(call["tool"], call["params"], result_step["result"], verification)
        else:
            reflection = {"passed": bool(verification.get("passed")), "decision": "continue",
                          "reason": str(verification.get("reason", "No critic configured."))}
    else:
        reflection = parsed_reflection

    if plan and completed_steps < len(plan) and call["tool"] == "application_action":
        current_step_text = plan[completed_steps]
        step_params: dict[str, Any] = call.get("params") or {}
        selected_action = (result_step.get("result") or {}).get("action") or step_params.get("action", "")
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

    reason_str = str(reflection.get("reason") or "")
    logger.info("[NODE: reflect] Kết quả phản ánh | plan_step=%s | passed=%s | reason=%s",
                plan[completed_steps] if plan and completed_steps < len(plan) else "none",
                reflection.get("passed"), reason_str[:200])

    return _record_reflection(state, call, reflection)


def _record_reflection(state: AgentState, call: Step,
                       reflection: dict[str, Any]) -> dict[str, Any]:
    """Persist critic output and feed a failed review back to the reasoner."""
    steps = list(state["steps"])
    decision = str(reflection.get("decision", "continue"))
    workflow_decision = "retry" if decision == "revise" else decision
    reflection = {**reflection, "workflow_decision": workflow_decision}
    reflection_step: Step = {"type": "reflection", "tool": call["tool"], "result": reflection,
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
            
    update: dict[str, Any] = {"steps": steps, "messages": messages}
    if workflow_decision == "replan":
        replan_count = state.get("replan_count", 0) + 1
        update.update({
            "replan_requested": True,
            "replan_count": replan_count,
            "plan_verified": False,
            "plan_feedback": str(reflection.get("reason", "Execution evidence invalidated the plan.")),
        })
        if replan_count > state.get("max_replans", 2):
            steps.append({
                "type": "final_answer",
                "content": "Đã đạt giới hạn lập lại kế hoạch; dừng để tránh thực thi sai.",
            })
            update.update({"steps": steps, "done": True, "replan_requested": False})
    elif workflow_decision in {"ask_user", "reconcile", "stop"}:
        steps.append({
            "type": "final_answer",
            "content": str(reflection.get("reason", "Không thể tiếp tục an toàn.")),
        })
        update.update({"steps": steps, "done": True})
    return update


def after_reflect(state: AgentState) -> str:
    """End after reflection or loop back to reason node."""
    if state["done"] or state["pending_tool"] is not None:
        logger.info("[ROUTER: after_reflect] → END")
        return "end"
    if state.get("replan_requested"):
        logger.info("[ROUTER: after_reflect] → PLAN (runtime evidence invalidated plan)")
        return "plan"
    logger.info("[ROUTER: after_reflect] → REASON")
    return "reason"
