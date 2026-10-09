"""Plan Reflect node for the LangGraph orchestration.

This node acts as a critic to evaluate a newly generated plan before it is
executed. It uses a dedicated ``plan_reflect_complete`` callback.
"""
from __future__ import annotations

import json
from typing import Any

from modules.agent_logging import get_agent_logger
from modules.inference import strip_think_tags

from ..prompts import PLAN_CRITIC_PROMPT
from ..state import AgentState, Completion

logger = get_agent_logger("reasoning")


def plan_reflect_node(state: AgentState, plan_reflect_complete: Completion | None) -> dict[str, Any]:
    """Review a newly generated plan before entering the tool loop."""
    plan = state.get("plan") or []
    user_msg = next((m.get("content", "") for m in state.get("messages", [])
                     if m.get("role") == "user"), "")
    logger.info("[NODE: plan_reflect] Đánh giá kế hoạch lần %d | plan=%s",
                state.get("plan_attempts", 0), plan)
    if not plan:
        return {"plan_verified": True}

    review = None
    # Manifest matching is advisory only. Natural-language plan steps can
    # legitimately describe an action without repeating its canonical
    # phrase. Log the hints for observability.
    plan_hints = []
    logger.info("[NODE: plan_reflect] Action hints | hints=%s", plan_hints)

    if review is None and plan_reflect_complete is not None:
        spec_to_check = dict(state.get('plan_spec') or {'steps': plan})
        spec_to_check.pop("tool_hints", None)
        critic_msgs = [
            {"role": "system", "content": PLAN_CRITIC_PROMPT},
            {"role": "user", "content": (
                f"Yêu cầu ban đầu: {user_msg}\n"
                f"Kế hoạch cần kiểm tra: {json.dumps(spec_to_check, ensure_ascii=False)}\n"
                "Kế hoạch có đạt yêu cầu và sẵn sàng thực thi không?"
            )},
        ]
        print("[AGENT TRACE] >> Plan Reflect node: LLM evaluating plan...", flush=True)
        raw = plan_reflect_complete(critic_msgs, max(0.1, state["temperature"] - 0.1)).strip()
        raw = strip_think_tags(raw)
        print(f"[AGENT TRACE] ── Plan Reflect output: {raw[:150]}", flush=True)
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                review = payload
        except (ValueError, json.JSONDecodeError):
            review = None
        if (not review or not isinstance(review.get("passed"), bool)
                or review.get("decision") not in {"continue", "revise"}
                or not isinstance(review.get("reason"), str)):
            review = {"passed": False, "decision": "revise",
                      "reason": "Plan Reflect không trả về JSON hợp lệ; cần Planner sinh lại kế hoạch."}

    if review is None:
        review = {"passed": True, "decision": "continue",
                  "reason": "Plan passed deterministic checks; no plan critic configured."}
    passed = bool(review.get("passed"))
    steps = list(state.get("steps", []))
    steps.append({"type": "plan_reflection", "result": review,
                  "iteration": state.get("iteration", 0),
                  "plan_attempt": state.get("plan_attempts", 0)})
    logger.info("[NODE: plan_reflect] Kết quả phản ánh | plan_step_count=%d | passed=%s | reason=%s",
                len(plan), passed, review.get("reason", "")[:240])
    if passed:
        return {"steps": steps, "plan_verified": True,
                "plan_feedback": "", "last_reflection": review}
    if state.get("plan_attempts", 0) >= 3:
        steps.append({
            "type": "final_answer",
            "content": "Không thể tạo kế hoạch hợp lệ sau 3 lần kiểm tra; chưa gọi tool để tránh thực hiện sai yêu cầu.",
        })
    return {"steps": steps, "plan_verified": False,
            "plan_feedback": str(review.get("reason", "Kế hoạch chưa đạt yêu cầu.")),
            "last_reflection": review,
            "done": state.get("plan_attempts", 0) >= 3}


def after_plan_reflect(state: AgentState) -> str:
    """Determine the next node based on the plan verification result."""
    if state.get("plan_verified"):
        logger.info("[ROUTER: after_plan_reflect] → REASON (plan passed)")
        return "reason"
    attempts = state.get("plan_attempts", 0)
    if attempts >= 3:
        logger.warning("[ROUTER: after_plan_reflect] Plan failed after %d attempts; stop safely", attempts)
        return "end"
    logger.info("[ROUTER: after_plan_reflect] → PLAN (regenerate; attempt=%d)", attempts + 1)
    return "plan"
