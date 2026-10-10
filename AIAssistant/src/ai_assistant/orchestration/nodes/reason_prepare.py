"""Context assembly for the reason node.

``reason_node`` is the ReAct dispatch loop: it decides what to ask the LLM,
routes validation/approval outcomes and produces the next state. Building the
prompt context (plan progress, execution contract, UI hints, directives and
forced envelopes) was ~200 lines of pure message preparation inside that loop;
this module owns it so the loop reads as a decision tree and the assembly can
be unit-tested without a fake ``ReasonContext``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ai_assistant.config.logging import get_agent_logger
from ai_assistant.orchestration.specialists.code import coding_workflow_guidance, coding_workflow_status
from ai_assistant.tools.action_manifest import rank_actions_for_step

from ..helpers import (
    has_rag_evidence_for_step,
    has_rag_evidence_without_plan,
    plan_step_execution_contract,
)
from ..prompts import REASONER_PROMPT
from ..state import OBSERVATION_SUMMARY_THRESHOLD, AgentState
from .summarize import summarize_messages

logger = get_agent_logger("reasoning")

#: Absolute ceiling for the assembled prompt; anything above is compacted
#: right before the LLM call (a second safety net after the step-1 compact).
_AGENT_HARD_LIMIT = 14000


@dataclass
class PreparedReasoning:
    """Everything the reason node needs from the assembled LLM context."""

    messages: list[dict[str, str]]
    #: A decided tool envelope (execution contract) to run instead of asking
    #: the LLM, or None when the model must decide.
    forced_envelope: dict[str, Any] | None
    step_contract: dict[str, Any]
    current_plan_step: str | None
    coding_status: Any
    #: When set, the reason node must return this state update immediately:
    #: every plan step was answered inline and the answers are aggregated.
    early_final: dict[str, Any] | None = None


def prepare_reasoning(
    state: AgentState,
    steps: list[dict[str, Any]],
    plan: list[str] | None,
    iteration: int,
    tool_call_count: int,
    done_count: int,
) -> PreparedReasoning:
    """Assemble the system context, execution contract and directives."""
    messages = list(state["messages"])

    # ── Step 1: compact BASE messages before adding any overhead ──────────
    base_chars = sum(len(m.get("content", "")) for m in messages)
    if (tool_call_count >= OBSERVATION_SUMMARY_THRESHOLD
            or (tool_call_count > 0 and base_chars > 11000)):
        messages = summarize_messages(messages)
        logger.info("[NODE: reason] Context compacted before REASONER_PROMPT: chars=%d tool_calls=%d",
                    base_chars, tool_call_count)
        logger.debug(f"[AGENT TRACE] ── Reason: tóm tắt messages base (tool_call_count={tool_call_count}).")

    # ── Step 2: prepend REASONER_PROMPT to the (already-compacted) system ─
    if messages and messages[0].get("role") == "system":
        original_sys = messages[0]["content"]
        messages[0] = {
            "role": "system",
            "content": REASONER_PROMPT + original_sys
        }

    coding_status = None
    if state.get("enforce_coding_workflow"):
        user_task = next(
            (message.get("content", "") for message in state.get("messages", [])
             if message.get("role") == "user"),
            "",
        )
        coding_status = coding_workflow_status(user_task, steps)
        if coding_status.missing:
            messages = [*messages, {
                "role": "system",
                "content": (
                    f"[Coding workflow status] kind={coding_status.kind}; "
                    f"missing={', '.join(coding_status.missing)}. "
                    f"{coding_workflow_guidance(coding_status)}"
                ),
            }]

    # ── Step 3: append plan progress + action hint ────────────────────────
    if plan:
        remaining = plan[done_count:] if done_count < len(plan) else []
        if remaining:
            logger.info("[NODE: reason] Plan progress: completed=%d/%d; current step=%s; remaining=%s",
                        done_count, len(plan), remaining[0], remaining)
            logger.debug(f"[AGENT TRACE] ── Reason: plan step {done_count + 1}/{len(plan)} → {remaining[0]}")
            messages = [*messages, {
                "role": "system",
                "content": f"[Ke hoach con lai] {json.dumps(remaining, ensure_ascii=False)}",
            }]
        else:
            logger.info("[NODE: reason] Plan progress: completed=%d/%d; all steps finished",
                        done_count, len(plan))
            logger.debug(f"[AGENT TRACE] ── Reason: plan completed {done_count}/{len(plan)}")
            messages = [*messages, {
                "role": "system",
                "content": "[Kế hoạch đã hoàn tất] Mọi bước trong kế hoạch đã được thực hiện xong. Hãy trả về final_answer để kết thúc, KHÔNG gọi thêm tool.",
            }]

    required_actions = state.get("required_ui_actions", [])
    completed_ui_actions = sum(
        1 for step in steps
        if (step.get("type") == "reflection" and step.get("tool") == "application_action"
            and step.get("result", {}).get("passed") is True)
    )
    expected_action = (required_actions[completed_ui_actions]
                       if completed_ui_actions < len(required_actions) else None)
    current_plan_step = (plan[done_count]
                         if plan and done_count < len(plan) else None)

    user_request = next((m.get("content", "") for m in state.get("messages", [])
                         if m.get("role") == "user"), "")

    from ai_assistant.tools.action_manifest import normalize_text

    plan_spec = state.get("plan_spec") or {}
    step_kinds = plan_spec.get("step_kinds") or [] if isinstance(plan_spec, dict) else []

    def _kind_for(step_text: str | None, index: int | None) -> str | None:
        if isinstance(step_kinds, list) and step_kinds:
            if index is not None and 0 <= index < len(step_kinds):
                return step_kinds[index]
            if index is None and len(step_kinds) == 1:
                return step_kinds[0]
        return None

    request_contract = plan_step_execution_contract(
        user_request, rank_actions_for_step, normalize_text,
        kind=_kind_for(None, None) if not plan else None,
    )
    step_contract = (plan_step_execution_contract(
                        current_plan_step, rank_actions_for_step, normalize_text,
                        kind=_kind_for(current_plan_step, done_count))
                     if current_plan_step
                     else (request_contract if request_contract["mode"] in {"rag_search", "direct_answer", "application_action"}
                           else {"mode": "model_selected"}))

    plan_total = len(plan or [])
    progress_total = len(required_actions) if required_actions else plan_total
    logger.info("[NODE: reason] selection context | plan_step=%s | execution=%s | expected_action=%s | completed=%d/%d",
                current_plan_step or "none", step_contract["mode"],
                step_contract.get("action", "none"),
                done_count if plan else completed_ui_actions, progress_total)

    if expected_action:
        logger.info("[NODE: reason] ToolApp hint | plan step %d/%d | expected action=%s | failed attempts=%d",
                    done_count + 1 if plan else completed_ui_actions + 1, progress_total,
                    expected_action.get("action"),
                    sum(1 for step in steps
                        if step.get("type") == "reflection"
                        and step.get("result", {}).get("passed") is False))
        logger.debug("[AGENT TRACE] ── Reason: ToolApp hint; gọi LLM tool-calling")
        messages = [*messages, {
            "role": "system",
            "content": (
                "[UI plan hint] Một workflow đã nhận diện action canonical "
                f"'{expected_action.get('action')}'. Đây chỉ là gợi ý; hãy tự đối chiếu "
                "với bước hiện tại và không lặp lại tool/action đã bị Reflect đánh giá thất bại."
            ),
        }]

    # ── Early final: plan finished via inline step answers ───────────────
    # Nếu mọi bước kế hoạch đã hoàn tất và các bước đó được trả lời bằng
    # step_answer, tổng hợp nội dung đó thành final_answer thay vì gọi LLM
    # lần nữa (LLM thường chỉ trả "Đã hoàn tất kế hoạch." và làm mất câu trả lời).
    if plan and done_count >= len(plan):
        step_answers = [
            step.get("content", "").strip()
            for step in steps
            if step.get("type") == "step_answer" and step.get("content", "").strip()
        ]
        if step_answers:
            logger.info(
                "[NODE: reason] All plan steps finished; aggregating %d step_answer(s) as final_answer.",
                len(step_answers),
            )
            final_content = "\n\n".join(step_answers)
            steps.append({"type": "final_answer", "content": final_content,
                          "iteration": iteration})
            updated_messages = list(state["messages"])
            updated_messages.append({"role": "assistant", "content": final_content})
            return PreparedReasoning(
                messages=messages,
                forced_envelope=None,
                step_contract=step_contract,
                current_plan_step=current_plan_step,
                coding_status=coding_status,
                early_final={
                    "iteration": iteration,
                    "steps": steps,
                    "messages": updated_messages,
                    "tool_call_count": tool_call_count,
                    "synthesize_after_rag": False,
                    "done": True,
                },
            )

    # ── Step 4: append last-in-context directive ──────────────────────────
    if current_plan_step:
        directive = f'[Buoc hien tai] "{current_plan_step}".'
        if step_contract["mode"] == "direct_answer":
            directive += " Đây là kiến thức chung: trả lời bằng step_answer, không gọi tool."
        elif step_contract["mode"] == "rag_search":
            directive += " Đây là thông nội bộ: phải dùng rag_search trước khi trả lời."
        elif step_contract["mode"] == "application_action":
            directive += f" Phải gọi application_action với action '{step_contract['action']}'."
        if state.get("enforce_coding_workflow") or state.get("required_ui_actions"):
            directive += " Chi thuc hien dung buoc nay; khong thuc hien cac buoc sau."
        else:
            directive += " Nếu đã thu thập đủ thông tin cho TẤT CẢ các bước, bạn có thể trả lời tất cả cùng lúc bằng một final_answer."

        candidates = rank_actions_for_step(current_plan_step)
        if candidates:
            best = [name for name, score in candidates if score == candidates[0][1]]
            directive += (f" Action canonical khop nhat theo manifest: {', '.join(best)}. "
                          "Day chi la goi y; chi chon action khac khi no ro rang phu hop hon.")
        messages = [*messages, {"role": "system", "content": directive}]

    # ── Step 5: final hard-limit guard right before calling LLM ──────────
    pre_call_chars = sum(len(m.get("content", "")) for m in messages)
    if pre_call_chars > _AGENT_HARD_LIMIT:
        messages = summarize_messages(messages)
        logger.info("[NODE: reason] Pre-call safety compact: chars=%d -> compacted", pre_call_chars)
        logger.debug(f"[AGENT TRACE] ── Reason: pre-call safety compact ({pre_call_chars} chars > {_AGENT_HARD_LIMIT})")

    forced_envelope: dict[str, Any] | None = None
    if step_contract["mode"] == "rag_search":
        has_evidence = (
            has_rag_evidence_for_step(steps, done_count)
            if current_plan_step else has_rag_evidence_without_plan(steps)
        )
        if not has_evidence:
            forced_envelope = {
                "kind": "tool", "tool": "rag_search",
                "params": {
                    "query": step_contract["query"],
                    **({"top_k": step_contract["top_k"]}
                       if step_contract.get("top_k") else {}),
                },
            }
    elif current_plan_step and step_contract["mode"] == "application_action":
        forced_envelope = {
            "kind": "tool", "tool": "application_action",
            "params": {"action": step_contract["action"]},
        }

    return PreparedReasoning(
        messages=messages,
        forced_envelope=forced_envelope,
        step_contract=step_contract,
        current_plan_step=current_plan_step,
        coding_status=coding_status,
    )
