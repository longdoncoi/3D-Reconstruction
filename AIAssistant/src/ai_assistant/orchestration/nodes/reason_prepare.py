"""Context assembly for the reason node.

``reason_node`` is the ReAct dispatch loop: it decides what to ask the LLM,
routes validation/approval outcomes and produces the next state. Building the
prompt context (plan progress, execution contract, UI hints, directives and
forced envelopes) was ~200 lines of pure message preparation inside that loop;
this module owns it so the loop reads as a decision tree and the assembly can
be unit-tested without a fake ``ReasonContext``.

The assembly itself is a fixed pipeline (ADR 0014): each stage is one private
function that takes the facts it needs and returns either the messages to append
or the value the next stage consumes. ``prepare_reasoning`` reads as the
sequence of those stages, so adding a directive is one small function rather
than another branch in a 200-line body.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from ai_assistant.config.logging import get_agent_logger
from ai_assistant.orchestration.specialists.code import coding_workflow_guidance, coding_workflow_status
from ai_assistant.tools.action_manifest import normalize_text, rank_actions_for_step

from ..helpers import (
    has_rag_evidence_for_step,
    has_rag_evidence_without_plan,
    plan_step_execution_contract,
)
from ..prompts import REASONER_PROMPT
from ..state import OBSERVATION_SUMMARY_THRESHOLD, AgentState, Step, has_step_type
from .summarize import summarize_messages

logger = get_agent_logger("reasoning")

#: Absolute ceiling for the assembled prompt; anything above is compacted
#: right before the LLM call (a second safety net after the step-1 compact).
_AGENT_HARD_LIMIT = 14000

#: Base-history size (characters) that triggers a compact once at least one
#: tool call has happened, even below ``OBSERVATION_SUMMARY_THRESHOLD`` calls.
_BASE_COMPACT_CHARS = 11000


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


@dataclass(frozen=True, slots=True)
class _Selection:
    """Plan/UI facts that several assembly stages share.

    Computed once so the contract stage, the hint stage and the logging stage
    cannot disagree about which step or which action is current.
    """

    current_plan_step: str | None
    expected_action: dict[str, Any] | None
    completed_ui_actions: int
    progress_total: int
    user_request: str


# ── Stage helpers (ADR 0014) ─────────────────────────────────────────────────


def _compact_base_messages(messages: list[dict[str, str]], tool_call_count: int) -> list[dict[str, str]]:
    """Stage 1 — summarise the base history before adding any prompt overhead."""
    base_chars = sum(len(m.get("content", "")) for m in messages)
    if not (tool_call_count >= OBSERVATION_SUMMARY_THRESHOLD
            or (tool_call_count > 0 and base_chars > _BASE_COMPACT_CHARS)):
        return messages
    logger.info("[NODE: reason] Context compacted before REASONER_PROMPT: chars=%d tool_calls=%d",
                base_chars, tool_call_count)
    logger.debug(f"[AGENT TRACE] ── Reason: tóm tắt messages base (tool_call_count={tool_call_count}).")
    return summarize_messages(messages)


def _with_reasoner_prompt(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Stage 2 — make ``REASONER_PROMPT`` the head of the (compacted) system turn."""
    if messages and messages[0].get("role") == "system":
        messages[0] = {"role": "system", "content": REASONER_PROMPT + messages[0]["content"]}
    return messages


def _coding_workflow_message(state: AgentState, steps: list[Step]) -> tuple[Any, dict[str, Any] | None]:
    """The coding-workflow status banner, or ``None`` when off or already complete.

    Returns ``(coding_status, message)``: the status is needed downstream even
    when there is nothing left to warn about.
    """
    if not state.get("enforce_coding_workflow"):
        return None, None
    user_task = next(
        (message.get("content", "") for message in state.get("messages", [])
         if message.get("role") == "user"),
        "",
    )
    coding_status = coding_workflow_status(user_task, steps)
    if not coding_status.missing:
        return coding_status, None
    return coding_status, {
        "role": "system",
        "content": (
            f"[Coding workflow status] kind={coding_status.kind}; "
            f"missing={', '.join(coding_status.missing)}. "
            f"{coding_workflow_guidance(coding_status)}"
        ),
    }


def _plan_progress_messages(plan: list[str] | None, done_count: int) -> list[dict[str, str]]:
    """Stage 3 — report what of the plan is done and what remains."""
    if not plan:
        return []
    remaining = plan[done_count:] if done_count < len(plan) else []
    if remaining:
        logger.info("[NODE: reason] Plan progress: completed=%d/%d; current step=%s; remaining=%s",
                    done_count, len(plan), remaining[0], remaining)
        logger.debug(f"[AGENT TRACE] ── Reason: plan step {done_count + 1}/{len(plan)} → {remaining[0]}")
        return [{
            "role": "system",
            "content": f"[Ke hoach con lai] {json.dumps(remaining, ensure_ascii=False)}",
        }]
    logger.info("[NODE: reason] Plan progress: completed=%d/%d; all steps finished", done_count, len(plan))
    logger.debug(f"[AGENT TRACE] ── Reason: plan completed {done_count}/{len(plan)}")
    return [{
        "role": "system",
        "content": "[Kế hoạch đã hoàn tất] Mọi bước trong kế hoạch đã được thực hiện xong. "
                   "Hãy trả về final_answer để kết thúc, KHÔNG gọi thêm tool.",
    }]


def _completed_ui_actions(steps: list[Step]) -> int:
    """How many required desktop actions a reflection has already confirmed."""
    return sum(
        1 for step in steps
        if (has_step_type(step, "reflection") and step.get("tool") == "application_action"
            and (step.get("result") or {}).get("passed") is True)
    )


def _user_request(state: AgentState) -> str:
    return next((m.get("content", "") for m in state.get("messages", [])
                 if m.get("role") == "user"), "")


def _selection_context(state: AgentState, steps: list[Step],
                       plan: list[str] | None, done_count: int) -> _Selection:
    """Resolve the current step, the next expected UI action and the request."""
    required_actions = state.get("required_ui_actions", [])
    completed = _completed_ui_actions(steps)
    expected_action = required_actions[completed] if completed < len(required_actions) else None
    current_plan_step = plan[done_count] if plan and done_count < len(plan) else None
    plan_total = len(plan or [])
    return _Selection(
        current_plan_step=current_plan_step,
        expected_action=expected_action,
        completed_ui_actions=completed,
        progress_total=len(required_actions) if required_actions else plan_total,
        user_request=_user_request(state),
    )


def _step_kind_for(step_kinds: Any, step_text: str | None, index: int | None) -> str | None:
    """Pick the declared kind for a step, when the plan spec recorded one."""
    if isinstance(step_kinds, list) and step_kinds:
        if index is not None and 0 <= index < len(step_kinds):
            return cast("str | None", step_kinds[index])
        if index is None and len(step_kinds) == 1:
            return cast("str | None", step_kinds[0])
    return None


def _select_step_contract(state: AgentState, user_request: str, plan: list[str] | None,
                          current_plan_step: str | None, done_count: int) -> dict[str, Any]:
    """Decide the execution contract for the current step, else for the request."""
    plan_spec = state.get("plan_spec") or {}
    step_kinds = plan_spec.get("step_kinds") or [] if isinstance(plan_spec, dict) else []
    request_contract = plan_step_execution_contract(
        user_request, rank_actions_for_step, normalize_text,
        kind=_step_kind_for(step_kinds, None, None) if not plan else None,
    )
    if current_plan_step:
        return plan_step_execution_contract(
            current_plan_step, rank_actions_for_step, normalize_text,
            kind=_step_kind_for(step_kinds, current_plan_step, done_count),
        )
    return (request_contract
            if request_contract["mode"] in {"rag_search", "direct_answer", "application_action"}
            else {"mode": "model_selected"})


def _log_selection(selection: _Selection, step_contract: dict[str, Any],
                   plan: list[str] | None, done_count: int, steps: list[Step]) -> None:
    """Emit the selection-context and ToolApp-hint diagnostics."""
    done_display = done_count if plan else selection.completed_ui_actions
    logger.info("[NODE: reason] selection context | plan_step=%s | execution=%s | expected_action=%s | completed=%d/%d",
                selection.current_plan_step or "none", step_contract["mode"],
                step_contract.get("action", "none"), done_display, selection.progress_total)
    if not selection.expected_action:
        return
    step = done_display + 1
    failed = sum(1 for s in steps
                 if has_step_type(s, "reflection") and (s.get("result") or {}).get("passed") is False)
    logger.info("[NODE: reason] ToolApp hint | plan step %d/%d | expected action=%s | failed attempts=%d",
                step, selection.progress_total, selection.expected_action.get("action"), failed)
    logger.debug("[AGENT TRACE] ── Reason: ToolApp hint; gọi LLM tool-calling")


def _ui_hint_message(selection: _Selection) -> dict[str, str] | None:
    """The "this action is expected" hint, which the model may still override."""
    if not selection.expected_action:
        return None
    return {
        "role": "system",
        "content": (
            "[UI plan hint] Một workflow đã nhận diện action canonical "
            f"'{selection.expected_action.get('action')}'. Đây chỉ là gợi ý; hãy tự đối chiếu "
            "với bước hiện tại và không lặp lại tool/action đã bị Reflect đánh giá thất bại."
        ),
    }


def _aggregate_step_answers(state: AgentState, steps: list[Step], messages: list[dict[str, str]],
                            selection: _Selection, step_contract: dict[str, Any],
                            coding_status: Any, plan: list[str] | None, done_count: int,
                            iteration: int, tool_call_count: int) -> PreparedReasoning | None:
    """Early final: every plan step was answered inline; aggregate, do not re-ask.

    Asking the LLM again here usually yields only "Đã hoàn tất kế hoạch." and
    loses the answers, so the inline content becomes the final answer.
    """
    if not (plan and done_count >= len(plan)):
        return None
    step_answers = [
        step.get("content", "").strip()
        for step in steps
        if has_step_type(step, "step_answer") and step.get("content", "").strip()
    ]
    if not step_answers:
        return None
    logger.info("[NODE: reason] All plan steps finished; aggregating %d step_answer(s) as final_answer.",
                len(step_answers))
    final_content = "\n\n".join(step_answers)
    steps.append({"type": "final_answer", "content": final_content, "iteration": iteration})
    updated_messages = [*state["messages"], {"role": "assistant", "content": final_content}]
    return PreparedReasoning(
        messages=messages,
        forced_envelope=None,
        step_contract=step_contract,
        current_plan_step=selection.current_plan_step,
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


def _directive_message(current_plan_step: str | None, step_contract: dict[str, Any],
                       state: AgentState, steps: list[Step], done_count: int) -> dict[str, str] | None:
    """Stage 4 — the last-in-context directive naming the current step."""
    if not current_plan_step:
        return None
    directive = f'[Buoc hien tai] "{current_plan_step}".'
    if step_contract["mode"] == "direct_answer":
        directive += " Đây là kiến thức chung: trả lời bằng step_answer, không gọi tool."
    elif step_contract["mode"] == "rag_search":
        # Evidence already fetched for this exact step (a parallel batch can
        # have fetched the neighbours in the same fan-out, ADR 0044): asking
        # for another search would both duplicate the work and contradict the
        # synthesis request the tool node just appended.
        if has_rag_evidence_for_step(steps, done_count):
            directive += (" Bằng chứng cho bước này đã có trong hội thoại:"
                          " trả lời bằng step_answer, KHÔNG gọi lại rag_search.")
        else:
            directive += " Đây là thông nội bộ: phải dùng rag_search trước khi trả lời."
    elif step_contract["mode"] == "application_action":
        directive += f" Phải gọi application_action với action '{step_contract['action']}'."
    if state.get("enforce_coding_workflow") or state.get("required_ui_actions"):
        directive += " Chi thuc hien dung buoc nay; khong thuc hien cac buoc sau."
    else:
        directive += (" Nếu đã thu thập đủ thông tin cho TẤT CẢ các bước, bạn có thể trả lời "
                      "tất cả cùng lúc bằng một final_answer.")

    candidates = rank_actions_for_step(current_plan_step)
    if candidates:
        best = [name for name, score in candidates if score == candidates[0][1]]
        directive += (f" Action canonical khop nhat theo manifest: {', '.join(best)}. "
                      "Day chi la goi y; chi chon action khac khi no ro rang phu hop hon.")
    return {"role": "system", "content": directive}


def _compact_before_call(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Stage 5 — the final hard-limit guard right before calling the LLM."""
    pre_call_chars = sum(len(m.get("content", "")) for m in messages)
    if pre_call_chars <= _AGENT_HARD_LIMIT:
        return messages
    logger.info("[NODE: reason] Pre-call safety compact: chars=%d -> compacted", pre_call_chars)
    logger.debug(f"[AGENT TRACE] ── Reason: pre-call safety compact ({pre_call_chars} chars > {_AGENT_HARD_LIMIT})")
    return summarize_messages(messages)


def _forced_envelope(step_contract: dict[str, Any], current_plan_step: str | None,
                     steps: list[Step], done_count: int) -> dict[str, Any] | None:
    """The execution contract decides the next tool before the model is asked."""
    if step_contract["mode"] == "rag_search":
        has_evidence = (has_rag_evidence_for_step(steps, done_count)
                        if current_plan_step else has_rag_evidence_without_plan(steps))
        if has_evidence:
            return None
        return {
            "kind": "tool", "tool": "rag_search",
            "params": {
                "query": step_contract["query"],
                **({"top_k": step_contract["top_k"]} if step_contract.get("top_k") else {}),
            },
        }
    if current_plan_step and step_contract["mode"] == "application_action":
        return {"kind": "tool", "tool": "application_action",
                "params": {"action": step_contract["action"]}}
    return None


def prepare_reasoning(
    state: AgentState,
    steps: list[Step],
    plan: list[str] | None,
    iteration: int,
    tool_call_count: int,
    done_count: int,
) -> PreparedReasoning:
    """Assemble the system context, execution contract and directives."""
    messages = _with_reasoner_prompt(_compact_base_messages(list(state["messages"]), tool_call_count))

    coding_status, coding_message = _coding_workflow_message(state, steps)
    if coding_message is not None:
        messages = [*messages, coding_message]

    messages = [*messages, *_plan_progress_messages(plan, done_count)]

    selection = _selection_context(state, steps, plan, done_count)
    step_contract = _select_step_contract(
        state, selection.user_request, plan, selection.current_plan_step, done_count,
    )
    _log_selection(selection, step_contract, plan, done_count, steps)

    hint = _ui_hint_message(selection)
    if hint is not None:
        messages = [*messages, hint]

    early = _aggregate_step_answers(
        state, steps, messages, selection, step_contract, coding_status,
        plan, done_count, iteration, tool_call_count,
    )
    if early is not None:
        return early

    directive = _directive_message(
        selection.current_plan_step, step_contract, state, steps, done_count,
    )
    if directive is not None:
        messages = [*messages, directive]

    return PreparedReasoning(
        messages=_compact_before_call(messages),
        forced_envelope=_forced_envelope(step_contract, selection.current_plan_step, steps, done_count),
        step_contract=step_contract,
        current_plan_step=selection.current_plan_step,
        coding_status=coding_status,
    )
