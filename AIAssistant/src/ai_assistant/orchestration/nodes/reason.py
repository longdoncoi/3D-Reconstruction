"""Reason node for the LangGraph orchestration.

This node runs the core ReAct loop: it evaluates the current state and plan,
queries the LLM for the next tool call or a final answer, and guards the
dispatch (validation errors, desktop-action mismatches, duplicate calls and
the approval gate).

Context assembly (system prompt, plan progress, execution contract,
directives, forced envelopes) lives in :mod:`reason_prepare` so this module
stays a decision tree. The decision tree itself was split further (ADR 0013):
each outcome the model can produce — an inline step answer, a final answer, a
schema validation failure, a forbidden transfer, a step/action mismatch, a
duplicate failed call, or a dispatch — is handled by one focused private
function that receives the shared :class:`_Turn` record and returns either the
partial state update or ``None`` to fall through to the next guard.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ai_assistant.config.logging import get_agent_logger
from ai_assistant.domain.approvals import (
    approval_covers_invocation,
    build_approval_preview,
    derive_approval_scope,
)
from ai_assistant.llm.inference import strip_think_tags
from ai_assistant.observability import record_step_guard
from ai_assistant.orchestration.specialists.code import coding_workflow_guidance
from ai_assistant.tools.action_manifest import rank_actions_for_step, step_matches_action

from ..helpers import (
    clean_project_research_answer,
    completed_plan_steps,
    has_rag_evidence_for_step,
)
from ..state import (
    MAX_STEP_MISMATCH_REJECTIONS,
    AgentState,
    ApprovalCovers,
    Completion,
    NeedsApproval,
    Parser,
    SelectSpecialist,
    is_context_exhausted,
)
from .reason_prepare import prepare_reasoning
from .summarize import format_code_citation, summarize_messages

logger = get_agent_logger("reasoning")

#: Parameters that change on every retry but must not affect the duplicate
#: failed-call fingerprint (ADR 0013 — lifted to module scope, was a nested def).
VOLATILE_PARAM_KEYS = frozenset({"request_id", "workflow_id"})


@dataclass
class ReasonContext:
    """Dependencies for the reason node."""
    complete: Completion
    parse: Parser
    needs_approval: NeedsApproval
    max_iterations: int
    cancel_checker: Callable[[], bool]
    select_specialist: SelectSpecialist | None
    #: Live probe for the single-use approval grant. When the engine has a
    #: gateway wired (production), an exhausted or mismatched grant sends the
    #: node back to the user instead of letting the loop retry a tool the
    #: gateway will refuse.
    approval_covers: ApprovalCovers | None = None


@dataclass
class _Turn:
    """Everything the per-outcome handlers need, computed once per iteration.

    Bundling these into one record keeps each handler's signature small and
    makes it obvious which facts are shared versus which are produced by a
    single guard.
    """
    state: AgentState
    ctx: ReasonContext
    steps: list[dict[str, Any]]
    iteration: int
    plan: list[str] | None
    tool_call_count: int
    done_count: int
    step_contract: dict[str, Any]
    current_plan_step: str | None
    coding_status: Any
    answer: str
    tool_name: str | None
    tool_params: dict[str, Any] | None

    def continue_update(self, **extra: Any) -> dict[str, Any]:
        """The common "keep looping" state update most guards return."""
        return {
            "iteration": self.iteration,
            "steps": self.steps,
            "tool_call_count": self.tool_call_count,
            **extra,
        }


def _user_request(state: AgentState) -> str:
    return next(
        (m.get("content", "") for m in state.get("messages", []) if m.get("role") == "user"),
        "",
    )


def _messages_with(turn: _Turn, *appended: dict[str, str]) -> list[dict[str, str]]:
    """The current message history plus *appended* turns."""
    return [*turn.state["messages"], *appended]


def approval_covers_run(state: AgentState, tool_name: str, params: dict[str, Any],
                        ctx: ReasonContext) -> bool:
    """True only when *this exact* invocation is still covered by an approval.

    ``approval_granted`` alone is no longer sufficient: the grant is bound to
    the fingerprint of the tool call the user saw in the preview, so a resumed
    run can not spend its approval on a different tool or different parameters.

    The fingerprint policy itself is domain-owned
    (:func:`~ai_assistant.domain.approvals.approval_covers_invocation`); this
    orchestration-side wrapper only reads the run state and consults the
    optional live grant probe.
    """
    if not state.get("approval_granted"):
        return False
    if not approval_covers_invocation(state.get("granted_fingerprint"), tool_name, params):
        return False
    if ctx.approval_covers is not None and not ctx.approval_covers(tool_name, params):
        return False
    return True


# ── Per-outcome handlers (ADR 0013) ─────────────────────────────────────────
# Each returns the partial state update for its outcome, or ``None`` when the
# turn does not match that outcome and the next guard should run.


def _compact_context(turn: _Turn) -> dict[str, Any]:
    """The model signalled the context budget is exhausted; summarise and retry."""
    compacted = summarize_messages(turn.state["messages"])
    turn.steps.append({"type": "context_compacted", "iteration": turn.iteration})
    updated = [*compacted, {"role": "system", "content":
                            "Context vừa được rút gọn. Tiếp tục bằng một JSON tool_call hoặc final khi đủ bằng chứng."}]
    return turn.continue_update(messages=updated, done=False)


def _handle_step_answer(turn: _Turn) -> dict[str, Any] | None:
    """Inline response for a general-knowledge plan step (or a hallucinated one)."""
    if turn.tool_name != "_step_answer":
        return None
    step_content = (turn.tool_params or {}).get("content", turn.answer)

    if turn.current_plan_step is None:
        logger.info("[NODE: reason] Treating step_answer without an active plan as final_answer.")
        turn.steps.append({"type": "final_answer", "content": step_content,
                           "iteration": turn.iteration})
        return turn.continue_update(
            messages=_messages_with(turn, {"role": "assistant", "content": turn.answer}),
            synthesize_after_rag=False,
            done=True,
        )

    # HARD CONSTRAINT for hallucination prevention: a step whose contract
    # demands project evidence cannot be answered from general knowledge.
    if (turn.step_contract["mode"] == "rag_search"
            and not has_rag_evidence_for_step(turn.steps, turn.done_count)):
        logger.warning(f"[NODE: reason] Blocking hallucinated step_answer for step: {turn.current_plan_step}")
        turn.steps.append({"type": "validation_error",
                           "content": f"Step '{turn.current_plan_step}' requires project-specific context. Use rag_search.",
                           "iteration": turn.iteration})
        return turn.continue_update(
            messages=_messages_with(
                turn,
                {"role": "assistant", "content": turn.answer},
                {"role": "user", "content":
                 f"LỖI: Bước '{turn.current_plan_step}' CẦN thông tin nội bộ. BẠN BẮT BUỘC PHẢI GỌI TOOL `rag_search` (với tham số query phù hợp) để tìm kiếm, KHÔNG ĐƯỢC tự bịa đặt bằng step_answer."},
            ),
            error_count=turn.state.get("error_count", 0) + 1,
        )

    logger.debug(f"[AGENT TRACE] ── Reason: step_answer for plan step {turn.done_count}.")
    logger.info("[NODE: reason] LLM trả lời trực tiếp cho bước kế hoạch (step_answer).")
    turn.steps.append({"type": "step_answer", "content": step_content, "iteration": turn.iteration})
    reflection_entry: dict[str, Any] = {
        "type": "reflection", "tool": "step_answer",
        "result": {"passed": True, "decision": "continue",
                   "reason": "General-knowledge step answered inline without tool."},
        "iteration": turn.iteration,
    }
    if turn.current_plan_step is not None:
        reflection_entry["plan_step_index"] = turn.done_count
    turn.steps.append(reflection_entry)
    return turn.continue_update(
        messages=_messages_with(turn, {"role": "assistant", "content": turn.answer}),
        synthesize_after_rag=False,
        done=False,
    )


def _handle_final_answer(turn: _Turn) -> dict[str, Any] | None:
    """The model wants to stop; block it when the plan/coding evidence is incomplete."""
    if turn.tool_name is not None:
        return None
    logger.debug("[AGENT TRACE] ── Reason: final answer.")
    logger.info("[NODE: reason] LLM quyết định dừng (final_answer).")
    coding_missing = turn.coding_status.missing if turn.coding_status else ()

    plan_missing = bool(
        turn.state.get("enforce_plan_completion")
        and not turn.state.get("enforce_coding_workflow")
        and turn.plan and turn.done_count < len(turn.plan)
    )
    if coding_missing or plan_missing:
        remaining = (turn.plan or [])[turn.done_count:] if plan_missing else []
        logger.warning(
            "[NODE: reason] Ignoring premature final answer; missing coding=%s remaining plan=%s",
            coding_missing, remaining,
        )
        turn.steps.append({
            "type": "coding_incomplete" if coding_missing else "plan_incomplete",
            "remaining": remaining,
            "missing": list(coding_missing),
            "iteration": turn.iteration,
        })
        coding_instruction = (
            f" {coding_workflow_guidance(turn.coding_status)}"
            if turn.coding_status and coding_missing else ""
        )
        prefix = "Coding task" if coding_missing else "Kế hoạch"
        return turn.continue_update(
            messages=_messages_with(
                turn,
                {"role": "assistant", "content": turn.answer},
                {"role": "user", "content": (
                    f"{prefix} chưa đủ bằng chứng để kết thúc. Không được kết thúc hoặc chỉ giải thích. "
                    f"Còn các bước kế hoạch: {json.dumps(remaining, ensure_ascii=False)}."
                    f"{coding_instruction} "
                    "Hãy tiếp tục bằng đúng một JSON tool_call phù hợp."
                )},
            ),
            done=False,
        )

    answer = turn.answer
    citation = format_code_citation(turn.state.get("messages", []), turn.steps)
    if citation:
        logger.info("[NODE: reason] Dùng formatter trích dẫn code đầy đủ từ read_file result.")
        answer = citation
    elif any(step.get("type") == "tool_call" and step.get("tool") == "rag_search"
             for step in turn.steps):
        answer = clean_project_research_answer(answer)
    turn.steps.append({"type": "final_answer", "content": answer})
    return turn.continue_update(done=True)


def _handle_validation_error(turn: _Turn) -> dict[str, Any] | None:
    """A parsed envelope failed schema validation; bounce it back for a retry."""
    if turn.tool_name != "_validation_error":
        return None
    error_detail = (turn.tool_params or {}).get("error", "Unknown validation error")
    original_tool = (turn.tool_params or {}).get("tool", "unknown")
    logger.warning(
        "[NODE: reason] Validation error cho tool '%s': %s",
        original_tool, error_detail,
    )
    logger.debug(
        f"[AGENT TRACE] ── Reason: VALIDATION ERROR cho '{original_tool}': {error_detail}",
    )
    turn.steps.append({
        "type": "validation_error", "tool": original_tool,
        "error": error_detail, "iteration": turn.iteration,
    })
    return turn.continue_update(
        messages=_messages_with(
            turn,
            {"role": "assistant", "content": turn.answer},
            {"role": "user", "content": (
                f"LỖI: Bạn vừa gọi tool '{original_tool}' nhưng tham số KHÔNG HỢP LỆ.\n"
                f"Chi tiết: {error_detail}\n\n"
                "Hãy đọc danh sách tool/action hợp lệ, đối chiếu với bước hiện tại "
                "và gọi lại tool phù hợp. KHÔNG ĐƯỢC trả lời bằng văn bản — "
                "chỉ trả về JSON tool_call."
            )},
        ),
    )


def _handle_forbidden_transfer(turn: _Turn) -> dict[str, Any] | None:
    """The model called the transfer tool directly instead of ``application_action``."""
    if turn.tool_name != "transfer_to_toolapp_agent":
        return None
    logger.warning("[NODE: reason] LLM called transfer_to_toolapp_agent, bouncing back to force application_action.")
    logger.debug("[AGENT TRACE] ── Reason: VALIDATION ERROR: called transfer_to_toolapp_agent")
    turn.steps.append({
        "type": "validation_error", "tool": turn.tool_name,
        "error": "Forbidden direct call to transfer_to_toolapp_agent.", "iteration": turn.iteration,
    })
    return turn.continue_update(
        messages=_messages_with(
            turn,
            {"role": "assistant", "content": turn.answer},
            {"role": "user", "content": (
                "LỖI: KHÔNG ĐƯỢC gọi transfer_to_toolapp_agent cho thao tác UI. "
                "Hãy gọi TRỰC TIẾP tool 'application_action' với action canonical phù hợp (ví dụ: viewer.load_2d)."
            )},
        ),
    )


def _guard_step_action_mismatch(turn: _Turn) -> dict[str, Any] | None:
    """Right tool, wrong desktop action for the current plan step.

    Returns an update when the call must be rejected, ``None`` when it may be
    dispatched (either the action matches, or the guard is exhausted and the
    Reflect node takes over).
    """
    if turn.tool_name != "application_action" or not turn.current_plan_step:
        return None
    selected_action = str((turn.tool_params or {}).get("action", ""))
    if step_matches_action(turn.current_plan_step, selected_action) is not False:
        return None

    expected = [name for name, _ in rank_actions_for_step(turn.current_plan_step)][:3]
    rejections = sum(
        1 for step in turn.steps
        if step.get("type") == "validation_error"
        and step.get("reason") == "step_action_mismatch"
        and step.get("plan_step_index") == turn.done_count
    )
    logger.warning("[NODE: reason] Step/action mismatch | step=%s | selected=%s | expected=%s | rejections=%d",
                   turn.current_plan_step, selected_action, expected, rejections)
    if rejections >= MAX_STEP_MISMATCH_REJECTIONS:
        record_step_guard("fallthrough")
        logger.error("[NODE: reason] Step guard exhausted; dispatching and relying on Reflect.")
        return None

    record_step_guard("rejected")
    turn.steps.append({"type": "validation_error", "reason": "step_action_mismatch",
                       "tool": turn.tool_name, "plan_step_index": turn.done_count,
                       "error": f"'{selected_action}' does not match step '{turn.current_plan_step}'",
                       "iteration": turn.iteration})
    return turn.continue_update(
        messages=_messages_with(
            turn,
            {"role": "assistant", "content": turn.answer},
            {"role": "user", "content": (
                f"LỖI: Action '{selected_action}' KHÔNG khớp bước hiện tại \"{turn.current_plan_step}\". "
                f"Action khớp theo manifest: {', '.join(expected)}. "
                "Hãy gọi lại application_action với action đúng. Chỉ trả về JSON tool_call.")},
        ),
    )


def stable_fingerprint(tool: str, params: dict[str, Any] | None) -> str:
    """Canonical tool+params identity that ignores volatile per-retry ids."""
    stable = {k: v for k, v in (params or {}).items() if k not in VOLATILE_PARAM_KEYS}
    return json.dumps({"tool": tool, "params": stable}, ensure_ascii=False, sort_keys=True)


def _failed_call_fingerprints(steps: list[dict[str, Any]]) -> set[str]:
    """Fingerprints of every tool call that a later reflection rejected."""
    failed: set[str] = set()
    for index, step in enumerate(steps):
        if step.get("type") != "reflection" or step.get("result", {}).get("passed") is not False:
            continue
        for prior in reversed(steps[:index]):
            if prior.get("type") == "tool_call":
                failed.add(json.dumps({"tool": prior.get("tool"), "params": prior.get("params", {})},
                                      ensure_ascii=False, sort_keys=True))
                break
    return failed


def _guard_duplicate_failed_call(turn: _Turn) -> dict[str, Any] | None:
    """Reject a tool call identical to one that already failed review."""
    tool_name = turn.tool_name
    if tool_name is None:  # pragma: no cover — _handle_final_answer catches None first
        return None
    fingerprint = stable_fingerprint(tool_name, turn.tool_params or {})
    if fingerprint not in _failed_call_fingerprints(turn.steps):
        return None
    turn.steps.append({"type": "validation_error", "tool": tool_name,
                       "error": "Identical tool call already failed review; choose a different query, scope, or tool.",
                       "iteration": turn.iteration})
    return turn.continue_update(
        messages=_messages_with(
            turn,
            {"role": "user", "content":
             "Không được lặp lại tool call vừa thất bại. Hãy đổi tham số/phạm vi hoặc chọn tool khám phá bổ sung."},
        ),
    )


def _record_thinking(turn: _Turn) -> None:
    """Surface the text the model wrote before the tool-call fence as a step."""
    clean_answer = strip_think_tags(turn.answer)
    think_text = clean_answer
    if "```tool_call" in clean_answer:
        think_text = clean_answer.split("```tool_call")[0].strip()
    elif "{" in clean_answer:
        think_text = clean_answer.split("{")[0].strip()
    if think_text:
        turn.steps.append({"type": "thinking", "content": think_text, "iteration": turn.iteration})


def _dispatch_tool_call(turn: _Turn) -> dict[str, Any]:
    """Delegation, tool_call step, and the approval gate."""
    tool_name = turn.tool_name
    if tool_name is None:  # pragma: no cover — _handle_final_answer catches None first
        return turn.continue_update()
    params = turn.tool_params or {}
    idempotency_key = ""
    if turn.ctx.select_specialist:
        delegation = turn.ctx.select_specialist(tool_name, params)
        idempotency_key = delegation.get("idempotency_key", "")
        turn.steps.append({"type": "delegation", "agent": delegation.get("specialist", "supervisor"),
                           "tool": tool_name, "idempotency_key": idempotency_key,
                           "remote_endpoint": delegation.get("remote_endpoint"),
                           "iteration": turn.iteration})

    logger.debug(f"[AGENT TRACE] ── Reason: tool call → tool='{tool_name}', params={str(params)[:80]}")
    logger.info("[NODE: reason] LLM quyết định gọi tool: %s, params: %s", tool_name, params)
    tool_call = {"type": "tool_call", "tool": tool_name,
                 "params": params, "idempotency_key": idempotency_key,
                 "iteration": turn.iteration}
    if turn.current_plan_step is not None:
        tool_call["plan_step_index"] = turn.done_count
    turn.steps.append(tool_call)

    updated_messages = _messages_with(turn, {"role": "assistant", "content": turn.answer})
    approval_scope = derive_approval_scope(
        turn.plan, turn.state.get("plan_spec"), turn.state.get("approval_scope"),
    )

    if turn.ctx.needs_approval(tool_name) and not approval_covers_run(
            turn.state, tool_name, params, turn.ctx):
        preview = build_approval_preview(
            scope_id=approval_scope,
            spec=turn.state.get("plan_spec"),
            tool_name=tool_name,
            parameters=params,
            remaining_steps=turn.plan[turn.done_count:]
            if turn.plan and turn.done_count < len(turn.plan) else [],
        )
        return {
            "iteration":    turn.iteration,
            "steps":        turn.steps,
            "messages":     updated_messages,
            "pending_tool": {"tool": tool_name, "params": params,
                             "approval_scope": approval_scope,
                             "approval_preview": preview},
            "approval_scope": approval_scope,
            "done":         True,
        }

    return {
        "iteration":       turn.iteration,
        "steps":           turn.steps,
        "messages":        updated_messages,
        "tool_call_count": turn.tool_call_count,
    }


def reason_node(state: AgentState, ctx: ReasonContext) -> dict[str, Any]:
    """Evaluate context and plan, then select the next tool to execute."""
    iteration = state["iteration"] + 1
    steps = list(state["steps"])
    logger.debug(f"\n[AGENT TRACE] ── Reason node: iter {iteration}")
    logger.info(f"[NODE: reason] Bắt đầu suy luận (iteration {iteration})")

    if ctx.cancel_checker():
        steps.append({"type": "cancelled", "content": "Task cancelled cooperatively.",
                      "iteration": iteration})
        return {"iteration": iteration, "steps": steps, "done": True, "cancelled": True}

    if iteration > ctx.max_iterations:
        logger.debug(f"[AGENT TRACE] ── Reason: đạt giới hạn ({ctx.max_iterations} iterations).")
        steps.append({"type": "final_answer",
                      "content": "Agent da dat gioi han so vong lap."})
        return {"iteration": iteration, "steps": steps, "done": True}

    plan = state.get("plan")
    tool_call_count = state.get("tool_call_count", 0)
    done_count = completed_plan_steps(steps, plan)

    prepared = prepare_reasoning(state, steps, plan, iteration, tool_call_count, done_count)
    if prepared.early_final is not None:
        return prepared.early_final

    if prepared.forced_envelope is not None:
        answer = json.dumps(prepared.forced_envelope, ensure_ascii=False)
        logger.info("[NODE: reason] Enforcing execution contract for step/request '%s': %s",
                    prepared.current_plan_step or _user_request(state), prepared.forced_envelope)
    else:
        logger.debug("[AGENT TRACE] ── Reason: đang gọi LLM...")
        answer = ctx.complete(prepared.messages, state["temperature"]).strip()

    logger.debug(f"[AGENT TRACE] ── Reason: LLM output ({len(answer)} chars): {answer[:120].replace(chr(10), ' ')}")
    # The raw LLM envelope is already logged once by the constrained
    # completion adapter ("Constrained LLM response").  Logging it again here
    # duplicated the full payload in the server log/Qt console.

    turn = _Turn(
        state=state, ctx=ctx, steps=steps, iteration=iteration, plan=plan,
        tool_call_count=tool_call_count, done_count=done_count,
        step_contract=prepared.step_contract,
        current_plan_step=prepared.current_plan_step,
        coding_status=prepared.coding_status,
        answer=answer, tool_name=None, tool_params=None,
    )

    if is_context_exhausted(answer):
        return _compact_context(turn)

    if not answer:
        steps.append({"type": "error", "content": "LLM tra ve rong."})
        return {"iteration": iteration, "steps": steps, "done": True}

    tool_name, tool_params = ctx.parse(answer)
    if (tool_name is None and prepared.current_plan_step
            and prepared.step_contract["mode"] == "direct_answer" and answer):
        tool_name, tool_params = "_step_answer", {"content": answer}
    logger.info("[NODE: reason] selected tool=%s action=%s (model output)",
                tool_name or "final_answer", (tool_params or {}).get("action", ""))
    turn.tool_name, turn.tool_params = tool_name, tool_params

    # ── Decision tree: first matching outcome wins ─────────────────────────
    for outcome in (
        _handle_step_answer,
        _handle_final_answer,
        _handle_validation_error,
        _handle_forbidden_transfer,
        _guard_step_action_mismatch,
        _guard_duplicate_failed_call,
    ):
        update = outcome(turn)
        if update is not None:
            return update

    _record_thinking(turn)
    return _dispatch_tool_call(turn)


def after_reason(state: AgentState) -> str:
    """Determine next step after the reason node."""
    if state["done"] or state["pending_tool"] is not None:
        logger.info("[ROUTER: after_reason] → END (done=%s, pending=%s)", state["done"], state["pending_tool"] is not None)
        logger.debug("[AGENT TRACE] ── Router: kết thúc (hoặc chờ phê duyệt)")
        return "end"

    steps = state.get("steps", [])
    if steps and steps[-1].get("type") == "validation_error":
        logger.info("[ROUTER: after_reason] → REASON (validation error, self-correct)")
        logger.debug("[AGENT TRACE] ── Router: → Reason node (validation error self-correct)")
        return "reason"
    if steps and steps[-1].get("type") == "plan_incomplete":
        logger.info("[ROUTER: after_reason] → REASON (premature final answer blocked)")
        logger.debug("[AGENT TRACE] ── Router: → Reason node (plan incomplete)")
        return "reason"
    if steps and steps[-1].get("type") == "coding_incomplete":
        logger.info("[ROUTER: after_reason] → REASON (coding evidence incomplete)")
        logger.debug("[AGENT TRACE] ── Router: → Reason node (coding workflow incomplete)")
        return "reason"
    if steps and steps[-1].get("type") == "context_compacted":
        logger.info("[ROUTER: after_reason] → REASON (context compacted)")
        logger.debug("[AGENT TRACE] ── Router: → Reason node (context compacted)")
        return "reason"

    if (len(steps) >= 2
            and steps[-2].get("type") == "step_answer"
            and steps[-1].get("type") == "reflection"):
        logger.info("[ROUTER: after_reason] → REASON (step_answer completed)")
        logger.debug("[AGENT TRACE] ── Router: → Reason node (step_answer)")
        return "reason"

    logger.info("[ROUTER: after_reason] → TOOL")
    logger.debug("[AGENT TRACE] ── Router: → Tool node")
    return "tool"
