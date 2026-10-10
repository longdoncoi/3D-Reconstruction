"""Reason node for the LangGraph orchestration.

This node runs the core ReAct loop: it evaluates the current state and plan,
queries the LLM for the next tool call or a final answer, and guards the
dispatch (validation errors, desktop-action mismatches, duplicate calls and
the approval gate).

Context assembly (system prompt, plan progress, execution contract,
directives, forced envelopes) lives in :mod:`reason_prepare` so this module
stays a decision tree.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Callable

from ai_assistant.config.logging import get_agent_logger
from ai_assistant.domain.approvals import approval_fingerprint
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
)
from .reason_prepare import prepare_reasoning
from .summarize import format_code_citation, summarize_messages

logger = get_agent_logger("reasoning")


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


def _user_request(state: AgentState) -> str:
    return next(
        (m.get("content", "") for m in state.get("messages", []) if m.get("role") == "user"),
        "",
    )


def approval_covers_run(state: AgentState, tool_name: str, params: dict[str, Any],
                        ctx: ReasonContext) -> bool:
    """True only when *this exact* invocation is still covered by an approval.

    ``approval_granted`` alone is no longer sufficient: the grant is bound to
    the fingerprint of the tool call the user saw in the preview, so a resumed
    run can not spend its approval on a different tool or different parameters.
    """
    if not state.get("approval_granted"):
        return False
    granted = state.get("granted_fingerprint") or ""
    if not hmac.compare_digest(granted, approval_fingerprint(tool_name, params)):
        # Empty ``granted_fingerprint`` (graph constructed outside the runner,
        # e.g. contract tests) falls back to the run-level grant, which the
        # gateway still refuses without a live capability.
        if granted:
            return False
    if ctx.approval_covers is not None and not ctx.approval_covers(tool_name, params):
        return False
    return True


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

    messages = prepared.messages
    step_contract = prepared.step_contract
    current_plan_step = prepared.current_plan_step
    coding_status = prepared.coding_status

    if prepared.forced_envelope is not None:
        answer = json.dumps(prepared.forced_envelope, ensure_ascii=False)
        logger.info("[NODE: reason] Enforcing execution contract for step/request '%s': %s",
                    current_plan_step or _user_request(state), prepared.forced_envelope)
    else:
        logger.debug("[AGENT TRACE] ── Reason: đang gọi LLM...")
        answer = ctx.complete(messages, state["temperature"]).strip()

    logger.debug(f"[AGENT TRACE] ── Reason: LLM output ({len(answer)} chars): {answer[:120].replace(chr(10), ' ')}")
    # The raw LLM envelope is already logged once by the constrained
    # completion adapter ("Constrained LLM response").  Logging it again here
    # duplicated the full payload in the server log/Qt console.

    if answer.casefold().startswith(("context quá dài", "context qua dai", "context too long")):
        compacted = summarize_messages(state["messages"])
        steps.append({"type": "context_compacted", "iteration": iteration})
        updated = [*compacted, {"role": "system", "content":
                                "Context vừa được rút gọn. Tiếp tục bằng một JSON tool_call hoặc final khi đủ bằng chứng."}]
        return {"iteration": iteration, "steps": steps, "messages": updated,
                "tool_call_count": tool_call_count, "done": False}

    if not answer:
        steps.append({"type": "error", "content": "LLM tra ve rong."})
        return {"iteration": iteration, "steps": steps, "done": True}

    tool_name, tool_params = ctx.parse(answer)

    if (tool_name is None and current_plan_step
            and step_contract["mode"] == "direct_answer" and answer):
        tool_name, tool_params = "_step_answer", {"content": answer}
    logger.info("[NODE: reason] selected tool=%s action=%s (model output)",
                tool_name or "final_answer", (tool_params or {}).get("action", ""))

    # ── Step answer: inline response for a general-knowledge plan step ──
    if tool_name == "_step_answer":
        step_content = (tool_params or {}).get("content", answer)

        if current_plan_step is None:
            logger.info("[NODE: reason] Treating step_answer without an active plan as final_answer.")
            steps.append({"type": "final_answer", "content": step_content,
                          "iteration": iteration})
            updated_messages = list(state["messages"])
            updated_messages.append({"role": "assistant", "content": answer})
            return {
                "iteration": iteration,
                "steps": steps,
                "messages": updated_messages,
                "tool_call_count": tool_call_count,
                "synthesize_after_rag": False,
                "done": True,
            }

        # HARD CONSTRAINT for hallucination prevention
        if (current_plan_step and step_contract["mode"] == "rag_search"
                and not has_rag_evidence_for_step(steps, done_count)):
            logger.warning(f"[NODE: reason] Blocking hallucinated step_answer for step: {current_plan_step}")
            steps.append({"type": "validation_error", "content": f"Step '{current_plan_step}' requires project-specific context. Use rag_search.", "iteration": iteration})
            updated_messages = list(state["messages"])
            updated_messages.append({"role": "assistant", "content": answer})
            updated_messages.append({"role": "user", "content": f"LỖI: Bước '{current_plan_step}' CẦN thông tin nội bộ. BẠN BẮT BUỘC PHẢI GỌI TOOL `rag_search` (với tham số query phù hợp) để tìm kiếm, KHÔNG ĐƯỢC tự bịa đặt bằng step_answer."})
            return {
                "iteration": iteration,
                "steps": steps,
                "messages": updated_messages,
                "tool_call_count": tool_call_count,
                "error_count": state.get("error_count", 0) + 1
            }

        logger.debug(f"[AGENT TRACE] ── Reason: step_answer for plan step {done_count}.")
        logger.info("[NODE: reason] LLM trả lời trực tiếp cho bước kế hoạch (step_answer).")
        steps.append({"type": "step_answer", "content": step_content,
                      "iteration": iteration})
        reflection_entry: dict[str, Any] = {
            "type": "reflection", "tool": "step_answer",
            "result": {"passed": True, "decision": "continue",
                       "reason": "General-knowledge step answered inline without tool."},
            "iteration": iteration,
        }
        if current_plan_step is not None:
            reflection_entry["plan_step_index"] = done_count
        steps.append(reflection_entry)
        updated_messages = list(state["messages"])
        updated_messages.append({"role": "assistant", "content": answer})
        return {
            "iteration": iteration,
            "steps": steps,
            "messages": updated_messages,
            "tool_call_count": tool_call_count,
            "synthesize_after_rag": False,
            "done": False,
        }

    if tool_name is None:
        logger.debug("[AGENT TRACE] ── Reason: final answer.")
        logger.info("[NODE: reason] LLM quyết định dừng (final_answer).")
        coding_missing = coding_status.missing if coding_status else ()

        plan_missing = bool(
            state.get("enforce_plan_completion")
            and not state.get("enforce_coding_workflow")
            and plan and done_count < len(plan)
        )
        if coding_missing or plan_missing:
            remaining = (plan or [])[done_count:] if plan_missing else []
            logger.warning(
                "[NODE: reason] Ignoring premature final answer; missing coding=%s remaining plan=%s",
                coding_missing, remaining,
            )
            steps.append({
                "type": "coding_incomplete" if coding_missing else "plan_incomplete",
                "remaining": remaining,
                "missing": list(coding_missing),
                "iteration": iteration,
            })
            updated_messages = list(state["messages"])
            updated_messages.append({"role": "assistant", "content": answer})
            coding_instruction = (
                f" {coding_workflow_guidance(coding_status)}"
                if coding_status and coding_missing else ""
            )
            prefix = "Coding task" if coding_missing else "Kế hoạch"
            updated_messages.append({
                "role": "user",
                "content": (
                    f"{prefix} chưa đủ bằng chứng để kết thúc. Không được kết thúc hoặc chỉ giải thích. "
                    f"Còn các bước kế hoạch: {json.dumps(remaining, ensure_ascii=False)}."
                    f"{coding_instruction} "
                    "Hãy tiếp tục bằng đúng một JSON tool_call phù hợp."
                ),
            })
            return {
                "iteration": iteration,
                "steps": steps,
                "messages": updated_messages,
                "tool_call_count": tool_call_count,
                "done": False,
            }

        citation = format_code_citation(state.get("messages", []), steps)
        if citation:
            logger.info("[NODE: reason] Dùng formatter trích dẫn code đầy đủ từ read_file result.")
            answer = citation
        elif any(step.get("type") == "tool_call" and step.get("tool") == "rag_search"
                 for step in steps):
            answer = clean_project_research_answer(answer)
        steps.append({"type": "final_answer", "content": answer})
        return {"iteration": iteration, "steps": steps, "done": True}

    # ── Validation error: xử lý ngay, không gửi sang tool node ────────
    if tool_name == "_validation_error":
        error_detail = (tool_params or {}).get("error", "Unknown validation error")
        original_tool = (tool_params or {}).get("tool", "unknown")
        logger.warning(
            "[NODE: reason] Validation error cho tool '%s': %s",
            original_tool, error_detail,
        )
        logger.debug(
            f"[AGENT TRACE] ── Reason: VALIDATION ERROR cho '{original_tool}': {error_detail}",
        )
        steps.append({
            "type": "validation_error", "tool": original_tool,
            "error": error_detail, "iteration": iteration,
        })
        updated_messages = list(state["messages"])
        updated_messages.append({"role": "assistant", "content": answer})
        updated_messages.append({
            "role": "user",
            "content": (
                f"LỖI: Bạn vừa gọi tool '{original_tool}' nhưng tham số KHÔNG HỢP LỆ.\n"
                f"Chi tiết: {error_detail}\n\n"
                "Hãy đọc danh sách tool/action hợp lệ, đối chiếu với bước hiện tại "
                "và gọi lại tool phù hợp. KHÔNG ĐƯỢC trả lời bằng văn bản — "
                "chỉ trả về JSON tool_call."
            ),
        })
        return {
            "iteration":       iteration,
            "steps":           steps,
            "messages":        updated_messages,
            "tool_call_count": tool_call_count,
        }

    # ── Guard: transfer_to_toolapp_agent block ────────
    if tool_name == "transfer_to_toolapp_agent":
        logger.warning("[NODE: reason] LLM called transfer_to_toolapp_agent, bouncing back to force application_action.")
        logger.debug("[AGENT TRACE] ── Reason: VALIDATION ERROR: called transfer_to_toolapp_agent")
        steps.append({
            "type": "validation_error", "tool": tool_name,
            "error": "Forbidden direct call to transfer_to_toolapp_agent.", "iteration": iteration,
        })
        updated_messages = list(state["messages"])
        updated_messages.append({"role": "assistant", "content": answer})
        updated_messages.append({
            "role": "user",
            "content": (
                "LỖI: KHÔNG ĐƯỢC gọi transfer_to_toolapp_agent cho thao tác UI. "
                "Hãy gọi TRỰC TIẾP tool 'application_action' với action canonical phù hợp (ví dụ: viewer.load_2d)."
            ),
        })
        return {
            "iteration":       iteration,
            "steps":           steps,
            "messages":        updated_messages,
            "tool_call_count": tool_call_count,
        }

    # ── Pre-dispatch guard: right tool, wrong desktop action ─────────────
    if tool_name == "application_action" and current_plan_step:
        selected_action = str((tool_params or {}).get("action", ""))
        if step_matches_action(current_plan_step, selected_action) is False:
            expected = [name for name, _ in rank_actions_for_step(current_plan_step)][:3]
            rejections = sum(
                1 for step in steps
                if step.get("type") == "validation_error"
                and step.get("reason") == "step_action_mismatch"
                and step.get("plan_step_index") == done_count
            )
            logger.warning("[NODE: reason] Step/action mismatch | step=%s | selected=%s | expected=%s | rejections=%d",
                           current_plan_step, selected_action, expected, rejections)
            if rejections < MAX_STEP_MISMATCH_REJECTIONS:
                record_step_guard("rejected")
                steps.append({"type": "validation_error", "reason": "step_action_mismatch",
                              "tool": tool_name, "plan_step_index": done_count,
                              "error": f"'{selected_action}' does not match step '{current_plan_step}'",
                              "iteration": iteration})
                updated_messages = list(state["messages"])
                updated_messages.append({"role": "assistant", "content": answer})
                updated_messages.append({"role": "user", "content": (
                    f"LỖI: Action '{selected_action}' KHÔNG khớp bước hiện tại \"{current_plan_step}\". "
                    f"Action khớp theo manifest: {', '.join(expected)}. "
                    "Hãy gọi lại application_action với action đúng. Chỉ trả về JSON tool_call.")})
                return {"iteration": iteration, "steps": steps, "messages": updated_messages,
                        "tool_call_count": tool_call_count}
            record_step_guard("fallthrough")
            logger.error("[NODE: reason] Step guard exhausted; dispatching and relying on Reflect.")

    clean_answer = strip_think_tags(answer)
    think_text = clean_answer
    if "```tool_call" in clean_answer:
        think_text = clean_answer.split("```tool_call")[0].strip()
    elif "{" in clean_answer:
        think_text = clean_answer.split("{")[0].strip()
    if think_text:
        steps.append({"type": "thinking", "content": think_text,
                      "iteration": iteration})

    params = tool_params or {}
    _VOLATILE_PARAM_KEYS = {"request_id", "workflow_id"}
    def _stable_fingerprint(tool: str, params: dict) -> str:
        stable = {k: v for k, v in (params or {}).items() if k not in _VOLATILE_PARAM_KEYS}
        return json.dumps({"tool": tool, "params": stable}, ensure_ascii=False, sort_keys=True)

    fingerprint = _stable_fingerprint(tool_name, params)
    failed_calls = set()
    for index, step in enumerate(steps):
        if step.get("type") != "reflection" or step.get("result", {}).get("passed") is not False:
            continue
        for prior in reversed(steps[:index]):
            if prior.get("type") == "tool_call":
                failed_calls.add(json.dumps({"tool": prior.get("tool"), "params": prior.get("params", {})},
                                             ensure_ascii=False, sort_keys=True))
                break
    if fingerprint in failed_calls:
        steps.append({"type": "validation_error", "tool": tool_name,
                      "error": "Identical tool call already failed review; choose a different query, scope, or tool.",
                      "iteration": iteration})
        updated_messages = list(state["messages"])
        updated_messages.append({"role": "user", "content":
                                 "Không được lặp lại tool call vừa thất bại. Hãy đổi tham số/phạm vi hoặc chọn tool khám phá bổ sung."})
        return {"iteration": iteration, "steps": steps, "messages": updated_messages,
                "tool_call_count": tool_call_count}

    idempotency_key = ""
    if ctx.select_specialist:
        delegation = ctx.select_specialist(tool_name, params)
        idempotency_key = delegation.get("idempotency_key", "")
        steps.append({"type": "delegation", "agent": delegation.get("specialist", "supervisor"),
                      "tool": tool_name, "idempotency_key": idempotency_key,
                      "remote_endpoint": delegation.get("remote_endpoint"),
                      "iteration": iteration})

    logger.debug(f"[AGENT TRACE] ── Reason: tool call → tool='{tool_name}', params={str(params)[:80]}")
    logger.info("[NODE: reason] LLM quyết định gọi tool: %s, params: %s", tool_name, params)
    tool_call = {"type": "tool_call", "tool": tool_name,
                  "params": params, "idempotency_key": idempotency_key,
                  "iteration": iteration}
    if current_plan_step is not None:
        tool_call["plan_step_index"] = done_count
    steps.append(tool_call)

    updated_messages = list(state["messages"])
    updated_messages.append({"role": "assistant", "content": answer})

    approval_scope = state.get("approval_scope")
    if not approval_scope:
        scope_payload = json.dumps({"plan": plan, "spec": state.get("plan_spec")},
                                   ensure_ascii=False, sort_keys=True)
        approval_scope = hashlib.sha256(scope_payload.encode()).hexdigest()[:16]

    if ctx.needs_approval(tool_name) and not approval_covers_run(state, tool_name, params, ctx):
        spec = state.get("plan_spec") or {}
        preview = {
            "scope_id": approval_scope,
            "goal": spec.get("goal", ""),
            "affected_areas": spec.get("affected_areas", []),
            "acceptance_criteria": spec.get("acceptance_criteria", []),
            "tool": tool_name, "params": params,
            "remaining_steps": plan[done_count:] if plan and done_count < len(plan) else [],
        }
        return {
            "iteration":    iteration,
            "steps":        steps,
            "messages":     updated_messages,
            "pending_tool": {"tool": tool_name, "params": params,
                              "approval_scope": approval_scope,
                              "approval_preview": preview},
            "approval_scope": approval_scope,
            "done":         True,
        }

    return {
        "iteration":       iteration,
        "steps":           steps,
        "messages":        updated_messages,
        "tool_call_count": tool_call_count,
    }


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
