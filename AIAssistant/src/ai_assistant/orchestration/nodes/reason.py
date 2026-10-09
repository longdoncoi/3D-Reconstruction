"""Reason node for the LangGraph orchestration.

This node runs the core ReAct loop, evaluating the current state and plan,
and querying the LLM for the next tool call or a final answer.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

from ai_assistant.config.logging import get_agent_logger
from ai_assistant.llm.inference import strip_think_tags
from ai_assistant.observability import record_step_guard
from ai_assistant.orchestration.specialists.code import coding_workflow_guidance, coding_workflow_status
from ai_assistant.tools.action_manifest import rank_actions_for_step, step_matches_action

from ..helpers import (
    clean_project_research_answer,
    completed_plan_steps,
    has_rag_evidence_for_step,
    has_rag_evidence_without_plan,
    plan_step_execution_contract,
)
from ..prompts import REASONER_PROMPT
from ..state import (
    MAX_STEP_MISMATCH_REJECTIONS,
    OBSERVATION_SUMMARY_THRESHOLD,
    AgentState,
    Completion,
    NeedsApproval,
    Parser,
    SelectSpecialist,
)
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

    messages = list(state["messages"])
    plan = state.get("plan")
    tool_call_count = state.get("tool_call_count", 0)
    done_count = completed_plan_steps(steps, plan)

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
        failed_attempts = sum(
            1 for step in steps
            if (step.get("type") == "reflection"
                and step.get("result", {}).get("passed") is False)
        )
        hint_index = done_count + 1 if plan else completed_ui_actions + 1
        logger.info("[NODE: reason] ToolApp hint | plan step %d/%d | expected action=%s | failed attempts=%d",
                    hint_index, progress_total, expected_action.get("action"), failed_attempts)
        logger.debug(f"[AGENT TRACE] ── Reason: ToolApp hint step {hint_index}/{progress_total}; gọi LLM tool-calling")
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
            return {
                "iteration": iteration,
                "steps": steps,
                "messages": updated_messages,
                "tool_call_count": tool_call_count,
                "synthesize_after_rag": False,
                "done": True,
            }

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
    _AGENT_HARD_LIMIT = 14000
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

    if forced_envelope is not None:
        answer = json.dumps(forced_envelope, ensure_ascii=False)
        logger.info("[NODE: reason] Enforcing execution contract for step/request '%s': %s",
                    current_plan_step or user_request, forced_envelope)
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
            remaining = plan[done_count:] if plan_missing else []
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
        
    if ctx.needs_approval(tool_name) and not state.get("approval_granted"):
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
