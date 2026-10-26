"""Tool execution node for the LangGraph orchestration.

This node executes the tool call selected by the Reasoner and updates
the state with the result, including observation summarization for RAG.

A plan may declare *parallel groups* (ADR 0044): consecutive independent
read-only steps that this node fans out **inside** the node, leaving the graph
topology, the checkpointer and every router untouched. A group merges back as
ordered ``tool_call``/``tool_result`` pairs labelled with ``batch_id`` and
``branch_index``, so reflection, plan-progress counting and HITL see exactly
what a sequential run would have produced. A group that is not provably
eligible falls back to the sequential path — the behaviour that shipped before
the feature — and reuses the same execution seam.
"""
from __future__ import annotations

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, cast

from ai_assistant.config.logging import get_agent_logger
from ai_assistant.domain.governance import redact_value
from ai_assistant.domain.parallel import PARALLEL_MAX_DEFAULT
from ai_assistant.domain.tools import describe_tool_result
from ai_assistant.observability import span
from ai_assistant.tools.action_manifest import normalize_text, rank_actions_for_step

from ..helpers import plan_step_execution_contract, valid_parallel_group
from ..state import (
    COMPLETION_TOOLS,
    LOW_RISK_TOOLS,
    SEMANTIC_REFLECTION_TOOLS,
    AgentState,
    Executor,
    Step,
    VerifyResult,
    has_step_type,
)
from .summarize import summarize_messages

logger = get_agent_logger("tool")


@dataclass
class ToolContext:
    """Dependencies for the tool node.

    The last three fields are the parallel-batch seams (ADR 0044). All three
    are defaulted so every existing construction site — the graph, tests,
    embeds — keeps working unchanged, and a bare context simply never batches.
    """

    execute: Executor
    verify_result: VerifyResult | None
    cancel_checker: Callable[[], bool]
    #: Tools whose invocation needs a human grant. A group containing one takes
    #: the sequential path: one grant covers exactly one invocation.
    requires_approval: frozenset[str] = frozenset()
    #: Precompute the supervisor delegation of a group's calls, in the
    #: dispatching thread, so branches only read a warm cache (ADR 0044 §6).
    warm_delegations: Callable[[list[tuple[str, dict[str, Any]]]], None] | None = None
    #: Fan-out width for one group; ``<= 1`` keeps the sequential path.
    max_parallel: int = PARALLEL_MAX_DEFAULT


def _rag_synthesis_state(state: AgentState, steps: list[Step],
                         result: dict[str, Any], last_step: Step,
                         error_count: int, *,
                         calls: int = 1,
                         batch_evidence: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the state update that asks the model to answer a RAG step.

    ``calls`` lets the parallel path charge a whole group to the tool-call
    budget in one update, and ``batch_evidence`` carries the siblings'
    observations so the following turns can answer their own steps from
    context instead of searching again.
    """
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
    if batch_evidence:
        messages.append({
            "role": "user",
            "content": (
                "Bằng chứng của các bước còn lại trong nhóm chạy song song"
                " (giữ nguyên thứ tự bước, dùng cho các bước sau):\n"
                f"```json\n{json.dumps(batch_evidence, ensure_ascii=False)[:evidence_limit]}\n```"
            ),
        })
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
            "tool_call_count": state.get("tool_call_count", 0) + calls,
            "error_count": error_count, "synthesize_after_rag": True}


def _rag_synthesis_update(state: AgentState, steps: list[Step],
                          tool_name: str, result: dict[str, Any],
                          last_step: Step, error_count: int) -> dict[str, Any] | None:
    """Build the step_answer synthesis request for verified RAG evidence.

    Returns the state update when the executed tool was ``rag_search``,
    else ``None`` so the caller continues with the generic routing.
    """
    if tool_name != "rag_search":
        return None
    return _rag_synthesis_state(state, steps, result, last_step, error_count)


def _completion_announcement_update(state: AgentState, steps: list[Step],
                                      tool_name: str, result: dict[str, Any],
                                      error_count: int) -> dict[str, Any] | None:
    """Announce a dispatched desktop action instead of reflecting on it.

    Returns the state update when the tool completed a user-visible action,
    else ``None`` so the caller continues with the generic routing.
    """
    if tool_name not in COMPLETION_TOOLS or not result.get("success"):
        return None
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


def _verified_reflection_update(state: AgentState, steps: list[Step],
                                  messages: list[dict[str, str]], tool_name: str,
                                  verification: dict[str, Any], result: dict[str, Any],
                                  last_step: Step, error_count: int,
                                  allowed_tools: Any, reason: str,
                                  require_membership: bool = True) -> dict[str, Any] | None:
    """Shortcut a verified result with a contract reflection, when eligible.

    The low-risk shortcut fires for members of its tool set; the
    semantic-not-required shortcut fires for everyone *outside* its set.
    Returns the state update on eligibility, else ``None``.
    """
    member = (tool_name in allowed_tools) if require_membership else (tool_name not in allowed_tools)
    if not member or not verification.get("passed") or "error" in result:
        return None
    steps.append({
        "type": "reflection", "tool": tool_name,
        "result": {"passed": True, "decision": "continue", "reason": reason},
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


@dataclass
class _ToolDispatch:
    """One pending tool call after pairing, execution and error counting."""

    tool_name: str
    params: dict[str, Any]
    result: dict[str, Any]
    last_step: Step
    error_count: int


@dataclass
class _BatchDispatch:
    """One executed parallel group, ready to be merged into the state."""

    batch_id: str
    head_step: Step
    #: Position of the head's ``tool_call`` in ``state["steps"]``.
    head_index: int
    #: ``(tool, params, plan index)`` in group order; the head is first.
    calls: list[tuple[str, dict[str, Any], int]]
    results: list[dict[str, Any]]
    #: True when the cooperative cancel gate skipped at least one branch.
    cancelled: bool


def _execute_one(ctx: ToolContext, tool_name: str,
                 params: dict[str, Any]) -> dict[str, Any]:
    """Run one call, containing its failure and logging only its shape.

    Shared by the sequential path and every batch branch: an exception becomes
    ``{"error": ...}`` (a branch that explodes must not kill the group, and the
    circuit breaker must be able to see the failure), and ADR 0032 keeps the
    log to the *shape* of what the tool returned.
    """
    logger.info("[NODE: tool] Thực thi: %s, params: %s", tool_name, str(redact_value(params))[:200])
    logger.debug("\n--- [LOG: TOOL NODE] Thuc thi: %s ---", tool_name)
    try:
        result = ctx.execute(tool_name, params)
        summary = describe_tool_result(result)
    except Exception as error:
        result = {"error": f"Tool exception: {error}"}
        logger.exception("[NODE: tool] Exception khi thực thi '%s'", tool_name)
        logger.debug("[LOG: TOOL NODE] Loi: %s", describe_tool_result(result))
        return result
    logger.info("[NODE: tool] Kết quả: %s", summary)
    logger.debug("[LOG: TOOL NODE] Ket qua: %s", summary)
    return result


def _rag_params(contract: dict[str, Any]) -> dict[str, Any]:
    """The tool parameters a RAG contract prescribes."""
    params: dict[str, Any] = {"query": contract["query"]}
    if contract.get("top_k"):
        params["top_k"] = contract["top_k"]
    return params


def _member_group(spec: dict[str, Any], head: int, plan: list[str]) -> list[int] | None:
    """The validated group whose first member is *head*, if there is one.

    The spec arrives from the planner in this process but, on a resumed run,
    from a checkpoint written by another one: it is re-validated here, with the
    same function the parser uses, before it can enable anything.
    """
    groups = spec.get("parallel_groups")
    if not isinstance(groups, list):
        return None
    for raw_group in groups:
        group = valid_parallel_group(raw_group, len(plan))
        if group is not None and group[0] == head:
            return group
    return None


def _plan_batch(state: AgentState, ctx: ToolContext, head_step: Step,
                tool_name: str, params: dict[str, Any]) -> list[tuple[str, dict[str, Any], int]] | None:
    """The member calls of the group this head starts, else ``None``.

    Fail-closed, and every refusal is a *group* refusal (ADR 0044 §3): a group
    runs only when each member is a declared ``rag_search`` step and no member
    needs an approval grant. One approval-gated member sends the whole group
    down the sequential path — a grant covers exactly one invocation, and the
    user must still see the steps in order. A declared ``model_selected`` or
    ``direct`` step never batches: the model owes that step its choice or its
    answer, and neither can happen inside a worker thread.
    """
    plan = state.get("plan")
    spec = state.get("plan_spec")
    head = head_step.get("plan_step_index")
    if not plan:
        return None
    if not isinstance(spec, dict):
        return None
    if not isinstance(head, int):
        return None
    if tool_name != "rag_search":
        return None
    if "rag_search" in ctx.requires_approval:
        return None
    group = _member_group(spec, head, plan)
    if group is None:
        return None
    kinds = spec.get("step_kinds")
    if not isinstance(kinds, list):
        return None
    calls: list[tuple[str, dict[str, Any], int]] = []
    for index in group:
        if index >= len(kinds):
            return None
        if kinds[index] != "rag_search":
            return None
        contract = plan_step_execution_contract(
            plan[index], rank_actions_for_step, normalize_text, kind="rag_search",
        )
        calls.append(("rag_search", _rag_params(contract), index))
    if ctx.max_parallel <= 1:
        return None
    # The head's own call is the model's (or the forced envelope's): its query
    # is honoured verbatim, the siblings are derived from their plan steps.
    calls[0] = (tool_name, params, head)
    return calls


def _run_batch(state: AgentState, ctx: ToolContext,
               calls: list[tuple[str, dict[str, Any], int]],
               head_step: Step, head_index: int) -> _BatchDispatch:
    """Execute the group concurrently, one bounded branch per member.

    The cancel gate runs in the dispatching thread, in group order, so *which*
    branches start is deterministic; every submitted branch is joined before
    this returns, so no thread outlives the state update. Each branch runs in
    a copy of the caller's context (``traceparent``/OTel context follows its
    own branch) and through the same executor the sequential path uses — there
    is no second gateway and no second executor.
    """
    if ctx.warm_delegations is not None:
        ctx.warm_delegations([(name, params) for name, params, _ in calls])
    results: list[dict[str, Any] | None] = [None] * len(calls)
    cancelled = False
    with span("agent.tool_batch", branches=len(calls)):
        pool = ThreadPoolExecutor(max_workers=min(ctx.max_parallel, len(calls)))
        futures: dict[int, Any] = {}
        try:
            for index, (name, params, _plan_index) in enumerate(calls):
                if ctx.cancel_checker():
                    cancelled = True
                    results[index] = {"error": "Tool batch branch cancelled."}
                    continue
                branch_context = contextvars.copy_context()
                futures[index] = pool.submit(branch_context.run, _execute_one, ctx, name, params)
            for index, future in futures.items():
                results[index] = future.result()
        finally:
            pool.shutdown(wait=True)
    return _BatchDispatch(
        batch_id=f"batch-{calls[0][2]}-{state.get('iteration', 0)}",
        head_step=head_step,
        head_index=head_index,
        calls=calls,
        results=cast(list[dict[str, Any]], results),
        cancelled=cancelled,
    )


def _batch_error_count(state: AgentState, results: list[dict[str, Any]]) -> int:
    """Error streak after a group: a clean group resets it, failures add up."""
    failures = sum(1 for result in results if "error" in result)
    return state.get("error_count", 0) + failures if failures else 0


def _record_batch(state: AgentState, ctx: ToolContext,
                  batch: _BatchDispatch) -> dict[str, Any]:
    """Merge one executed group into the state as ordered call/result pairs.

    The head's call is already in the log; the siblings are synthesised here so
    the call/result pairing the node depends on stays exact (one result per
    call). Only the head asks for a step answer — the synthesis request names
    the current step — and the siblings' evidence rides along in the same
    update so the following turns can answer their steps from context.
    """
    steps = list(state["steps"])
    # The dispatch paired the head by identity and carried its position, so the
    # head is stamped in place instead of being searched for again.
    steps[batch.head_index] = {**batch.head_step, "batch_id": batch.batch_id,
                               "branch_index": 0}
    for index, (tool_name, params, plan_index) in enumerate(batch.calls):
        if index == 0:
            continue
        steps.append({
            "type": "tool_call", "tool": tool_name, "params": params,
            "plan_step_index": plan_index, "batch_id": batch.batch_id,
            "branch_index": index, "iteration": state["iteration"],
        })
    for index, ((tool_name, params, plan_index), result) in enumerate(
            zip(batch.calls, batch.results, strict=True)):
        if ctx.verify_result is not None:
            steps.append({"type": "verification", "tool": tool_name,
                          "result": ctx.verify_result(tool_name, params, result),
                          "iteration": state["iteration"]})
        steps.append({
            "type": "tool_result", "tool": tool_name, "result": result,
            "plan_step_index": plan_index, "batch_id": batch.batch_id,
            "branch_index": index, "iteration": state["iteration"],
        })
    error_count = _batch_error_count(state, batch.results)
    if error_count >= 3:
        steps.append({"type": "final_answer",
                      "content": "Ngắt mạch (Circuit Breaker): nhóm tool song song gặp lỗi liên tiếp. Dừng tác vụ để tránh vòng lặp."})
        return {"steps": steps, "done": True, "error_count": error_count}
    if batch.cancelled:
        steps.append({"type": "cancelled", "content": "Task cancelled cooperatively.",
                      "iteration": state["iteration"]})
        return {"steps": steps, "done": True, "cancelled": True, "error_count": error_count}
    plan = state.get("plan") or []
    siblings = [
        {"step": plan[plan_index], "result": result}
        for (_tool, _params, plan_index), result in zip(batch.calls, batch.results, strict=True)
    ][1:]
    return _rag_synthesis_state(state, steps, batch.results[0], batch.head_step,
                                error_count, calls=len(batch.calls),
                                batch_evidence=siblings)


def _dispatch(state: AgentState, ctx: ToolContext) -> _ToolDispatch | _BatchDispatch | dict[str, Any]:
    """Cancel gate, pending-call pairing, payload validation and execution.

    Phase 1 of the tool node (review #18). Returns a complete state update
    when the run must stop before anything is recorded — cooperative cancel,
    a tool node reached without a pending call, an invalid payload, or a Qt
    acknowledgement the UI still owes us; a :class:`_BatchDispatch` when the
    head starts a parallel group; otherwise the record handed to :func:`_record`.
    """
    if ctx.cancel_checker():
        steps = list(state["steps"])
        steps.append({"type": "cancelled", "content": "Task cancelled cooperatively.",
                      "iteration": state.get("iteration", 0)})
        return {"steps": steps, "done": True, "cancelled": True}

    call_steps = [s for s in state["steps"] if has_step_type(s, "tool_call")]
    result_steps = [s for s in state["steps"] if has_step_type(s, "tool_result")]
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

    batch = _plan_batch(state, ctx, last_step, tool_name, params)
    if batch is not None:
        head_index = next(
            index for index, step in enumerate(state["steps"]) if step is last_step
        )
        logger.info("[NODE: tool] Thực thi nhóm song song %d bước (head=%d).",
                    len(batch), batch[0][2])
        return _run_batch(state, ctx, batch, last_step, head_index)

    result = _execute_one(ctx, tool_name, params)

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
    return _ToolDispatch(tool_name, params, result, last_step, error_count)


def _record(state: AgentState, ctx: ToolContext, dispatched: _ToolDispatch) -> dict[str, Any]:
    """Write the observable outcome of an executed tool (review #18, phase 2).

    Result steps, the circuit breaker, optional verification, RAG/completion
    shortcuts, the tool-result message and the verified-reflection fast paths.
    """
    tool_name = dispatched.tool_name
    params = dispatched.params
    result = dispatched.result
    last_step = dispatched.last_step
    error_count = dispatched.error_count

    steps = list(state["steps"])
    tool_result_entry: Step = {
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

    rag_synthesis = _rag_synthesis_update(state, steps, tool_name, result, last_step, error_count)
    if rag_synthesis is not None:
        return rag_synthesis

    result_text = json.dumps(result, ensure_ascii=False, indent=2)
    if len(result_text) > 8000:
        result_text = result_text[:8000] + "\n... [truncated]"

    completion = _completion_announcement_update(state, steps, tool_name, result, error_count)
    if completion is not None:
        return completion

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

    low_risk = _verified_reflection_update(
        state, steps, messages, tool_name, verification, result, last_step,
        error_count, LOW_RISK_TOOLS, "Verified low-risk read-only tool result.")
    if low_risk is not None:
        return low_risk

    semantic_free = _verified_reflection_update(
        state, steps, messages, tool_name, verification, result, last_step,
        error_count, SEMANTIC_REFLECTION_TOOLS,
        "Verified result contract; semantic critic not required.",
        require_membership=False)
    if semantic_free is not None:
        return semantic_free

    return {
        "steps":           steps,
        "messages":        messages,
        "tool_call_count": state.get("tool_call_count", 0) + 1,
        "error_count":     error_count,
    }


def tool_node(state: AgentState, ctx: ToolContext) -> dict[str, Any]:
    """Execute the chosen tool and verify its result.

    Thin coordinator (review #18): :func:`_dispatch` pairs and runs the
    pending call (or the group it heads), :func:`_record`/:func:`_record_batch`
    write everything the UI and routers then observe.
    """
    logger.info("[NODE: tool] Bắt đầu thực thi tool.")
    dispatched = _dispatch(state, ctx)
    if isinstance(dispatched, dict):
        return dispatched
    if isinstance(dispatched, _BatchDispatch):
        return _record_batch(state, ctx, dispatched)
    return _record(state, ctx, dispatched)


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
            and has_step_type(state["steps"][-1], "reflection")):
        logger.info("[ROUTER: after_tool] → REASON (verified low-risk tool)")
        return "reason"
    logger.info("[ROUTER: after_tool] → REFLECT")
    logger.debug("[LOG: TOOL ROUTER] -> Reflect Node ---")
    return "reflect"
