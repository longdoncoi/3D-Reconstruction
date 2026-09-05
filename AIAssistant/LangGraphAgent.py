"""LangGraph orchestration for the local 3D-Reconstruction agent.

The graph owns the agent loop; the host application owns model inference and
tool execution.  This keeps Qt-specific actions outside the Python process.

Architecture (ReAct + Plan-and-Execute):
  START -> plan -> plan_reflect -> reason -> tool -> reflect (loop) -> END

  plan   : Sinh ke hoach (danh sach cac buoc) truoc khi bat dau tool loop.
  plan_reflect : Đánh giá kế hoạch một lần; nếu không đạt thì quay lại plan.
  reason : Quan sat ket qua tool truoc do, quyet dinh buoc tiep theo hoac ket thuc.
  tool   : Thuc thi tool duoc chon.

── REASON / REFLECT POLICY (2026-08-27) ─────────────────────────────────────
Each plan step is selected by the LLM.  A desktop-action hint, when available,
is advisory only; it is never a request-specific hard-coded dispatch. Reflect
reviews the actual tool result against the current plan step. A failed review
is appended to the next Reason context so the model can choose a different
valid tool or parameters. Invalid tool/action names are handled by the normal
validation feedback loop.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from modules.action_manifest import (
    action_matches_plan_step,
    canonical_action,
    match_action_intent,
    split_step_by_manifest_phrases,
)
from modules.agent_logging import get_agent_logger
from modules.checkpointing import build_checkpointer
from modules.coding_agent import coding_workflow_guidance, coding_workflow_status
from modules.observability import span

logger = get_agent_logger("reasoning")

# Completion-signal tools: ngay sau khi success -> done=True, khong lap lai
_COMPLETION_TOOLS = {"application_action"}
_LOW_RISK_TOOLS = {
    "read_file", "list_directory", "find_files", "search_text", "analyze_code",
    "git_diff", "get_project_status", "validate_file", "rag_search",
}
_OBSERVATION_SUMMARY_THRESHOLD = 4   # tong ket observations sau N tool calls
_OBSERVATION_KEEP_LAST = 2           # giu lai N tool-result messages gan nhat
_CONTEXT_SYSTEM_LIMIT = 6000
_CONTEXT_MESSAGE_LIMIT = 1800
_CONTEXT_TOTAL_LIMIT = 14000
_SEMANTIC_REFLECTION_TOOLS = {"run_command", "write_file", "patch_file", "replace_file_content", "multi_replace_file_content", "create_directory", "application_action"}

_PLANNER_PROMPT = """Bạn là Planner (Người lập kế hoạch) của hệ thống AI Assistant.
Nhiệm vụ của bạn là phân tích yêu cầu của người dùng và lập ra một kế hoạch ngắn gọn, từng bước một.
KHÔNG sử dụng bất kỳ công cụ (tool) nào.
Trả lời CHÍNH XÁC theo JSON object với schema:
{"requires_plan": true/false, "goal": "mục tiêu chuẩn hóa", "affected_areas": ["..."],
 "acceptance_criteria": ["tiêu chí kiểm chứng được"], "verification_commands": ["lệnh an toàn"],
 "steps": ["bước 1", "bước 2", ...]}
Đặt requires_plan=false và các mảng rỗng nếu yêu cầu chỉ có một bước độc lập.
Với yêu cầu điều khiển giao diện (mở/tải ảnh, mô hình, DICOM, tái tạo),
hãy lập các bước thực thi trực tiếp trong ứng dụng; không lập bước mở ứng dụng, tìm tài liệu,
tìm mã nguồn, RAG hoặc tìm vị trí tệp.
Với yêu cầu điều khiển giao diện gồm nhiều hành động nối bằng "và"/"sau đó"/"rồi",
BẮT BUỘC tách mỗi hành động thành một bước riêng biệt trong "steps".
Không được gộp 2 hành động canonical vào cùng một bước.
Với yêu cầu engineering/coding có thay đổi repository, kế hoạch phải bao phủ
đọc/định vị source liên quan, thay đổi được duyệt, xem lại diff và kiểm chứng
bằng test/lint/compile/build phù hợp. Không coi việc tìm thấy file là đã hoàn thành."""

_CRITIC_PROMPT = """Bạn là Critic (Người phản biện) của hệ thống AI Assistant.
Nhiệm vụ của bạn là đánh giá xem kết quả thực thi của một tool có thực sự hoàn thành mục tiêu trong yêu cầu ban đầu của người dùng hay không. Nếu tool được gọi hoàn toàn sai mục đích (gọi nhầm tool), hãy đánh giá là thất bại (passed: false).
Hãy trả về ĐÚNG MỘT JSON object (không kèm text) với format:
{"passed": true/false, "decision": "continue"/"revise", "reason": "Lý do chi tiết..."}
- passed: true nếu kết quả trả về hợp lệ, thành công VÀ đúng mục tiêu của người dùng. false nếu có lỗi, sai mục tiêu, hoặc kết quả không đúng mong đợi.
- decision: "continue" nếu có thể đi tiếp, "revise" nếu cần reasoner sửa lỗi hoặc gọi tool khác.
- reason: giải thích ngắn gọn tại sao."""

_PLAN_CRITIC_PROMPT = """Bạn là Critic kiểm tra kế hoạch của hệ thống AI Assistant.
Đánh giá kế hoạch trước khi bất kỳ tool nào được gọi. Kế hoạch đạt (passed=true)
khi ngắn gọn, không trùng bước, bao phủ đúng các mục tiêu người dùng và mỗi bước
đều là một hành động cần thiết có thể thực thi/kiểm chứng. Với yêu cầu giao diện,
không thêm bước chuẩn bị chung như tìm tài liệu, RAG, tìm mã nguồn hoặc tìm vị trí
tệp nếu người dùng không yêu cầu. Nếu kế hoạch thừa, thiếu hoặc có bước không thể
ánh xạ tới mục tiêu/tool phù hợp, trả về passed=false để Planner sinh lại.
Trả về ĐÚNG MỘT JSON object:
{"passed": true/false, "decision": "continue"/"revise", "reason": "Lý do chi tiết..."}
"""

_REASONER_PROMPT = """Bạn là Reasoner (Người ra quyết định) của hệ thống AI Assistant.
Nhiệm vụ của bạn là dựa vào yêu cầu của người dùng, ngữ cảnh hiện tại và kế hoạch đã đề ra để quyết định bước đi tiếp theo.
Bạn phải tuân thủ nghiêm ngặt các quy tắc và định dạng kết quả (JSON) đã được hướng dẫn.
"""


def _completed_plan_steps(steps: list[dict[str, Any]], plan: list[str] | None) -> int:
    """Count consecutive accepted plan steps without double-counting retries.

    Each tool call is stamped with the plan step it was intended to advance.
    Older persisted sessions lack that stamp, so retain the former sequential
    interpretation only for those sessions.  This keeps plan progress stable
    when a tool is retried or when post-plan evidence is collected.
    """
    if not plan:
        return 0
    reflections = [
        step for step in steps
        if step.get("type") == "reflection" and step.get("result", {}).get("passed") is True
    ]
    indexed = [step for step in reflections if isinstance(step.get("plan_step_index"), int)]
    if not indexed:
        return min(len(reflections), len(plan))
    accepted = {
        step["plan_step_index"] for step in indexed
        if 0 <= step["plan_step_index"] < len(plan)
    }
    completed = 0
    while completed in accepted:
        completed += 1
    return completed


def _normalise_plan_payload(payload: dict[str, Any]) -> tuple[list[str] | None, dict[str, Any]]:
    """Accept the current structured planner contract and older ``plan`` JSON."""
    requires_plan = bool(payload.get("requires_plan"))
    raw_steps = payload.get("steps", payload.get("plan", []))
    plan = [str(item).strip() for item in raw_steps if str(item).strip()] if isinstance(raw_steps, list) else []
    spec = {
        "requires_plan": requires_plan,
        "goal": str(payload.get("goal", "")).strip(),
        "affected_areas": [str(item) for item in payload.get("affected_areas", [])
                           if str(item).strip()] if isinstance(payload.get("affected_areas", []), list) else [],
        "acceptance_criteria": [str(item) for item in payload.get("acceptance_criteria", [])
                                if str(item).strip()] if isinstance(payload.get("acceptance_criteria", []), list) else [],
        "verification_commands": [str(item) for item in payload.get("verification_commands", [])
                                  if str(item).strip()] if isinstance(payload.get("verification_commands", []), list) else [],
        "steps": plan,
    }
    return (plan or None) if requires_plan else None, spec


class AgentState(TypedDict):
    messages:        list[dict[str, str]]   # lich su hoi thoai
    steps:           list[dict[str, Any]]   # danh sach steps (cho UI)
    iteration:       int                    # so vong lap da chay
    temperature:     float                  # nhiet do sinh van ban
    done:            bool                   # da hoan thanh chua
    pending_tool:    dict[str, Any] | None  # tool dang cho phe duyet
    plan:            list[str] | None       # ke hoach cac buoc
    plan_spec:       dict[str, Any] | None  # structured planner contract
    approval_granted: bool                  # one approval covers this plan scope
    approval_scope:   str                   # stable hash of task/plan scope
    cancelled:        bool                  # cooperative cancellation requested
    tool_call_count: int                    # so lan goi tool (trigger summarization)
    last_reflection: dict[str, Any] | None  # deterministic critic result
    error_count:     int                    # so lan tool loi lien tiep
    resume_with_reflection: bool            # resume after an asynchronous UI ACK
    skip_reflect:    bool                   # a verified low-risk tool was auto-reviewed
    required_ui_actions: list[dict[str, Any]]  # canonical UI actions required by the host intent matcher
    plan_verified:    bool                     # kế hoạch hiện tại đã qua Plan Reflect
    plan_attempts:    int                      # số lần Planner đã sinh kế hoạch
    plan_feedback:   str                      # phản hồi gần nhất cho Planner
    enforce_plan_completion: bool             # coding tasks cannot finish before plan steps pass
    enforce_coding_workflow: bool             # coding tasks need observable implementation evidence


Completion    = Callable[[list[dict[str, str]], float], str]
Parser        = Callable[[str], tuple[str | None, dict[str, Any] | None]]
Executor      = Callable[[str, dict[str, Any]], dict[str, Any]]
NeedsApproval = Callable[[str], bool]
SelectSpecialist = Callable[[str, dict[str, Any]], dict[str, Any]]
VerifyResult = Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]]
ReflectResult = Callable[[str, dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]]


class LocalAgentGraph:
    """ReAct + Plan-and-Execute graph cho local llama.cpp model."""

    def __init__(self, complete: Completion, parse: Parser, execute: Executor,
                 needs_approval: NeedsApproval, max_iterations: int,
                 emit: Callable[[dict[str, Any]], None] | None = None,
                 select_specialist: SelectSpecialist | None = None,
                 verify_result: VerifyResult | None = None,
                 reflect_result: ReflectResult | None = None,
                 plan_complete: Completion | None = None,
                 reflect_complete: Completion | None = None,
                 plan_reflect_complete: Completion | None = None,
                 cancel_checker: Callable[[], bool] | None = None) -> None:
        self._complete       = complete
        self._parse          = parse
        self._execute        = execute
        self._needs_approval = needs_approval
        self._max_iterations = max_iterations
        self._emit = emit
        self._select_specialist = select_specialist
        self._verify_result = verify_result
        self._reflect_result = reflect_result
        # Planner and critic require JSON schemas different from the tool/final
        # envelope used by the ReAct reasoner.
        self._plan_complete = plan_complete or complete
        self._reflect_complete = reflect_complete or complete
        # Kept separate so callers/tests can provide independent planner and
        # plan-review completions. If omitted, deterministic checks still run
        # and a plan is accepted without consuming the tool-critic callback.
        self._plan_reflect_complete = plan_reflect_complete
        self._cancel_checker = cancel_checker or (lambda: False)
        self._emitted_steps = 0

        builder = StateGraph(AgentState)
        builder.add_node("plan",   self._traced("plan", self._plan))
        builder.add_node("plan_reflect", self._traced("plan_reflect", self._plan_reflect))
        builder.add_node("reason", self._traced("reason", self._reason))
        builder.add_node("tool",   self._traced("tool", self._tool))
        builder.add_node("reflect", self._traced("reflect", self._reflect))

        # A resumed UI workflow already has an ACK result. Review that result
        # before reasoning again, rather than planning or dispatching another action.
        builder.add_conditional_edges(START, self._initial_node,
                                      {"plan": "plan", "reflect": "reflect"})
        builder.add_conditional_edges("plan", self._after_plan,
                                      {"plan_reflect": "plan_reflect", "reason": "reason"})
        builder.add_conditional_edges("plan_reflect", self._after_plan_reflect,
                                      {"plan": "plan", "reason": "reason", "end": END})
        builder.add_conditional_edges("reason", self._after_reason,
                                      {"tool": "tool", "reason": "reason", "end": END})
        # A tool normally returns to the reasoning loop.  UI actions and
        # approval-gated tools instead set ``done``/``pending_tool`` and must
        # stop immediately: their result is completed asynchronously by the
        # desktop client.  An unconditional edge here would call the model
        # again and dispatch a second desktop action before the first ACK.
        builder.add_conditional_edges("tool", self._after_tool,
                                      {"reason": "reason", "reflect": "reflect", "end": END})
        builder.add_conditional_edges("reflect", self._after_reflect,
                                      {"reason": "reason", "end": END})

        self._graph = builder.compile(checkpointer=build_checkpointer())

    def _traced(self, name: str, handler: Callable[[AgentState], dict[str, Any]]) -> Callable[[AgentState], dict[str, Any]]:
        def invoke(state: AgentState) -> dict[str, Any]:
            with span(f"agent.{name}", iteration=str(state.get("iteration", 0))):
                result = handler(state)
            if self._emit and result.get("steps"):
                steps = result["steps"]
                for step in steps[self._emitted_steps:]:
                    self._emit(step)
                self._emitted_steps = len(steps)
            return result
        return invoke

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, messages: list[dict[str, str]], session_id: str,
            temperature: float, steps: list[dict[str, Any]] | None = None,
            iteration: int = 0, resume_with_reflection: bool = False,
            required_ui_actions: list[dict[str, Any]] | None = None,
            supervisor_route: str | None = None,
            enforce_plan_completion: bool = False,
            approval_granted: bool = False,
            approval_scope: str = "") -> AgentState:
        self._emitted_steps = len(steps or [])
        restored_plan = next(
            (step.get("steps") for step in reversed(steps or []) if step.get("type") == "plan"),
            None,
        )
        prior_plan_review = next(
            (step.get("result", {}) for step in reversed(steps or [])
             if step.get("type") == "plan_reflection"), None)

        config = {"configurable": {"thread_id": session_id}}

        input_state: dict[str, Any] = {
            "messages":        messages,
            "steps":           steps or [],
            "iteration":       iteration,
            "temperature":     temperature,
            "done":            False,
            "pending_tool":    None,
            "plan":            restored_plan,
            "plan_spec":       next((step.get("spec") for step in reversed(steps or [])
                                      if step.get("type") == "plan"), None),
            "approval_granted": approval_granted,
            "approval_scope":   approval_scope,
            "cancelled":        False,
            "tool_call_count": 0,
            "last_reflection": None,
            "error_count":     0,
            "resume_with_reflection": resume_with_reflection,
            "skip_reflect":    False,
            "plan_verified":   bool(prior_plan_review and prior_plan_review.get("passed") is True),
            "plan_attempts":   sum(1 for step in (steps or []) if step.get("type") == "plan"),
            "plan_feedback":   "",
            "routing_plan":    supervisor_route,
            "enforce_plan_completion": enforce_plan_completion,
            "enforce_coding_workflow": enforce_plan_completion,
        }

        # FIX: `required_ui_actions` used to be overwritten with `[]` on every
        # call unless the host explicitly re-passed it — including on the
        # resume-after-UI-ACK call that re-enters at `reflect`. That silently
        # wiped the UI plan's routing gate mid-flow, which made `_reflect`
        # fall back to the full critic evaluation (scored against the WHOLE
        # original request) instead of the completed-bypass path, and made
        # `_reason`'s routing gate forget which step came next. Only reset to
        # `[]` for a genuinely fresh task; on resume, if the host doesn't
        # supply the list, recover it from the last checkpoint instead of
        # dropping it.
        if required_ui_actions is not None:
            input_state["required_ui_actions"] = required_ui_actions
        elif not resume_with_reflection:
            input_state["required_ui_actions"] = []
        else:
            try:
                prior = self._graph.get_state(config)
                prior_values = prior.values if prior else {}
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    "[run] Không thể đọc checkpoint trước đó để khôi phục "
                    "required_ui_actions (session=%s): %s", session_id, error,
                )
                prior_values = {}
            input_state["required_ui_actions"] = prior_values.get("required_ui_actions", [])
            logger.info(
                "[run] Khôi phục required_ui_actions từ checkpoint (session=%s): %s",
                session_id, input_state["required_ui_actions"],
            )

        return self._graph.invoke(input_state, config=config)

    @staticmethod
    def _initial_node(state: AgentState) -> str:
        """Review a completed asynchronous action before reasoning again."""
        return "reflect" if state.get("resume_with_reflection") else "plan"

    @staticmethod
    def _after_plan(state: AgentState) -> str:
        if state.get("plan_verified") or not state.get("plan"):
            logger.info("[ROUTER: after_plan] → REASON (plan already verified/no plan)")
            return "reason"
        logger.info("[ROUTER: after_plan] → PLAN_REFLECT")
        return "plan_reflect"

    # ── Node: plan ─────────────────────────────────────────────────────────────

    def _plan(self, state: AgentState) -> dict[str, Any]:
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
            {"role": "system", "content": _PLANNER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Yêu cầu hiện tại: {user_msg}\n"
                    f"Phản hồi kiểm tra kế hoạch trước (nếu có): {state.get('plan_feedback', '')}\n"
                    "Nếu đây là yêu cầu giao diện, chỉ đưa vào plan các kết quả giao diện "
                    "mà người dùng yêu cầu; không tự thêm bước chuẩn bị không cần thiết."
                ),
            },
        ]
        print("[AGENT TRACE] ▶ Plan node: đang sinh kế hoạch...", flush=True)
        raw = self._plan_complete(planning_msgs, max(0.1, state["temperature"] - 0.1)).strip()
        print(f"[AGENT TRACE] ── Plan: kế hoạch thô: {raw[:200]}", flush=True)

        plan: list[str] | None = None
        plan_spec: dict[str, Any] = {}
        parsed = False
        try:
            payload = json.loads(raw)
            parsed = isinstance(payload, dict)
            if parsed:
                plan, plan_spec = _normalise_plan_payload(payload)
                if plan:
                    expanded = []
                    for step in plan:
                        expanded.extend(split_step_by_manifest_phrases(step))
                    plan = expanded
        except (ValueError, json.JSONDecodeError):
            plan = None

        if plan:
            steps = list(state["steps"])
            steps.append({"type": "plan", "steps": plan, "spec": plan_spec})
            print(f"[AGENT TRACE] ── Plan: {plan}", flush=True)
            logger.info(f"[NODE: plan] Đã sinh kế hoạch: {plan}")
            return {"plan": plan, "steps": steps,
                    "plan_verified": False,
                    "plan_attempts": state.get("plan_attempts", 0) + 1,
                    "plan_spec": plan_spec,
                    "plan_feedback": ""}

        if not parsed:
            print("[AGENT TRACE] ── Plan: không parse được, tiếp tục không có plan.", flush=True)
            logger.info("[NODE: plan] Không sinh được kế hoạch.")
        elif plan_spec.get("requires_plan"):
            print("[AGENT TRACE] ── Plan: planner yêu cầu kế hoạch nhưng không có bước hợp lệ.", flush=True)
            logger.info("[NODE: plan] Planner không trả về bước kế hoạch hợp lệ.")
        else:
            print("[AGENT TRACE] ── Plan: planner xác nhận không cần kế hoạch.", flush=True)
            logger.info("[NODE: plan] Planner xác nhận tác vụ một bước.")
        return {"plan": None}

    # ── Node: plan_reflect ───────────────────────────────────────────────────

    def _plan_reflect(self, state: AgentState) -> dict[str, Any]:
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
        # phrase (for example, "Reconstruct 3D model từ ảnh đã tải"). Log the
        # hints for observability, but let the semantic Plan Critic decide so
        # valid plans are not rejected by exact-string matching.
        plan_hints = [match_action_intent(step) for step in plan]
        logger.info("[NODE: plan_reflect] Action hints | hints=%s", plan_hints)

        if review is None and self._plan_reflect_complete is not None:
            critic_msgs = [
                {"role": "system", "content": _PLAN_CRITIC_PROMPT},
                {"role": "user", "content": (
                    f"Yêu cầu ban đầu: {user_msg}\n"
                    f"Kế hoạch cần kiểm tra: {json.dumps(state.get('plan_spec') or {'steps': plan}, ensure_ascii=False)}\n"
                    "Kế hoạch có đạt yêu cầu và sẵn sàng thực thi không?"
                )},
            ]
            print("[AGENT TRACE] ▶ Plan Reflect node: LLM đang đánh giá kế hoạch...", flush=True)
            raw = self._plan_reflect_complete(critic_msgs, max(0.1, state["temperature"] - 0.1)).strip()
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

    @staticmethod
    def _after_plan_reflect(state: AgentState) -> str:
        if state.get("plan_verified"):
            logger.info("[ROUTER: after_plan_reflect] → REASON (plan passed)")
            return "reason"
        attempts = state.get("plan_attempts", 0)
        if attempts >= 3:
            logger.warning("[ROUTER: after_plan_reflect] Plan failed after %d attempts; stop safely", attempts)
            return "end"
        logger.info("[ROUTER: after_plan_reflect] → PLAN (regenerate; attempt=%d)", attempts + 1)
        return "plan"

    # ── Node: reason ───────────────────────────────────────────────────────────

    def _reason(self, state: AgentState) -> dict[str, Any]:
        iteration = state["iteration"] + 1
        steps     = list(state["steps"])
        print(f"\n[AGENT TRACE] ── Reason node: iter {iteration}", flush=True)
        logger.info(f"[NODE: reason] Bắt đầu suy luận (iteration {iteration})")

        if self._cancel_checker():
            steps.append({"type": "cancelled", "content": "Task cancelled cooperatively.",
                          "iteration": iteration})
            return {"iteration": iteration, "steps": steps, "done": True, "cancelled": True}

        if iteration > self._max_iterations:
            print(f"[AGENT TRACE] ── Reason: đạt giới hạn ({self._max_iterations} iterations).", flush=True)
            steps.append({"type": "final_answer",
                          "content": "Agent da dat gioi han so vong lap."})
            return {"iteration": iteration, "steps": steps, "done": True}

        messages        = list(state["messages"])
        if messages and messages[0].get("role") == "system":
            original_sys = messages[0]["content"]
            messages[0] = {
                "role": "system",
                "content": _REASONER_PROMPT + original_sys
            }

        plan            = state.get("plan")
        tool_call_count = state.get("tool_call_count", 0)
        done_count      = _completed_plan_steps(steps, plan)

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

        # Nhac nho plan neu co
        if plan:
            # A planned step is complete only after critic/verification accepted
            # it. Failed revisions remain visible as current work.
            remaining      = plan[done_count:] if done_count < len(plan) else []
            if remaining:
                logger.info("[NODE: reason] Plan progress: completed=%d/%d; current step=%s; remaining=%s",
                            done_count, len(plan), remaining[0], remaining)
                print(f"[AGENT TRACE] ── Reason: plan step {done_count + 1}/{len(plan)} → {remaining[0]}", flush=True)
                messages = [*messages, {
                    "role":    "system",
                    "content": f"[Ke hoach con lai] {json.dumps(remaining, ensure_ascii=False)}",
                }]

        # A manifest workflow is only optional context for the reasoner.  It
        # must never replace LLM tool selection or hard-code a case-specific
        # action sequence.
        required_actions = state.get("required_ui_actions", [])
        completed_ui_actions = sum(
            1 for step in steps
            if (step.get("type") == "reflection" and step.get("tool") == "application_action"
                and step.get("result", {}).get("passed") is True)
        )
        expected_action = (required_actions[completed_ui_actions]
                           if completed_ui_actions < len(required_actions) else None)
        plan_action_hint = None
        if plan and done_count < len(plan):
            plan_action_hint = match_action_intent(plan[done_count])
        expected_action = expected_action or plan_action_hint
        current_plan_step = (plan[done_count]
                             if plan and done_count < len(plan) else None)
        plan_total = len(plan or [])
        progress_total = len(required_actions) if required_actions else plan_total
        logger.info("[NODE: reason] selection context | plan_step=%s | expected_tool=%s | expected_action=%s | completed=%d/%d",
                    current_plan_step or "none", "application_action" if expected_action else "model-selected",
                    expected_action.get("action") if expected_action else "none",
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
            print(f"[AGENT TRACE] ── Reason: ToolApp hint step {hint_index}/{progress_total}; gọi LLM tool-calling", flush=True)
            messages = [*messages, {
                "role": "system",
                "content": (
                    "[UI plan hint] Một workflow đã nhận diện action canonical "
                    f"'{expected_action.get('action')}'. Đây chỉ là gợi ý; hãy tự đối chiếu "
                    "với bước hiện tại và không lặp lại tool/action đã bị Reflect đánh giá thất bại."
                ),
            }]

        # Observation summarization
        if tool_call_count >= _OBSERVATION_SUMMARY_THRESHOLD:
            messages = _summarize_messages(messages)
            print(f"[AGENT TRACE] ── Reason: tóm tắt messages (tool_call_count={tool_call_count}).", flush=True)

        print("[AGENT TRACE] ── Reason: đang gọi LLM...", flush=True)
        # logger.debug(f"[NODE: reason] Tin nhắn gửi đến LLM: {json.dumps(messages, ensure_ascii=False)}")
        answer = self._complete(messages, state["temperature"]).strip()
        print(f"[AGENT TRACE] ── Reason: LLM output ({len(answer)} chars): {answer[:120].replace(chr(10), ' ')}", flush=True)
        logger.info(f"[NODE: reason] LLM trả về raw answer: {answer}")

        if answer.casefold().startswith(("context quá dài", "context qua dai", "context too long")):
            compacted = _summarize_messages(state["messages"])
            steps.append({"type": "context_compacted", "iteration": iteration})
            updated = [*compacted, {"role": "system", "content":
                                    "Context vừa được rút gọn. Tiếp tục bằng một JSON tool_call hoặc final khi đủ bằng chứng."}]
            return {"iteration": iteration, "steps": steps, "messages": updated,
                    "tool_call_count": tool_call_count, "done": False}

        if not answer:
            steps.append({"type": "error", "content": "LLM tra ve rong."})
            return {"iteration": iteration, "steps": steps, "done": True}

        tool_name, tool_params = self._parse(answer)
        logger.info("[NODE: reason] selected tool=%s action=%s (model output)",
                    tool_name or "final_answer", (tool_params or {}).get("action", ""))

        if tool_name is None:
            print("[AGENT TRACE] ── Reason: final answer.", flush=True)
            logger.info("[NODE: reason] LLM quyết định dừng (final_answer).")
            coding_missing = coding_status.missing if coding_status else ()
            # Natural-language plan entries describe intent, whereas the Coding
            # workflow has observable completion conditions (source, approved
            # mutation, diff and verification).  Do not force a model to emit
            # arbitrary extra calls merely to consume prose plan entries.
            plan_missing = bool(
                state.get("enforce_plan_completion")
                and not state.get("enforce_coding_workflow")
                and plan and done_count < len(plan)
            )
            if coding_missing or plan_missing:
                # A model may emit a prose answer after a large observation
                # result even though coding evidence or planned work remains.
                # Do not mark the task complete; give it a compact correction turn.
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
                updated_messages.append({
                    "role": "user",
                    "content": (
                        "Coding task chưa đủ bằng chứng để kết thúc. Không được kết thúc hoặc chỉ giải thích. "
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
            citation = _format_code_citation(state.get("messages", []), steps)
            if citation:
                logger.info("[NODE: reason] Dùng formatter trích dẫn code đầy đủ từ read_file result.")
                answer = citation
            steps.append({"type": "final_answer", "content": answer})
            return {"iteration": iteration, "steps": steps, "done": True}

        # ── Validation error: xử lý ngay, không gửi sang tool node ────────
        # Khi _parse trả về _validation_error, nghĩa là LLM đã gọi tool với
        # tham số không hợp lệ. Thay vì chạy qua tool→reflect, ta đưa lỗi
        # trực tiếp vào messages để LLM tự sửa ở lượt suy luận tiếp.
        if tool_name == "_validation_error":
            error_detail = (tool_params or {}).get("error", "Unknown validation error")
            original_tool = (tool_params or {}).get("tool", "unknown")
            logger.warning(
                "[NODE: reason] Validation error cho tool '%s': %s",
                original_tool, error_detail,
            )
            print(
                f"[AGENT TRACE] ── Reason: VALIDATION ERROR cho '{original_tool}': {error_detail}",
                flush=True,
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

        # Extract thinking (text truoc tool_call block)
        think_text = answer
        if "```tool_call" in answer:
            think_text = answer.split("```tool_call")[0].strip()
        elif "{" in answer:
            think_text = answer.split("{")[0].strip()
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
        if self._select_specialist:
            delegation = self._select_specialist(tool_name, params)
            idempotency_key = delegation.get("idempotency_key", "")
            steps.append({"type": "delegation", "agent": delegation.get("specialist", "supervisor"),
                          "tool": tool_name, "idempotency_key": idempotency_key,
                          "iteration": iteration})
        print(f"[AGENT TRACE] ── Reason: tool call → tool='{tool_name}', params={str(params)[:80]}", flush=True)
        logger.info("[NODE: reason] LLM quyết định gọi tool: %s, params: %s", tool_name, params)
        tool_call = {"type": "tool_call", "tool": tool_name,
                      "params": params, "idempotency_key": idempotency_key,
                      "iteration": iteration}
        if current_plan_step is not None:
            tool_call["plan_step_index"] = done_count
        steps.append(tool_call)

        updated_messages = list(state["messages"])
        updated_messages.append({"role": "assistant", "content": answer})
        # Specialist handoffs are audit/step metadata only.  They must not be
        # injected as an internal role into the next model payload: the model
        # receives one unified user/assistant/system conversation contract.

        approval_scope = state.get("approval_scope")
        if not approval_scope:
            scope_payload = json.dumps({"plan": plan, "spec": state.get("plan_spec")},
                                       ensure_ascii=False, sort_keys=True)
            approval_scope = hashlib.sha256(scope_payload.encode()).hexdigest()[:16]
        if self._needs_approval(tool_name) and not state.get("approval_granted"):
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

    @staticmethod
    def _after_reason(state: AgentState) -> str:
        if state["done"] or state["pending_tool"] is not None:
            logger.info("[ROUTER: after_reason] → END (done=%s, pending=%s)", state["done"], state["pending_tool"] is not None)
            print("[AGENT TRACE] ── Router: kết thúc (hoặc chờ phê duyệt)", flush=True)
            return "end"
        # Validation error: quay lại reason để LLM tự sửa (không qua tool)
        steps = state.get("steps", [])
        if steps and steps[-1].get("type") == "validation_error":
            logger.info("[ROUTER: after_reason] → REASON (validation error, self-correct)")
            print("[AGENT TRACE] ── Router: → Reason node (validation error self-correct)", flush=True)
            return "reason"
        if steps and steps[-1].get("type") == "plan_incomplete":
            logger.info("[ROUTER: after_reason] → REASON (premature final answer blocked)")
            print("[AGENT TRACE] ── Router: → Reason node (plan incomplete)", flush=True)
            return "reason"
        if steps and steps[-1].get("type") == "coding_incomplete":
            logger.info("[ROUTER: after_reason] → REASON (coding evidence incomplete)")
            print("[AGENT TRACE] ── Router: → Reason node (coding workflow incomplete)", flush=True)
            return "reason"
        if steps and steps[-1].get("type") == "context_compacted":
            logger.info("[ROUTER: after_reason] → REASON (context compacted)")
            print("[AGENT TRACE] ── Router: → Reason node (context compacted)", flush=True)
            return "reason"
        logger.info("[ROUTER: after_reason] → TOOL")
        print("[AGENT TRACE] ── Router: → Tool node", flush=True)
        return "tool"

    # ── Node: tool ─────────────────────────────────────────────────────────────

    def _tool(self, state: AgentState) -> dict[str, Any]:
        logger.info("[NODE: tool] Bắt đầu thực thi tool.")
        if self._cancel_checker():
            steps = list(state["steps"])
            steps.append({"type": "cancelled", "content": "Task cancelled cooperatively.",
                          "iteration": state.get("iteration", 0)})
            return {"steps": steps, "done": True, "cancelled": True}
        # Tim tool_call chua co tool_result tuong ung
        call_steps   = [s for s in state["steps"] if s.get("type") == "tool_call"]
        result_steps = [s for s in state["steps"] if s.get("type") == "tool_result"]
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

        logger.info("[NODE: tool] Thực thi: %s, params: %s", tool_name, str(params)[:200])
        print(f"\n--- [LOG: TOOL NODE] Thuc thi: {tool_name} ---")
        try:
            result = self._execute(tool_name, params)
            logger.info("[NODE: tool] Kết quả: %s", str(result)[:500])
            print(f"[LOG: TOOL NODE] Ket qua: {result}")
        except Exception as error:  # noqa: BLE001
            result = {"error": f"Tool exception: {error}"}
            logger.error("[NODE: tool] Exception khi thực thi '%s': %s", tool_name, error)
            print(f"[LOG: TOOL NODE] Loi: {result}")

        error_count = state.get("error_count", 0)
        if "error" in result:
            error_count += 1
        else:
            error_count = 0

        # Desktop actions are executed by Qt, outside this process.  Stop the
        # graph until the client explicitly acknowledges the request instead
        # of treating dispatch as success.
        if result.get("pending_ui_ack"):
            return {
                "steps": state.get("steps", []),
                "pending_tool": {"tool": tool_name, "params": params, "ui_ack": True},
                "done": True,
            }

        steps = list(state["steps"])
        steps.append({"type": "tool_result", "tool": tool_name,
                      "result": result, "iteration": state["iteration"]})

        if error_count >= 3:
            steps.append({"type": "final_answer", "content": f"Ngắt mạch (Circuit Breaker): Tool '{tool_name}' gặp lỗi 3 lần liên tiếp. Dừng tác vụ để tránh vòng lặp."})
            return {"steps": steps, "done": True, "error_count": error_count}
        verification = {"passed": "error" not in result, "reason": "No verifier configured."}
        if self._verify_result:
            verification = self._verify_result(tool_name, params, result)
            steps.append({"type": "verification", "tool": tool_name, "result": verification,
                          "iteration": state["iteration"]})

        result_text = json.dumps(result, ensure_ascii=False, indent=2)
        if len(result_text) > 8000:
            result_text = result_text[:8000] + "\n... [truncated]"

        # ── COMPLETION_TOOLS: đánh dấu hoàn thành nhưng vẫn đi qua Reflect ────────
        # Không sinh final_answer tại đây — để Reflect Critic xác nhận rồi Reason mới kết thúc.
        if tool_name in _COMPLETION_TOOLS and result.get("success"):
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

        # ── Tool binh thuong: tiep tuc vong lap ──────────────────────────────
        messages = list(state["messages"])
        messages.append({
            "role":    "user",
            "content": (
                f"Tool `{tool_name}` tra ve:\n```json\n{result_text}\n```\n\n"
                f"Phan tich ket qua va thuc hien buoc tiep theo, "
                f"hoac tra loi cuoi cung neu da du thong tin."
            ),
        })
        if tool_name in _LOW_RISK_TOOLS and verification.get("passed") and "error" not in result:
            # Read-only tools cannot change application state. Record a passed
            # reflection for plan accounting without paying for a critic LLM call.
            steps.append({
                "type": "reflection", "tool": tool_name,
                "result": {"passed": True, "decision": "continue",
                            "reason": "Verified low-risk read-only tool result."},
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
        if (tool_name not in _SEMANTIC_REFLECTION_TOOLS
                and verification.get("passed") and "error" not in result):
            steps.append({
                "type": "reflection", "tool": tool_name,
                "result": {"passed": True, "decision": "continue",
                            "reason": "Verified result contract; semantic critic not required."},
                "iteration": state["iteration"],
            })
            if isinstance(last_step.get("plan_step_index"), int):
                steps[-1]["plan_step_index"] = last_step["plan_step_index"]
            return {"steps": steps, "messages": messages,
                    "tool_call_count": state.get("tool_call_count", 0) + 1,
                    "error_count": error_count, "skip_reflect": True}
        return {
            "steps":           steps,
            "messages":        messages,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "error_count":     error_count,
        }

    @staticmethod
    def _after_tool(state: AgentState) -> str:
        """End after an asynchronous or approval-gated tool result."""
        if state["done"] or state["pending_tool"] is not None:
            logger.info("[ROUTER: after_tool] → END (done=%s, pending=%s)", state["done"], state["pending_tool"] is not None)
            print("--- [LOG: TOOL ROUTER] Ket thuc (cho ACK/phe duyet) ---")
            return "end"
        if (state.get("skip_reflect") and state.get("steps")
                and state["steps"][-1].get("type") == "reflection"):
            logger.info("[ROUTER: after_tool] → REASON (verified low-risk tool)")
            return "reason"
        logger.info("[ROUTER: after_tool] → REFLECT")
        print("--- [LOG: TOOL ROUTER] -> Reflect Node ---")
        return "reflect"

    # ── Node: reflect ────────────────────────────────────────────────────────

    def _reflect(self, state: AgentState) -> dict[str, Any]:
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

        # Xây dựng message cho Critic LLM
        tool_name = call["tool"]
        tool_params = json.dumps(call["params"], ensure_ascii=False)
        tool_result = json.dumps(result_step["result"], ensure_ascii=False)
        if len(tool_result) > 2000:
            tool_result = tool_result[:2000] + "... [truncated]"

        user_msg = next((m["content"] for m in state.get("messages", []) if m.get("role") == "user"), "")
        plan = state.get("plan")
        plan_text = f"Kế hoạch hiện tại: {plan}\n" if plan else ""

        # FIX: scope the critic down to the current plan step. Without this,
        # a critic given the full multi-part `user_msg` will (correctly, but
        # unhelpfully) notice the other steps aren't done yet and mark THIS
        # step as failed, which is what produced the "chỉ tải ảnh 2D nhưng
        # yêu cầu cần cả mô hình 3D và ảnh DICOM" false-negative.
        scope_note = ""
        completed_plan_steps = _completed_plan_steps(state["steps"], plan)
        if plan and completed_plan_steps < len(plan):
            current_step_text = plan[completed_plan_steps]
            scope_note = (
                f"LƯU Ý QUAN TRỌNG: Đây là MỘT bước trong một kế hoạch nhiều bước. "
                f"Bước đang cần đánh giá là: \"{current_step_text}\". "
                f"CHỈ đánh giá xem tool call này có hoàn thành ĐÚNG bước hiện tại hay không. "
                f"KHÔNG đánh giá là thất bại chỉ vì các bước KHÁC trong kế hoạch chưa được thực hiện — "
                f"những bước đó sẽ được xử lý ở các lượt tiếp theo.\n"
            )

        critic_msgs = [
            {"role": "system", "content": _CRITIC_PROMPT},
            {
                "role": "user",
                "content": f"Yêu cầu ban đầu của người dùng: {user_msg}\n{plan_text}{scope_note}Tool đã gọi: {tool_name}\nTham số: {tool_params}\nKết quả: {tool_result}\n\nDựa vào yêu cầu và kết quả này, tool đã gọi có hoàn thành đúng mục tiêu của bước hiện tại không?"
            }
        ]

        print("[AGENT TRACE] ▶ Reflect node: LLM đang đánh giá kết quả tool...", flush=True)
        raw = self._reflect_complete(critic_msgs, max(0.1, state["temperature"] - 0.1)).strip()
        print(f"[AGENT TRACE] ── Reflect node LLM output: {raw[:150]}", flush=True)

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
            # An LLM critic may be optimistic about a syntactically valid but
            # empty search or a failed command.  Deterministic verification is
            # the authority for the tool-result contract.
            reflection = self._reflect_result(call["tool"], call["params"], result_step["result"], verification) \
                if self._reflect_result else {
                    "passed": False, "decision": "revise",
                    "reason": verification.get("reason", "Verification failed."),
                }
        elif not reflection or "passed" not in reflection:
            logger.warning("[NODE: reflect] LLM parse lỗi, dùng verification fallback.")
            if self._reflect_result:
                reflection = self._reflect_result(call["tool"], call["params"], result_step["result"], verification)
            else:
                reflection = {"passed": bool(verification.get("passed")), "decision": "continue",
                              "reason": verification.get("reason", "No critic configured.")}

        # A successful Qt acknowledgement only proves that the selected action
        # ran.  It does not prove that it completed the current plan step. Use
        # the action manifest's phrase data as a generic semantic guard so a
        # critic hallucination cannot accept (for example) a 3D-view action as
        # a 3D-load step. This is data-driven and applies to every UI action.
        if plan and completed_plan_steps < len(plan) and call["tool"] == "application_action":
            current_step_text = plan[completed_plan_steps]
            selected_action = result_step.get("result", {}).get("action") or call.get("params", {}).get("action", "")
            canonical_selected = canonical_action(str(selected_action))
            match = action_matches_plan_step(str(selected_action), current_step_text)
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
                    plan[completed_plan_steps] if plan and completed_plan_steps < len(plan) else "none",
                    reflection.get("passed"), reflection.get("reason", "")[:200])

        return self._record_reflection(state, call, reflection)

    @staticmethod
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

    @staticmethod
    def _after_reflect(state: AgentState) -> str:
        if state["done"] or state["pending_tool"] is not None:
            logger.info("[ROUTER: after_reflect] → END")
            return "end"
        logger.info("[ROUTER: after_reflect] → REASON")
        return "reason"

# ── Observation Summarization ───────────────────────────────────────────────

def _summarize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep a bounded, useful context for the next Reasoner turn.

    The old implementation retained every system note and four complete tool
    messages.  A single large source read could therefore push the request over
    the model context window; the host returned ``Context quá dài`` and the
    graph repeatedly treated that sentinel as a final answer.  This compactor
    preserves the original task and the latest observations while enforcing a
    character budget independent of the number or domain of tools used.
    """
    if len(messages) <= 4 and sum(len(m.get("content", "")) for m in messages) <= _CONTEXT_TOTAL_LIMIT:
        return messages

    def clipped(message: dict[str, str], limit: int) -> dict[str, str]:
        content = message.get("content", "")
        if len(content) <= limit:
            return message
        marker = "\n… [context clipped]"
        return {**message, "content": content[:max(0, limit - len(marker))] + marker}

    first_system_index = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
    first_user_index = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
    recent_count = _OBSERVATION_KEEP_LAST * 2
    recent_indices = list(range(max(0, len(messages) - recent_count), len(messages)))
    protected = set(recent_indices)
    if first_system_index is not None:
        protected.add(first_system_index)
    if first_user_index is not None:
        protected.add(first_user_index)

    middle = [messages[i] for i in range(len(messages)) if i not in protected]
    parts = [
        f"[{m.get('role', '')}] {m.get('content', '')[:300]}"
        for m in middle
    ]
    result: list[dict[str, str]] = []
    if first_system_index is not None:
        result.append(clipped(messages[first_system_index], _CONTEXT_SYSTEM_LIMIT))
    if first_user_index is not None and first_user_index != first_system_index:
        result.append(clipped(messages[first_user_index], _CONTEXT_MESSAGE_LIMIT))
    if parts:
        result.append({
            "role": "system",
            "content": "[Tom tat cac buoc da thuc hien]\n" + "\n".join(parts) + "\n[Het tom tat]",
        })
    for index in recent_indices:
        if index not in {first_system_index, first_user_index}:
            result.append(clipped(messages[index], _CONTEXT_MESSAGE_LIMIT))

    # Preserve the latest observation if the fixed total budget is still tight.
    total = sum(len(m.get("content", "")) for m in result)
    if total > _CONTEXT_TOTAL_LIMIT:
        for index in range(len(result) - 1, -1, -1):
            message = result[index]
            if message.get("role") == "system" and index == 0:
                continue
            excess = total - _CONTEXT_TOTAL_LIMIT
            content = message.get("content", "")
            new_length = max(240, len(content) - excess)
            result[index] = clipped(message, new_length)
            total = sum(len(m.get("content", "")) for m in result)
            if total <= _CONTEXT_TOTAL_LIMIT:
                break
    return result


def _format_code_citation(messages: list[dict[str, str]], steps: list[dict[str, Any]]) -> str | None:
    """Return a complete, syntax-highlighted citation for symbol requests."""
    user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    normalized = user_msg.casefold()
    # A request that merely names a function may ask for analysis, review or a
    # fix.  Only an explicit citation request may replace the model's final
    # explanation with a source fence.
    citation_terms = ("trích dẫn", "trich dan", "cite", "quote", "show code", "source code")
    if not any(term in normalized for term in citation_terms):
        return None
    for step in reversed(steps):
        if step.get("type") != "tool_result" or step.get("tool") != "read_file":
            continue
        result = step.get("result", {})
        content = result.get("content") if isinstance(result, dict) else None
        if not isinstance(content, str) or not content.strip() or result.get("error"):
            continue
        path = str(result.get("path", "source"))
        ext = os.path.splitext(path)[1].lower()
        language = {".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".h": "cpp",
                    ".hpp": "cpp", ".c": "c", ".py": "python"}.get(ext, "text")
        showing = str(result.get("showing", ""))
        source_ref = f"`{path}`" + (f" ({showing})" if showing else "")
        return f"Đoạn mã đầy đủ của hàm được trích dẫn từ {source_ref}:\n\n```{language}\n{content.rstrip()}\n```"
    return None
