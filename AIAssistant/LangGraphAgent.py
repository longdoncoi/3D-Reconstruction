"""LangGraph orchestration for the local 3D-Reconstruction agent.

The graph owns the agent loop; the host application owns model inference and
tool execution.  This keeps Qt-specific actions outside the Python process.

Architecture (ReAct + Plan-and-Execute):
  START -> plan -> reason -> tool -> reflect (loop) -> END

  plan   : Sinh ke hoach (danh sach cac buoc) truoc khi bat dau tool loop.
  reason : Quan sat ket qua tool truoc do, quyet dinh buoc tiep theo hoac ket thuc.
  tool   : Thuc thi tool duoc chon.

── FIX LOG (2026-08-27) ─────────────────────────────────────────────────────
Bug: multi-step UI plans (vi du "tai anh 2D + mo hinh 3D + anh DICOM") bi
     ket cung sau buoc dau tien. `viewer.load_2d` chay thanh cong va duoc Qt
     ACK, nhung khi resume vao node `reflect`, Critic LLM van duoc goi va
     cham "failed" vi no duoc dua ca `user_msg` goc (bao gom ca 3 yeu cau)
     thay vi chi bao gom buoc hien tai trong plan. Sau do gate UI-routing bi
     "ket" o `expected_action` cu (vi reflection khong duoc tinh la passed),
     con message phan hoi lai bao LLM "dung lap lai call truoc" -> LLM
     hoang mang va doan sang mot action khac khong hop le
     (`reconstruction.load_3d_model` thay vi `viewer.load_3d`) -> vong lap
     validation_error lap lai den khi cham max_iterations.

Cac fix trong file nay:
  1. `run()`      : khong con ghi de `required_ui_actions` thanh `[]` khi
                     host khong truyen lai tham so nay luc resume — doc lai
                     tu checkpoint truoc do de khong lam mat context UI-plan.
  2. `_reflect()` : nhanh bypass cho `application_action` khop dung
                     `expected_action` khong con phu thuoc vao `verify_result`
                     ben ngoai (co the tra ve passed=False sai) — chi kiem
                     tra truc tiep ket qua tool co loi hay khong.
  3. `_reflect()` : khi van phai goi Critic LLM (khong khop bypass), them
                     "scope_note" cho critic biet CHI danh gia buoc hien tai
                     cua plan, tranh cham fail chi vi cac buoc khac chua lam.
  4. `_record_reflection()`: khi reflection that bai va dang co
                     `expected_action` ro rang, chen thang ten action can
                     goi lai vao thong bao phan hoi, tranh LLM doan sai ten
                     action o vong lap tu-sua tiep theo.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from modules.checkpointing import build_checkpointer
from modules.observability import span

logger = logging.getLogger("langgraph_agent")

# Completion-signal tools: ngay sau khi success -> done=True, khong lap lai
_COMPLETION_TOOLS = {"application_action"}
_LOW_RISK_TOOLS = {
    "read_file", "list_directory", "search_text", "analyze_code",
    "get_project_status", "validate_file", "rag_search",
}
_OBSERVATION_SUMMARY_THRESHOLD = 4   # tong ket observations sau N tool calls
_OBSERVATION_KEEP_LAST = 2           # giu lai N tool-result messages gan nhat

_PLANNER_PROMPT = """Bạn là Planner (Người lập kế hoạch) của hệ thống AI Assistant.
Nhiệm vụ của bạn là phân tích yêu cầu của người dùng và lập ra một kế hoạch ngắn gọn, từng bước một.
KHÔNG sử dụng bất kỳ công cụ (tool) nào.
Trả lời CHÍNH XÁC theo JSON object:
{"requires_plan": true/false, "plan": ["bước 1", "bước 2", ...]}
Đặt requires_plan=false và plan=[] nếu yêu cầu chỉ có một bước độc lập.
Với yêu cầu điều khiển giao diện (mở/tải ảnh, mô hình, DICOM, tái tạo),
hãy lập các bước thực thi trực tiếp trong ứng dụng; không lập bước tìm tài liệu,
tìm mã nguồn, RAG hoặc tìm vị trí tệp."""

_CRITIC_PROMPT = """Bạn là Critic (Người phản biện) của hệ thống AI Assistant.
Nhiệm vụ của bạn là đánh giá xem kết quả thực thi của một tool có thực sự hoàn thành mục tiêu trong yêu cầu ban đầu của người dùng hay không. Nếu tool được gọi hoàn toàn sai mục đích (gọi nhầm tool), hãy đánh giá là thất bại (passed: false).
Hãy trả về ĐÚNG MỘT JSON object (không kèm text) với format:
{"passed": true/false, "decision": "continue"/"revise", "reason": "Lý do chi tiết..."}
- passed: true nếu kết quả trả về hợp lệ, thành công VÀ đúng mục tiêu của người dùng. false nếu có lỗi, sai mục tiêu, hoặc kết quả không đúng mong đợi.
- decision: "continue" nếu có thể đi tiếp, "revise" nếu cần reasoner sửa lỗi hoặc gọi tool khác.
- reason: giải thích ngắn gọn tại sao."""

_REASONER_PROMPT = """Bạn là Reasoner (Người ra quyết định) của hệ thống AI Assistant.
Nhiệm vụ của bạn là dựa vào yêu cầu của người dùng, ngữ cảnh hiện tại và kế hoạch đã đề ra để quyết định bước đi tiếp theo.
Bạn phải tuân thủ nghiêm ngặt các quy tắc và định dạng kết quả (JSON) đã được hướng dẫn.
"""


class AgentState(TypedDict):
    messages:        list[dict[str, str]]   # lich su hoi thoai
    steps:           list[dict[str, Any]]   # danh sach steps (cho UI)
    iteration:       int                    # so vong lap da chay
    temperature:     float                  # nhiet do sinh van ban
    done:            bool                   # da hoan thanh chua
    pending_tool:    dict[str, Any] | None  # tool dang cho phe duyet
    plan:            list[str] | None       # ke hoach cac buoc
    tool_call_count: int                    # so lan goi tool (trigger summarization)
    last_reflection: dict[str, Any] | None  # deterministic critic result
    error_count:     int                    # so lan tool loi lien tiep
    resume_with_reflection: bool            # resume after an asynchronous UI ACK
    skip_reflect:    bool                   # a verified low-risk tool was auto-reviewed
    required_ui_actions: list[dict[str, Any]]  # canonical UI actions required by the host intent matcher


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
                 reflect_complete: Completion | None = None) -> None:
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
        self._emitted_steps = 0

        builder = StateGraph(AgentState)
        builder.add_node("plan",   self._traced("plan", self._plan))
        builder.add_node("reason", self._traced("reason", self._reason))
        builder.add_node("tool",   self._traced("tool", self._tool))
        builder.add_node("reflect", self._traced("reflect", self._reflect))

        # A resumed UI workflow already has an ACK result. Review that result
        # before reasoning again, rather than planning or dispatching another action.
        builder.add_conditional_edges(START, self._initial_node,
                                      {"plan": "plan", "reflect": "reflect"})
        builder.add_edge("plan", "reason")
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
            required_ui_actions: list[dict[str, Any]] | None = None) -> AgentState:
        self._emitted_steps = len(steps or [])
        restored_plan = next(
            (step.get("steps") for step in (steps or []) if step.get("type") == "plan"),
            None,
        )

        config = {"configurable": {"thread_id": session_id}}

        input_state: dict[str, Any] = {
            "messages":        messages,
            "steps":           steps or [],
            "iteration":       iteration,
            "temperature":     temperature,
            "done":            False,
            "pending_tool":    None,
            "plan":            restored_plan,
            "tool_call_count": 0,
            "last_reflection": None,
            "error_count":     0,
            "resume_with_reflection": resume_with_reflection,
            "skip_reflect":    False,
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

    # ── Node: plan ─────────────────────────────────────────────────────────────

    def _plan(self, state: AgentState) -> dict[str, Any]:
        """Sinh ke hoach cac buoc. Skip neu cau ngan (likely UI action)."""
        logger.info("[NODE: plan] Bắt đầu sinh kế hoạch.")
        logger.debug(f"[NODE: plan] State hiện tại (iteration: {state.get('iteration')}): messages={len(state.get('messages', []))} steps={len(state.get('steps', []))}")
        messages = state["messages"]

        for step in state.get("steps", []):
            if step.get("type") == "plan":
                logger.info(f"[NODE: plan] Đã có kế hoạch từ trước: {step.get('steps')}")
                return {"plan": step.get("steps")}

        # Skip planning nếu đang ở giữa task (đã có tool_call từ vòng lặp trước)
        if any(step.get("type") == "tool_call" for step in state.get("steps", [])):
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
                    f"Yêu cầu hiện tại: {user_msg}"
                ),
            },
        ]
        print("[AGENT TRACE] ▶ Plan node: đang sinh kế hoạch...", flush=True)
        raw = self._plan_complete(planning_msgs, max(0.1, state["temperature"] - 0.1)).strip()
        print(f"[AGENT TRACE] ── Plan: kế hoạch thô: {raw[:200]}", flush=True)

        plan: list[str] | None = None
        requires_plan = False
        parsed = False
        try:
            payload = json.loads(raw)
            parsed = isinstance(payload, dict)
            requires_plan = bool(payload.get("requires_plan")) if isinstance(payload, dict) else False
            plan = payload.get("plan") if isinstance(payload, dict) and requires_plan else None
            if not isinstance(plan, list):
                plan = None
        except (ValueError, json.JSONDecodeError):
            plan = None

        if plan:
            steps = list(state["steps"])
            steps.append({"type": "plan", "steps": plan})
            print(f"[AGENT TRACE] ── Plan: {plan}", flush=True)
            logger.info(f"[NODE: plan] Đã sinh kế hoạch: {plan}")
            return {"plan": plan, "steps": steps}

        if not parsed:
            print("[AGENT TRACE] ── Plan: không parse được, tiếp tục không có plan.", flush=True)
            logger.info("[NODE: plan] Không sinh được kế hoạch.")
        elif requires_plan:
            print("[AGENT TRACE] ── Plan: planner yêu cầu kế hoạch nhưng không có bước hợp lệ.", flush=True)
            logger.info("[NODE: plan] Planner không trả về bước kế hoạch hợp lệ.")
        else:
            print("[AGENT TRACE] ── Plan: planner xác nhận không cần kế hoạch.", flush=True)
            logger.info("[NODE: plan] Planner xác nhận tác vụ một bước.")
        return {"plan": None}

    # ── Node: reason ───────────────────────────────────────────────────────────

    def _reason(self, state: AgentState) -> dict[str, Any]:
        iteration = state["iteration"] + 1
        steps     = list(state["steps"])
        print(f"\n[AGENT TRACE] ── Reason node: iter {iteration}", flush=True)
        logger.info(f"[NODE: reason] Bắt đầu suy luận (iteration {iteration})")

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

        # Nhac nho plan neu co
        if plan:
            # A planned step is complete only after critic/verification accepted
            # it. Failed revisions remain visible as current work.
            done_count     = sum(1 for s in steps if s.get("type") == "reflection"
                                 and s.get("result", {}).get("passed") is True)
            remaining      = plan[done_count:] if done_count < len(plan) else []
            if remaining:
                messages = [*messages, {
                    "role":    "system",
                    "content": f"[Ke hoach con lai] {json.dumps(remaining, ensure_ascii=False)}",
                }]

        # Observation summarization
        if tool_call_count >= _OBSERVATION_SUMMARY_THRESHOLD:
            messages = _summarize_messages(messages)
            print(f"[AGENT TRACE] ── Reason: tóm tắt messages (tool_call_count={tool_call_count}).", flush=True)

        print("[AGENT TRACE] ── Reason: đang gọi LLM...", flush=True)
        logger.debug(f"[NODE: reason] Tin nhắn gửi đến LLM: {json.dumps(messages, ensure_ascii=False)}")
        answer = self._complete(messages, state["temperature"]).strip()
        print(f"[AGENT TRACE] ── Reason: LLM output ({len(answer)} chars): {answer[:120].replace(chr(10), ' ')}", flush=True)
        logger.info(f"[NODE: reason] LLM trả về raw answer: {answer}")

        if not answer:
            steps.append({"type": "error", "content": "LLM tra ve rong."})
            return {"iteration": iteration, "steps": steps, "done": True}

        tool_name, tool_params = self._parse(answer)

        if tool_name is None:
            print("[AGENT TRACE] ── Reason: final answer.", flush=True)
            logger.info("[NODE: reason] LLM quyết định dừng (final_answer).")
            steps.append({"type": "final_answer", "content": answer})
            return {"iteration": iteration, "steps": steps, "done": True}

        required_actions = state.get("required_ui_actions", [])
        completed_ui_actions = sum(
            1 for step in steps
            if (step.get("type") == "reflection" and step.get("tool") == "application_action"
                and step.get("result", {}).get("passed") is True)
        )
        expected_action = (required_actions[completed_ui_actions]
                           if completed_ui_actions < len(required_actions) else None)
        if expected_action and (
            tool_name != "application_action"
            or (tool_params or {}).get("action") != expected_action.get("action")
        ):
            expected_id = expected_action["action"]
            logger.warning("[NODE: reason] UI route requires application_action '%s', got '%s'.",
                           expected_id, tool_name)
            steps.append({"type": "validation_error", "tool": tool_name,
                          "error": f"UI request requires application_action: {expected_id}",
                          "iteration": iteration})
            updated_messages = list(state["messages"])
            updated_messages.append({"role": "assistant", "content": answer})
            updated_messages.append({
                "role": "system",
                "content": (
                    "[UI routing override] This is a desktop command. Do NOT call rag_search, "
                    "search_text, read_file, or any research tool. Return exactly one tool call: "
                    f'{{"kind":"tool","tool":"application_action","params":{{"action":"{expected_id}"}}}}.'
                ),
            })
            return {"iteration": iteration, "steps": steps, "messages": updated_messages,
                    "tool_call_count": tool_call_count}

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
                    f"Hãy đọc danh sách action hợp lệ ở trên và gọi lại "
                    f"application_action với đúng tên action. "
                    f"KHÔNG ĐƯỢC trả lời bằng văn bản — chỉ trả về JSON tool_call."
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
        if self._select_specialist:
            delegation = self._select_specialist(tool_name, params)
            steps.append({"type": "delegation", "agent": delegation.get("specialist", "supervisor"),
                          "tool": tool_name, "idempotency_key": delegation.get("idempotency_key", ""),
                          "iteration": iteration})
        print(f"[AGENT TRACE] ── Reason: tool call → tool='{tool_name}', params={str(params)[:80]}", flush=True)
        logger.info("[NODE: reason] LLM quyết định gọi tool: %s, params: %s", tool_name, params)
        steps.append({"type": "tool_call", "tool": tool_name,
                      "params": params, "iteration": iteration})

        updated_messages = list(state["messages"])
        updated_messages.append({"role": "assistant", "content": answer})
        if self._select_specialist and delegation.get("instruction"):
            updated_messages.append({
                "role": "system",
                "content": f"[Specialist handoff: {delegation.get('specialist')}] {delegation['instruction']}",
            })

        if self._needs_approval(tool_name):
            return {
                "iteration":    iteration,
                "steps":        steps,
                "messages":     updated_messages,
                "pending_tool": {"tool": tool_name, "params": params},
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
        logger.info("[ROUTER: after_reason] → TOOL")
        print("[AGENT TRACE] ── Router: → Tool node", flush=True)
        return "tool"

    # ── Node: tool ─────────────────────────────────────────────────────────────

    def _tool(self, state: AgentState) -> dict[str, Any]:
        logger.info("[NODE: tool] Bắt đầu thực thi tool.")
        # Tim tool_call chua co tool_result tuong ung
        call_steps   = [s for s in state["steps"] if s.get("type") == "tool_call"]
        result_steps = [s for s in state["steps"] if s.get("type") == "tool_result"]
        if len(call_steps) > len(result_steps):
            last_step = call_steps[len(result_steps)]
        else:
            last_step = state["steps"][-1]

        tool_name = last_step["tool"]
        params    = last_step["params"]

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

        # Desktop actions are executed by Qt, outside this process.  Stop the
        # graph until the client explicitly acknowledges the request instead
        # of treating dispatch as success.
        if result.get("pending_ui_ack"):
            return {
                "steps": steps,
                "pending_tool": {"tool": tool_name, "params": params, "ui_ack": True},
                "done": True,
            }

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
            return {
                "steps":           steps,
                "messages":        messages,
                "tool_call_count": state.get("tool_call_count", 0) + 1,
                "error_count":     error_count,
                "skip_reflect":    True,
            }
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

        # A UI workflow is composed of separately acknowledged actions.  Once
        # Qt confirms the expected action, that *plan step* is complete even
        # though the overall user request can still have later UI actions.
        # Do not let the semantic critic mark it failed merely for that reason;
        # doing so would keep the route pinned to the already-finished action.
        required_actions = state.get("required_ui_actions", [])
        completed_actions = sum(
            1 for step in state["steps"]
            if (step.get("type") == "reflection" and step.get("tool") == "application_action"
                and step.get("result", {}).get("passed") is True)
        )
        expected_action = (required_actions[completed_actions]
                           if completed_actions < len(required_actions) else None)
        actual_action = result_step.get("result", {}).get("action")

        # FIX: the bypass used to require `verification.get("passed")`, which
        # comes from an externally-configured `_verify_result` callback. If
        # that callback (or a missing `required_ui_actions` list — see the
        # `run()` fix above) ever disagrees for a result that has no actual
        # error, the bypass silently failed to trigger and this step fell
        # through to the full critic call below — scored against the ENTIRE
        # original request instead of just this action. Judge success here
        # directly from the tool result itself, which is the ground truth for
        # "did the desktop action Qt acknowledged actually error out".
        action_succeeded = "error" not in result_step.get("result", {})
        if (call["tool"] == "application_action" and expected_action
                and actual_action == expected_action.get("action")
                and action_succeeded):
            reflection = {
                "passed": True,
                "decision": "continue",
                "reason": f"Qt acknowledged UI step: {actual_action}.",
            }
            return self._record_reflection(state, call, reflection, expected_action)

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
        if plan and completed_actions < len(plan):
            current_step_text = plan[completed_actions]
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

        if not reflection or "passed" not in reflection:
            logger.warning("[NODE: reflect] LLM parse lỗi, dùng verification fallback.")
            if self._reflect_result:
                reflection = self._reflect_result(call["tool"], call["params"], result_step["result"], verification)
            else:
                reflection = {"passed": bool(verification.get("passed")), "decision": "continue",
                              "reason": verification.get("reason", "No critic configured.")}

        logger.info("[NODE: reflect] Kết quả phản ánh: passed=%s, reason=%s",
                    reflection.get("passed"), reflection.get("reason", "")[:200])

        return self._record_reflection(state, call, reflection, expected_action)

    @staticmethod
    def _record_reflection(state: AgentState, call: dict[str, Any],
                           reflection: dict[str, Any],
                           expected_action: dict[str, Any] | None = None) -> dict[str, Any]:
        """Persist critic output and feed a failed review back to the reasoner."""
        steps = list(state["steps"])
        steps.append({"type": "reflection", "tool": call["tool"], "result": reflection,
                      "iteration": state["iteration"]})
        messages = list(state["messages"])
        if not reflection.get("passed"):
            logger.warning("[NODE: reflect] Review FAILED cho tool '%s': %s",
                           call["tool"], reflection.get("reason", ""))
            # FIX: when we already know the exact next required UI action,
            # say so explicitly. Previously the feedback only said "don't
            # repeat the same failing call", which — combined with the
            # UI-routing gate still pointing at the SAME expected action —
            # left the reasoner no valid move and it started guessing at
            # unrelated/invalid action names (e.g. "reconstruction.load_3d_model"
            # instead of the real "viewer.load_3d").
            hint = ""
            if expected_action:
                hint = (
                    f" Bước tiếp theo BẮT BUỘC là gọi application_action với "
                    f"action=\"{expected_action.get('action')}\" — không được đổi sang tool hoặc action khác."
                )
            messages.append({"role": "system", "content": (
                "[Independent review failed] " + str(reflection.get("reason", "Revise the approach.")) +
                " Do not repeat the same failing call; inspect evidence or choose a safer next step." + hint)})
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
    """Rut gon lich su hoi thoai: giu system + first-user-task,
    tom tat cac tool results cu, giu N messages gan nhat.
    """
    if len(messages) <= 4:
        return messages

    system     = [m for m in messages if m.get("role") == "system"]
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    keep_count = _OBSERVATION_KEEP_LAST * 2
    middle     = messages[2:-keep_count] if len(messages) > 2 + keep_count else []
    recent     = messages[-keep_count:]

    if not middle:
        return messages

    parts = []
    for m in middle:
        role    = m.get("role", "")
        content = m.get("content", "")[:300]
        parts.append(f"[{role}] {content}")

    summary_msg = {
        "role":    "system",
        "content": "[Tom tat cac buoc da thuc hien]\n" + "\n".join(parts) + "\n[Het tom tat]",
    }

    result = list(system)
    if first_user:
        result.append(first_user)
    result.append(summary_msg)
    result.extend(recent)
    return result