"""Unit tests for the reason node's guards and the post-reason router.

``reason_node`` is the ReAct dispatch loop and the one place the engine's
self-correction guards live. Most of its branches are failure paths the happy
suite never reaches, so they were untested: an empty model answer, a
hallucinated inline answer for a step that needs internal context, a malformed
tool call, the forbidden ``transfer_to_toolapp_agent`` bounce, the desktop
step/action mismatch guard (both its reject and its exhausted fallthrough),
and the duplicate-failed-call guard.

Each test drives the node through a fake ``ReasonContext`` whose ``complete``
and ``parse`` callbacks stand in for the model, so the guards are exercised
without an LLM.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_assistant.orchestration.nodes.reason import (  # noqa: E402
    ReasonContext,
    after_reason,
    reason_node,
)

#: A plan step whose execution contract resolves to ``application_action``
#: with ``admin.login`` as the canonical action.
LOGIN_STEP = "Dang nhap vao admin"
#: A plan step whose contract resolves to ``rag_search`` (internal context).
INTERNAL_STEP = "Tim thong tin noi bo ve kien truc du an"


def _state(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "messages": [{"role": "user", "content": "mo ta cong viec"}],
        "steps": [],
        "iteration": 0,
        "temperature": 0.3,
        "done": False,
        "pending_tool": None,
        "plan": None,
        "plan_spec": None,
        "approval_granted": False,
        "granted_fingerprint": "",
        "approval_scope": "",
        "cancelled": False,
        "tool_call_count": 0,
        "last_reflection": None,
        "error_count": 0,
        "resume_with_reflection": False,
        "skip_reflect": False,
        "synthesize_after_rag": False,
        "required_ui_actions": [],
        "plan_verified": False,
        "plan_attempts": 0,
        "plan_feedback": "",
        "enforce_plan_completion": False,
        "enforce_coding_workflow": False,
    }
    base.update(over)
    return base


def _ctx(*, answer: str = "{}", parse: tuple[str | None, dict | None] = (None, None),
         needs_approval=None) -> ReasonContext:
    return ReasonContext(
        complete=lambda _messages, _temperature: answer,
        parse=lambda _text: parse,
        needs_approval=needs_approval or (lambda _name: False),
        max_iterations=5,
        cancel_checker=lambda: False,
        select_specialist=None,
    )


class EmptyAnswerTests(unittest.TestCase):
    def test_an_empty_model_answer_ends_the_run_with_an_error_step(self) -> None:
        out = reason_node(_state(), _ctx(answer=""))
        self.assertTrue(out["done"])
        self.assertEqual(out["steps"][-1]["type"], "error")
        self.assertEqual(out["steps"][-1]["content"], "LLM tra ve rong.")


class EarlyFinalTests(unittest.TestCase):
    def test_fully_answered_plan_is_aggregated_without_calling_the_model(self) -> None:
        """Every plan step answered inline is summed into one final_answer."""
        steps = [
            {"type": "step_answer", "content": "cau tra loi 1"},
            {"type": "reflection", "tool": "step_answer",
             "result": {"passed": True}, "plan_step_index": 0},
        ]
        out = reason_node(_state(plan=["s1"], steps=steps), _ctx(answer="ignored"))
        self.assertTrue(out["done"])
        final = out["steps"][-1]
        self.assertEqual(final["type"], "final_answer")
        self.assertEqual(final["content"], "cau tra loi 1")
        self.assertEqual(out["messages"][-1]["role"], "assistant")


class HallucinatedStepAnswerTests(unittest.TestCase):
    """A step needing internal context may not be answered from thin air."""

    def test_inline_answer_for_an_internal_step_is_blocked(self) -> None:
        out = reason_node(
            _state(plan=[INTERNAL_STEP]),
            _ctx(parse=("_step_answer", {"content": "bịa đặt"})),
        )
        self.assertNotIn("done", out)
        last = out["steps"][-1]
        self.assertEqual(last["type"], "validation_error")
        self.assertIn("requires project-specific context", last["content"])
        # The correction is fed back so the model retries with rag_search.
        self.assertIn("rag_search", out["messages"][-1]["content"])
        self.assertEqual(out["error_count"], 1)


class ValidationErrorResponseTests(unittest.TestCase):
    def test_a_malformed_tool_call_bounces_back_with_the_error_detail(self) -> None:
        out = reason_node(
            _state(),
            _ctx(parse=("_validation_error",
                        {"error": "missing required param", "tool": "run_command"})),
        )
        last = out["steps"][-1]
        self.assertEqual(last["type"], "validation_error")
        self.assertEqual(last["tool"], "run_command")
        self.assertEqual(last["error"], "missing required param")
        self.assertIn("missing required param", out["messages"][-1]["content"])

    def test_defaults_the_tool_name_when_the_parser_omits_it(self) -> None:
        out = reason_node(_state(), _ctx(parse=("_validation_error", {"error": "boom"})))
        self.assertEqual(out["steps"][-1]["tool"], "unknown")


class TransferToToolappAgentGuardTests(unittest.TestCase):
    def test_direct_transfer_calls_are_bounced(self) -> None:
        out = reason_node(_state(), _ctx(parse=("transfer_to_toolapp_agent", {})))
        last = out["steps"][-1]
        self.assertEqual(last["type"], "validation_error")
        self.assertEqual(last["tool"], "transfer_to_toolapp_agent")
        # The corrective message steers the model to call application_action.
        self.assertIn("application_action", out["messages"][-1]["content"])


class StepActionMismatchGuardTests(unittest.TestCase):
    """The right tool with the wrong desktop action is caught pre-dispatch."""

    def _run(self, prior_rejections: int) -> dict[str, Any]:
        prior = [
            {"type": "validation_error", "reason": "step_action_mismatch",
             "plan_step_index": 0, "tool": "application_action"}
            for _ in range(prior_rejections)
        ]
        return reason_node(
            _state(plan=[LOGIN_STEP], steps=prior),
            _ctx(parse=("application_action", {"action": "admin.logout"})),
        )

    def test_a_mismatched_action_is_rejected_with_the_expected_candidates(self) -> None:
        out = self._run(prior_rejections=0)
        last = out["steps"][-1]
        self.assertEqual(last["type"], "validation_error")
        self.assertEqual(last["reason"], "step_action_mismatch")
        self.assertEqual(last["plan_step_index"], 0)
        # The correction lists the canonical actions the manifest ranks.
        self.assertIn("admin.login", out["messages"][-1]["content"])

    def test_the_guard_allows_two_retries_before_exhausting(self) -> None:
        self.assertNotIn("done", self._run(prior_rejections=1))
        self.assertNotIn("pending_tool", self._run(prior_rejections=1))

    def test_an_exhausted_guard_dispatches_instead_of_looping(self) -> None:
        """After two rejections the node stops bouncing and lets Reflect judge."""
        out = self._run(prior_rejections=2)
        # The call is dispatched (a tool_call step is emitted) rather than
        # bounced a third time.
        self.assertEqual(out["steps"][-1]["type"], "tool_call")
        self.assertEqual(out["steps"][-1]["tool"], "application_action")
        self.assertEqual(out["steps"][-1]["params"], {"action": "admin.logout"})

    def test_a_matching_action_is_dispatched_untouched(self) -> None:
        out = reason_node(
            _state(plan=[LOGIN_STEP]),
            _ctx(parse=("application_action", {"action": "admin.login"})),
        )
        self.assertEqual(out["steps"][-1]["type"], "tool_call")
        self.assertEqual(out["steps"][-1]["params"], {"action": "admin.login"})


class DuplicateFailedCallGuardTests(unittest.TestCase):
    def test_repeating_a_tool_call_that_failed_review_is_rejected(self) -> None:
        steps = [
            {"type": "tool_call", "tool": "rag_search", "params": {"query": "q"}},
            {"type": "reflection", "tool": "rag_search", "result": {"passed": False}},
        ]
        out = reason_node(
            _state(steps=steps),
            _ctx(parse=("rag_search", {"query": "q"})),
        )
        last = out["steps"][-1]
        self.assertEqual(last["type"], "validation_error")
        self.assertIn("already failed review", last["error"])
        self.assertIn("Không được lặp lại", out["messages"][-1]["content"])

    def test_changing_the_parameters_makes_the_call_allowed_again(self) -> None:
        steps = [
            {"type": "tool_call", "tool": "rag_search", "params": {"query": "q"}},
            {"type": "reflection", "tool": "rag_search", "result": {"passed": False}},
        ]
        out = reason_node(
            _state(steps=steps),
            _ctx(parse=("rag_search", {"query": "a different query"})),
        )
        self.assertEqual(out["steps"][-1]["type"], "tool_call")


class ThinkingStepTests(unittest.TestCase):
    """The model's reasoning before the tool_call fence is surfaced for the UI."""

    def test_text_before_the_tool_call_fence_is_captured_as_thinking(self) -> None:
        answer = (
            "Truoc het toi se doc file do.\n"
            "```tool_call\n{\"tool\": \"read_file\", \"params\": {\"path\": \"a.txt\"}}\n```"
        )
        out = reason_node(
            _state(),
            _ctx(answer=answer, parse=("read_file", {"path": "a.txt"})),
        )
        types = [step["type"] for step in out["steps"]]
        # The reasoning is recorded, then the call it decided on is dispatched.
        self.assertEqual(types, ["thinking", "tool_call"])
        self.assertEqual(out["steps"][0]["content"], "Truoc het toi se doc file do.")
        self.assertEqual(out["steps"][1]["tool"], "read_file")

    def test_json_without_a_fence_leaves_no_thinking_trail(self) -> None:
        """Pure-JSON output has no prose to surface, so no thinking step."""
        out = reason_node(
            _state(),
            _ctx(answer='{"kind": "final"}', parse=("read_file", {"path": "a.txt"})),
        )
        self.assertEqual([step["type"] for step in out["steps"]], ["tool_call"])


class AfterReasonRouterTests(unittest.TestCase):
    """A guard that asks for another turn routes back to Reason."""

    def _route(self, last_step_type: str) -> str:
        return after_reason(_state(steps=[{"type": last_step_type}]))

    def test_validation_error_routes_back_to_reason(self) -> None:
        self.assertEqual(self._route("validation_error"), "reason")

    def test_plan_incomplete_routes_back_to_reason(self) -> None:
        self.assertEqual(self._route("plan_incomplete"), "reason")

    def test_done_routes_to_end(self) -> None:
        self.assertEqual(after_reason(_state(done=True)), "end")

    def test_pending_approval_routes_to_end(self) -> None:
        self.assertEqual(after_reason(_state(pending_tool={"tool": "t"})), "end")

    def test_a_plain_tool_call_routes_to_the_tool_node(self) -> None:
        self.assertEqual(self._route("tool_call"), "tool")


if __name__ == "__main__":
    unittest.main()
