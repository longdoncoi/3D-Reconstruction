"""Guard-path coverage for the LLM reflect node.

``orchestration/nodes/reflect.py`` reviews the most recent tool result with a
critic model before the next ReAct turn. The node is a chain of fallbacks —
critic output that is not JSON, not an object, or not the right shape must
degrade to the deterministic verifier rather than crash the run. These tests
pin each rung of that chain, the oversized-result truncation, the plan-scope
semantic override and both exits of the ``after_reflect`` router.
"""
from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

from ai_assistant.orchestration.nodes.reflect import (
    ReflectContext,
    after_reflect,
    reflect_node,
)


def _state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "messages": [{"role": "user", "content": "mở file README"}],
        "steps": [],
        "iteration": 1,
        "temperature": 0.2,
        "done": False,
        "pending_tool": None,
        "plan": None,
    }
    base.update(overrides)
    return base


def _steps(tool: str = "read_file", result: dict | None = None) -> list[dict]:
    return [
        {"type": "tool_call", "tool": tool, "params": {"path": "README.md"}},
        {"type": "tool_result", "tool": tool, "result": result or {"content": "# hi"}},
    ]


def _ctx(raw: str = '{"passed": true, "decision": "continue", "reason": "ok"}',
         reflect_result=None) -> ReflectContext:
    return ReflectContext(reflect_complete=lambda _msgs, _temp: raw,
                          reflect_result=reflect_result)


def _verdict(outcome: dict[str, Any]) -> dict[str, Any]:
    """The critic decision, read from the reflection step it is recorded on."""
    return outcome["steps"][-1]["result"]


class EmptyStepSetTests(unittest.TestCase):
    def test_no_tool_result_means_nothing_to_review(self):
        self.assertEqual(reflect_node(_state(), _ctx()), {})

    def test_a_tool_call_without_its_result_means_nothing_to_review(self):
        state = _state(steps=[{"type": "tool_call", "tool": "read_file", "params": {}}])
        self.assertEqual(reflect_node(state, _ctx()), {})


class CriticOutputValidationTests(unittest.TestCase):
    """Every malformed critic payload must fall back, never raise."""

    def _recorded(self, raw: str, reflect_result=None) -> dict:
        state = _state(steps=_steps())
        return reflect_node(state, _ctx(raw, reflect_result))

    def test_a_non_object_critic_payload_is_discarded(self):
        """A JSON array parses, but it is not a review, so the fallback runs."""
        outcome = self._recorded("[1, 2, 3]")
        self.assertEqual(_verdict(outcome)["passed"], True)
        self.assertEqual(_verdict(outcome)["decision"], "continue")

    def test_a_non_boolean_verdict_is_discarded(self):
        """``"yes"`` is not a verdict — only a real boolean is trusted."""
        outcome = self._recorded('{"passed": "yes", "decision": "continue", "reason": "x"}')
        self.assertTrue(_verdict(outcome)["passed"])

    def test_an_unknown_decision_is_discarded(self):
        outcome = self._recorded('{"passed": true, "decision": "maybe", "reason": "x"}')
        self.assertEqual(_verdict(outcome)["decision"], "continue")

    def test_a_non_string_reason_is_discarded(self):
        outcome = self._recorded('{"passed": true, "decision": "continue", "reason": 42}')
        self.assertEqual(_verdict(outcome)["decision"], "continue")

    def test_prose_from_the_critic_is_discarded(self):
        outcome = self._recorded("I think it worked fine.")
        self.assertEqual(_verdict(outcome)["decision"], "continue")

    def test_a_missing_verdict_key_falls_back_to_the_verifier(self):
        """``"passed" not in reflection`` is its own rejection arm."""
        outcome = self._recorded('{"decision": "continue", "reason": "x"}')
        self.assertEqual(_verdict(outcome)["passed"], True)

    def test_a_configured_reflector_replaces_the_failed_parse(self):
        """When the host supplies its own reflector, that result is used."""
        def _reflect(tool, params, result, verification):
            return {"passed": False, "decision": "revise",
                    "reason": f"host critic rejected {tool}"}

        outcome = self._recorded("not json", reflect_result=_reflect)
        self.assertEqual(_verdict(outcome)["decision"], "revise")
        self.assertIn("read_file", _verdict(outcome)["reason"])


class ResultSizeAndPlanScopeTests(unittest.TestCase):
    def test_an_oversized_result_is_truncated_before_it_reaches_the_critic(self):
        """The critic prompt is size-bounded; a huge payload must not blow it up."""
        captured: list[list[dict]] = []

        def _complete(messages, _temperature):
            captured.append(messages)
            return '{"passed": true, "decision": "continue", "reason": "ok"}'

        state = _state(steps=_steps(result={"content": "x" * 5000}))
        reflect_node(state, ReflectContext(reflect_complete=_complete, reflect_result=None))
        rendered = captured[0][1]["content"]
        self.assertIn("... [truncated]", rendered)
        self.assertLess(len(rendered), 3000)

    def test_a_plan_step_that_does_not_match_the_action_is_forced_to_revise(self):
        """The semantic check outranks a permissive critic verdict."""
        steps = [
            {"type": "tool_call", "tool": "application_action",
             "params": {"action": "open_browser"}},
            {"type": "tool_result", "tool": "application_action",
             "result": {"success": True, "action": "open_browser"}},
        ]
        state = _state(steps=steps, plan=["Đọc file README"])
        with mock.patch(
            "ai_assistant.orchestration.nodes.reflect.step_matches_action",
            return_value=False,
        ):
            outcome = reflect_node(state, _ctx())
        self.assertEqual(_verdict(outcome)["decision"], "revise")
        self.assertIn("does not match current plan step",
                      _verdict(outcome)["reason"])

    def test_a_matching_action_keeps_the_critic_verdict(self):
        steps = [
            {"type": "tool_call", "tool": "application_action",
             "params": {"action": "open_browser"}},
            {"type": "tool_result", "tool": "application_action",
             "result": {"success": True}},
        ]
        state = _state(steps=steps, plan=["Mở trình duyệt"])
        with mock.patch(
            "ai_assistant.orchestration.nodes.reflect.step_matches_action",
            return_value=True,
        ):
            outcome = reflect_node(state, _ctx())
        self.assertEqual(_verdict(outcome)["decision"], "continue")


    def test_a_failed_verification_is_handed_to_the_host_reflector(self):
        """An errored tool result never reaches the critic as a pass."""
        def _reflect(tool, params, result, verification):
            return {"passed": False, "decision": "revise",
                    "reason": f"verifier rejected {tool}"}

        steps = [
            {"type": "tool_call", "tool": "read_file", "params": {}},
            {"type": "tool_result", "tool": "read_file",
             "result": {"error": "File không tồn tại"}},
        ]
        outcome = reflect_node(_state(steps=steps), _ctx(reflect_result=_reflect))
        self.assertEqual(_verdict(outcome)["decision"], "revise")
        self.assertIn("verifier rejected", _verdict(outcome)["reason"])

    def test_a_failed_verification_without_a_host_reflector_becomes_a_revision(self):
        steps = [
            {"type": "tool_call", "tool": "read_file", "params": {}},
            {"type": "tool_result", "tool": "read_file",
             "result": {"error": "File không tồn tại"}},
        ]
        outcome = reflect_node(_state(steps=steps), _ctx())
        self.assertEqual(_verdict(outcome)["passed"], False)
        self.assertEqual(_verdict(outcome)["decision"], "revise")
        # The verifier reports no reason of its own, so the generic one is used.
        self.assertEqual(_verdict(outcome)["reason"], "Verification failed.")


class FailedReviewFeedbackTests(unittest.TestCase):
    def test_a_failed_review_adds_a_recovery_message(self):
        state = _state(steps=_steps())
        outcome = reflect_node(
            state, _ctx('{"passed": false, "decision": "revise", "reason": "no evidence"}'),
        )
        feedback = outcome["messages"][-1]
        self.assertEqual(feedback["role"], "system")
        self.assertIn("Independent review failed", feedback["content"])
        self.assertIn("no evidence", feedback["content"])
        self.assertEqual(_verdict(outcome)["passed"], False)

    def test_a_passing_review_adds_no_feedback(self):
        state = _state(steps=_steps())
        outcome = reflect_node(state, _ctx())
        self.assertEqual(len(outcome["messages"]), 1)

    def test_a_plan_step_index_is_carried_onto_the_reflection_step(self):
        call = {"type": "tool_call", "tool": "read_file", "params": {},
                "plan_step_index": 2}
        state = _state(steps=[call, {"type": "tool_result", "tool": "read_file",
                                     "result": {}}])
        outcome = reflect_node(state, _ctx())
        reflection = next(s for s in outcome["steps"] if s["type"] == "reflection")
        self.assertEqual(reflection["plan_step_index"], 2)


class AfterReflectRouterTests(unittest.TestCase):
    def test_a_finished_run_ends(self):
        self.assertEqual(after_reflect(_state(done=True)), "end")

    def test_a_run_awaiting_tool_approval_ends(self):
        self.assertEqual(after_reflect(_state(pending_tool={"tool": "write"})), "end")

    def test_an_ordinary_run_loops_back_to_reason(self):
        self.assertEqual(after_reflect(_state()), "reason")


if __name__ == "__main__":
    unittest.main()
