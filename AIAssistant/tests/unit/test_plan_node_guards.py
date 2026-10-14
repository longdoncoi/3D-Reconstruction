"""Guard-path coverage for the plan and plan-reflect nodes.

The plan node is a staircase of early exits — a verified plan is reused, an
unreviewed plan is preserved, a mid-task loop skips planning entirely — and
only the bottom of the staircase (calling the planner) had tests. The
plan-reflect node degrades through its own chain: no plan, a well-formed
critic verdict, garbage from the critic, and the three-strikes stop. Each
rung is pinned here, including the critic's valid-JSON arm that the
happy-path tests never produce.
"""
from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import MagicMock

from ai_assistant.orchestration.nodes.plan import plan_node
from ai_assistant.orchestration.nodes.plan_reflect import (
    after_plan_reflect,
    plan_reflect_node,
)


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


class PlanReuseTests(unittest.TestCase):
    def test_a_verified_plan_is_reused_without_calling_the_planner(self):
        """A passed reflection means the plan step below it is authoritative."""
        planner = MagicMock()
        state = _state(steps=[
            {"type": "plan", "steps": ["Buoc mot", "Buoc hai"], "spec": {"goal": "g"}},
            {"type": "plan_reflection", "result": {"passed": True}},
        ])
        outcome = plan_node(state, planner)
        self.assertEqual(outcome["plan"], ["Buoc mot", "Buoc hai"])
        self.assertEqual(outcome["plan_spec"], {"goal": "g"})
        self.assertTrue(outcome["plan_verified"])
        planner.assert_not_called()

    def test_an_unreviewed_plan_is_preserved(self):
        """Only a *failed* reflection authorises regeneration."""
        planner = MagicMock()
        state = _state(steps=[
            {"type": "plan", "steps": ["Buoc mot"], "spec": {}},
        ])
        outcome = plan_node(state, planner)
        self.assertEqual(outcome["plan"], ["Buoc mot"])
        self.assertNotIn("plan_verified", outcome)
        planner.assert_not_called()

    def test_a_mid_task_loop_skips_planning(self):
        """A tool_call with no reflection means planning already happened."""
        planner = MagicMock()
        state = _state(steps=[{"type": "tool_call", "tool": "read_file", "params": {}}])
        self.assertEqual(plan_node(state, planner), {"plan": None})
        planner.assert_not_called()


class PlannerCallTests(unittest.TestCase):
    def test_a_request_without_a_user_message_still_reaches_the_planner(self):
        """The user-message scan finds nobody, but planning proceeds anyway."""
        planner = MagicMock(return_value=json.dumps({"steps": ["Lam X"], "requires_plan": True}))
        state = _state(messages=[{"role": "system", "content": "context only"}])
        outcome = plan_node(state, planner)
        planner.assert_called_once()
        self.assertEqual(outcome["plan"], ["Lam X"])
        self.assertEqual(outcome["plan_spec"]["steps"], ["Lam X"])

    def test_a_planner_demanding_a_plan_but_giving_no_steps_yields_none(self):
        """``requires_plan`` with no usable steps is a planner failure, not a plan."""
        planner = MagicMock(return_value=json.dumps({"requires_plan": True}))
        outcome = plan_node(_state(), planner)
        self.assertIsNone(outcome["plan"])
        self.assertTrue(outcome["plan_spec"]["requires_plan"])

    def test_unparseable_planner_output_yields_no_plan(self):
        planner = MagicMock(return_value="just prose, no JSON")
        outcome = plan_node(_state(), planner)
        self.assertIsNone(outcome["plan"])

    def test_a_planner_confirming_a_single_step_yields_none_with_a_spec(self):
        """No plan needed, but the spec still guides the reasoner."""
        planner = MagicMock(return_value=json.dumps({"goal": "tra loi nhanh"}))
        outcome = plan_node(_state(), planner)
        self.assertIsNone(outcome["plan"])
        self.assertEqual(outcome["plan_spec"]["goal"], "tra loi nhanh")

    def test_a_valid_json_non_object_payload_yields_no_plan(self):
        """A JSON array parses but is not a plan payload — distinct from garbage."""
        planner = MagicMock(return_value="[1, 2, 3]")
        outcome = plan_node(_state(), planner)
        self.assertIsNone(outcome["plan"])

    def test_a_passed_review_without_a_plan_step_still_replans(self):
        """The verified-reuse loop finds no ``plan`` step, so planning proceeds."""
        planner = MagicMock(return_value="just prose, no JSON")
        state = _state(steps=[
            {"type": "plan_reflection", "result": {"passed": True}},
        ])
        self.assertIsNone(plan_node(state, planner)["plan"])
        planner.assert_called_once()

    def test_a_result_step_without_a_plan_does_not_skip_planning(self):
        planner = MagicMock(return_value="just prose, no JSON")
        state = _state(steps=[
            {"type": "tool_result", "tool": "read_file", "result": {"content": "x"}},
        ])
        self.assertIsNone(plan_node(state, planner)["plan"])
        planner.assert_called_once()


class PlanReflectTests(unittest.TestCase):
    def _run(self, plan, raw: str | None, **over):
        complete = (lambda _msgs, _temp: raw) if raw is not None else None
        return plan_reflect_node(_state(plan=plan, **over), complete)

    def test_an_empty_plan_is_trivially_verified(self):
        self.assertEqual(self._run(None, None), {"plan_verified": True})

    def test_a_plan_without_a_configured_critic_passes_deterministically(self):
        """No critic callback means the deterministic checks are the verdict."""
        outcome = plan_reflect_node(_state(plan=["Buoc mot"]), None)
        self.assertTrue(outcome["plan_verified"])
        verdict = outcome["steps"][-1]["result"]
        self.assertIn("no plan critic configured", verdict["reason"])

    def test_a_valid_json_non_object_critic_reply_forces_a_regeneration(self):
        """A JSON array parses but is not a review — distinct from garbage."""
        outcome = self._run(["Buoc mot"], "[1, 2]")
        self.assertFalse(outcome["plan_verified"])
        self.assertIn("sinh lại", outcome["plan_feedback"])

    def test_a_well_formed_critic_pass_verifies_the_plan(self):
        outcome = self._run(
            ["Buoc mot"],
            json.dumps({"passed": True, "decision": "continue", "reason": "good"}),
        )
        self.assertTrue(outcome["plan_verified"])
        self.assertEqual(outcome["plan_feedback"], "")
        reflection = outcome["steps"][-1]
        self.assertEqual(reflection["type"], "plan_reflection")
        self.assertTrue(reflection["result"]["passed"])

    def test_garbage_from_the_critic_forces_a_regeneration(self):
        outcome = self._run(["Buoc mot"], "not json at all")
        self.assertFalse(outcome["plan_verified"])
        self.assertFalse(outcome["done"])
        self.assertIn("sinh lại", outcome["plan_feedback"])

    def test_a_failing_critic_verdict_requests_changes(self):
        outcome = self._run(
            ["Buoc mot"],
            json.dumps({"passed": False, "decision": "revise", "reason": "too vague"}),
        )
        self.assertFalse(outcome["plan_verified"])
        self.assertEqual(outcome["plan_feedback"], "too vague")

    def test_three_failures_stop_the_run_safely(self):
        """After three attempts the node answers instead of looping forever."""
        outcome = self._run(
            ["Buoc mot"],
            json.dumps({"passed": False, "decision": "revise", "reason": "too vague"}),
            plan_attempts=3,
        )
        self.assertTrue(outcome["done"])
        self.assertEqual(outcome["steps"][-1]["type"], "final_answer")


class AfterPlanReflectTests(unittest.TestCase):
    def test_a_verified_plan_goes_to_reason(self):
        self.assertEqual(after_plan_reflect({"plan_verified": True}), "reason")

    def test_an_unverified_plan_is_regenerated(self):
        self.assertEqual(
            after_plan_reflect({"plan_verified": False, "plan_attempts": 1}), "plan",
        )

    def test_three_attempts_end_the_run(self):
        self.assertEqual(
            after_plan_reflect({"plan_verified": False, "plan_attempts": 3}), "end",
        )


if __name__ == "__main__":
    unittest.main()
