"""Guard-path coverage for the reason-context assembly stages.

``prepare_reasoning`` is a fixed pipeline (ADR 0014) whose stages are pure
enough to test directly. Three arms had no coverage: the plan-spec step-kind
lookup (both the indexed hit and the single-kind fallback), the early-final
aggregator's "plan done but nothing was answered inline" exit, and the
pre-call hard-limit guard that compacts an oversized prompt instead of
sending it to the model.
"""
from __future__ import annotations

import unittest
from typing import Any

from ai_assistant.orchestration.nodes.reason_prepare import (
    _compact_before_call,
    _step_kind_for,
    prepare_reasoning,
)


def _state(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "messages": [{"role": "user", "content": "mo ta cong viec"}],
        "steps": [],
        "iteration": 1,
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


class StepKindLookupTests(unittest.TestCase):
    """The plan spec may declare one kind per step for the execution contract."""

    def test_an_indexed_step_returns_its_declared_kind(self):
        self.assertEqual(_step_kind_for(["direct_answer", "rag_search"], "step", 1), "rag_search")

    def test_an_out_of_range_index_yields_no_kind(self):
        self.assertIsNone(_step_kind_for(["direct_answer"], "step", 5))

    def test_a_lonely_kind_applies_when_no_index_is_known(self):
        """A single-kind spec describes the whole request, not a step."""
        self.assertEqual(_step_kind_for(["rag_search"], "request", None), "rag_search")

    def test_several_kinds_without_an_index_yield_no_kind(self):
        self.assertIsNone(_step_kind_for(["a", "b"], "request", None))

    def test_a_missing_or_empty_spec_yields_no_kind(self):
        self.assertIsNone(_step_kind_for(None, "step", 0))
        self.assertIsNone(_step_kind_for([], "step", 0))
        self.assertIsNone(_step_kind_for("rag_search", "step", 0))


class EarlyFinalExitTests(unittest.TestCase):
    def test_a_finished_plan_with_no_inline_answers_is_not_an_early_final(self):
        """All steps done but nothing answered inline: the loop must continue,
        not aggregate an empty answer."""
        plan = ["Lam viec mot"]
        prepared = prepare_reasoning(
            _state(plan=plan), steps=[], plan=plan,
            iteration=1, tool_call_count=0, done_count=1,
        )
        self.assertIsNone(prepared.early_final)

    def test_inline_step_answers_become_the_final_answer(self):
        plan = ["Lam viec mot"]
        steps = [{"type": "step_answer", "content": "xong"}]
        prepared = prepare_reasoning(
            _state(plan=plan), steps=steps, plan=plan,
            iteration=1, tool_call_count=0, done_count=1,
        )
        self.assertIsNotNone(prepared.early_final)
        self.assertTrue(prepared.early_final["done"])


class PreCallCompactTests(unittest.TestCase):
    def test_a_prompt_within_budget_passes_through_untouched(self):
        messages = [{"role": "user", "content": "hi"}]
        self.assertIs(_compact_before_call(messages), messages)

    def test_an_oversized_prompt_is_compacted_before_the_call(self):
        """The hard limit is a safety net independent of the step-1 compact."""
        messages = [{"role": "user", "content": "x" * 15000}]
        with self.assertLogs("agent.reasoning", level="INFO"):
            compacted = _compact_before_call(messages)
        total = sum(len(m.get("content", "")) for m in compacted)
        self.assertLess(total, 15000)
        self.assertTrue(all(m["role"] for m in compacted))


if __name__ == "__main__":
    unittest.main()
