"""Guard-path coverage for the tool execution node.

``orchestration/nodes/tool.py`` runs the tool the reasoner chose, verifies the
result and routes onward. The graph contract tests exercise its happy path;
these tests drive the rest of its decision tree — cooperative cancellation, a
missing or malformed pending call, an executor that raises, the three-strike
circuit breaker, the verified-RAG synthesis request, the oversized-result
truncation, the completion and low-risk shortcuts, and every arm of
``after_tool``.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_assistant.orchestration.nodes.tool import ToolContext, after_tool, tool_node


def _state(steps, *, iteration=1, error_count=0, messages=None, plan=None,
           tool_call_count=0, done=False, pending_tool=None, **extra):
    state = {
        "steps": list(steps),
        "iteration": iteration,
        "error_count": error_count,
        "messages": list(messages if messages is not None else [{"role": "user", "content": "q"}]),
        "plan": list(plan or []),
        "tool_call_count": tool_call_count,
        "done": done,
        "pending_tool": pending_tool,
    }
    state.update(extra)
    return state


def _call(tool="read_file", params=None, **extra):
    step = {
        "type": "tool_call",
        "tool": tool,
        "params": {"path": "x"} if params is None else params,
    }
    step.update(extra)
    return step


def _ctx(result=None, *, error=None, cancel=False, verify=None):
    def _execute(tool_name, params):
        if error is not None:
            raise error
        return {"success": True} if result is None else result

    return ToolContext(
        execute=_execute,
        verify_result=verify,
        cancel_checker=lambda: cancel,
    )


class ToolNodeLogRedactionTests(unittest.TestCase):
    def test_a_credential_never_reaches_the_execution_log(self) -> None:
        """Params are logged before execution; they must arrive redacted."""
        params = {"action": "admin.login", "username": "alice", "password": "hunter2-secret"}
        with self.assertLogs("agent.tool", level="DEBUG") as captured:
            tool_node(_state([_call(params=params)]), _ctx())
        joined = "\n".join(captured.output)
        self.assertNotIn("hunter2-secret", joined)
        self.assertIn("[redacted]", joined)

    def test_the_tool_payload_never_reaches_the_execution_log(self) -> None:
        """ADR 0032: the result is logged too, and it may hold the file the
        tool just read. The log states the *shape* of the result, never its
        payload — otherwise the plaintext log becomes a second copy of
        everything the at-rest cipher protects."""
        result = {"success": True, "content": "the private document body"}
        with self.assertLogs("agent.tool", level="DEBUG") as captured:
            tool_node(_state([_call()]), _ctx(result))
        joined = "\n".join(captured.output)
        self.assertNotIn("the private document body", joined)
        self.assertIn("content=<25 chars>", joined)

    def test_the_payload_still_reaches_the_caller(self) -> None:
        """The log is redacted; the *result* is not. The agent must see what
        the tool actually returned."""
        result = {"success": True, "content": "the private document body"}
        with self.assertLogs("agent.tool", level="DEBUG"):
            update = tool_node(_state([_call()]), _ctx(result))
        recorded = [step for step in update["steps"] if step.get("type") == "tool_result"]
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]["result"], result)


class ToolNodeGuardTests(unittest.TestCase):
    def test_cooperative_cancellation_short_circuits(self):
        update = tool_node(_state([_call()], done=False), _ctx(cancel=True))
        self.assertTrue(update["done"])
        self.assertTrue(update["cancelled"])
        self.assertEqual(update["steps"][-1]["type"], "cancelled")

    def test_reaching_the_node_without_a_pending_call_is_an_error(self):
        update = tool_node(_state([]), _ctx())
        self.assertTrue(update["done"])
        self.assertIn("without a pending tool call", update["steps"][-1]["content"])

    def test_an_already_answered_call_does_not_run_twice(self):
        """One tool_call plus one tool_result means nothing is pending."""
        steps = [_call(), {"type": "tool_result", "tool": "read_file", "result": {}}]
        update = tool_node(_state(steps), _ctx())
        self.assertTrue(update["done"])
        self.assertEqual(update["steps"][-1]["type"], "error")

    def test_a_call_without_a_tool_name_is_rejected(self):
        update = tool_node(_state([_call(tool="")]), _ctx())
        self.assertTrue(update["done"])
        self.assertIn("invalid payload", update["steps"][-1]["content"])

    def test_a_call_with_non_mapping_params_is_rejected(self):
        update = tool_node(_state([_call(params="not-a-dict")]), _ctx())
        self.assertTrue(update["done"])
        self.assertIn("invalid payload", update["steps"][-1]["content"])

    def test_an_executor_exception_becomes_an_error_result(self):
        update = tool_node(_state([_call()]), _ctx(error=RuntimeError("boom")))
        self.assertEqual(update["error_count"], 1)
        self.assertIn("Tool exception: boom", update["steps"][-1]["result"]["error"])

    def test_a_successful_call_resets_the_error_streak(self):
        update = tool_node(
            _state([_call()], error_count=2), _ctx(result={"value": 1}),
        )
        self.assertEqual(update["error_count"], 0)

    def test_three_consecutive_failures_trip_the_circuit_breaker(self):
        update = tool_node(
            _state([_call()], error_count=2), _ctx(result={"error": "still broken"}),
        )
        self.assertTrue(update["done"])
        self.assertEqual(update["error_count"], 3)
        final = update["steps"][-1]
        self.assertEqual(final["type"], "final_answer")
        self.assertIn("Circuit Breaker", final["content"])

    def test_the_pending_call_carries_its_plan_step_index(self):
        update = tool_node(
            _state([_call(plan_step_index=1)]), _ctx(result={"value": 2}),
        )
        self.assertEqual(update["steps"][-2]["plan_step_index"], 1)

    def test_verification_result_is_recorded_as_a_step(self):
        def _verify(tool_name, params, result):
            return {"passed": False, "reason": "evidence missing"}

        update = tool_node(_state([_call()]), _ctx(verify=_verify))
        verification = [s for s in update["steps"] if s["type"] == "verification"]
        self.assertEqual(len(verification), 1)
        self.assertFalse(verification[0]["result"]["passed"])

    def test_without_a_verifier_a_clean_result_counts_as_verified(self):
        update = tool_node(_state([_call()]), _ctx(result={"value": 3}))
        self.assertNotIn("verification", [s["type"] for s in update["steps"]])
        # read_file is low-risk and clean, so the semantic critic is skipped.
        self.assertTrue(update["skip_reflect"])

    def test_an_unverified_failing_result_reaches_the_semantic_critic(self):
        update = tool_node(
            _state([_call()]), _ctx(result={"error": "nope"}, verify=lambda *_: {"passed": False, "reason": "bad"}),
        )
        self.assertEqual(update["error_count"], 1)
        self.assertNotIn("skip_reflect", update)

    def test_an_oversized_result_is_truncated_in_the_prompt(self):
        update = tool_node(
            _state([_call("git_diff")]), _ctx(result={"content": "x" * 20000}),
        )
        self.assertIn("... [truncated]", update["messages"][-1]["content"])

    def test_a_compact_result_is_not_truncated(self):
        update = tool_node(_state([_call("git_diff")]), _ctx(result={"content": "short"}))
        self.assertNotIn("... [truncated]", update["messages"][-1]["content"])


class ToolNodeRoutingTests(unittest.TestCase):
    def test_completion_tool_success_skips_the_semantic_critic(self):
        """A dispatched desktop action is announced, not reflected on."""
        update = tool_node(
            _state([_call("application_action", {"action": "open_browser"})]),
            _ctx(result={"success": True, "action": "open_browser"}),
        )
        self.assertIn("đã thực thi thành công", update["messages"][-1]["content"])
        self.assertNotIn("skip_reflect", update)

    def test_low_risk_verified_tool_skips_the_semantic_critic(self):
        update = tool_node(
            _state([_call("read_file")]),
            _ctx(result={"content": "ok"}, verify=lambda *_: {"passed": True, "reason": "fine"}),
        )
        self.assertTrue(update["skip_reflect"])
        self.assertEqual(update["steps"][-1]["type"], "reflection")
        self.assertIn("low-risk", update["steps"][-1]["result"]["reason"])

    def test_a_verified_tool_outside_both_shortcuts_still_gets_a_reflection(self):
        """Neither low-risk nor semantically reviewed: contract check only."""
        update = tool_node(
            _state([_call("some_other_tool")]),
            _ctx(result={"value": 1}, verify=lambda *_: {"passed": True, "reason": "contract"}),
        )
        self.assertTrue(update["skip_reflect"])
        self.assertIn(
            "semantic critic not required", update["steps"][-1]["result"]["reason"],
        )

    def test_a_semantically_reviewed_tool_falls_through_to_reflect(self):
        """run_command needs the critic, so no skip flag and no reflection step."""
        update = tool_node(
            _state([_call("run_command")]),
            _ctx(result={"output": "ok"}, verify=lambda *_: {"passed": True, "reason": "ok"}),
        )
        self.assertNotIn("skip_reflect", update)
        self.assertNotIn("reflection", [s["type"] for s in update["steps"]])


class RagEvidenceTests(unittest.TestCase):
    def test_rag_search_requests_step_answer_synthesis(self):
        update = tool_node(
            _state([_call("rag_search", {"query": "q"})]),
            _ctx(result={"documents": ["d"]}),
        )
        self.assertTrue(update["synthesize_after_rag"])
        self.assertIn("RAG evidence đã được xác minh", update["messages"][-1]["content"])
        self.assertIn("```json", update["messages"][-1]["content"])

    def test_rag_evidence_is_truncated_to_the_evidence_limit(self):
        update = tool_node(
            _state([_call("rag_search", {"query": "q"})]),
            _ctx(result={"documents": ["y" * 20000]}),
        )
        self.assertIn("... [truncated]", update["messages"][-1]["content"])

    def test_rag_directs_the_answer_at_the_current_plan_step(self):
        state = _state(
            [_call("rag_search", {"query": "q"}, plan_step_index=1)],
            plan=["first step", "second step"],
        )
        update = tool_node(state, _ctx(result={"documents": ["d"]}))
        content = update["messages"][-1]["content"]
        self.assertIn('"second step"', content)
        self.assertIn("không lặp lại ý đã nêu", content)

    def test_rag_with_an_out_of_range_plan_step_omits_the_directive(self):
        state = _state(
            [_call("rag_search", {"query": "q"}, plan_step_index=9)],
            plan=["only step"],
        )
        update = tool_node(state, _ctx(result={"documents": ["d"]}))
        content = update["messages"][-1]["content"]
        self.assertNotIn("Trả lời ngắn gọn ĐÚNG bước", content)

    def test_rag_without_a_plan_step_index_omits_the_directive(self):
        update = tool_node(
            _state([_call("rag_search", {"query": "q"})]),
            _ctx(result={"documents": ["d"]}),
        )
        self.assertNotIn("Trả lời ngắn gọn ĐÚNG bước", update["messages"][-1]["content"])


class AfterToolRouterTests(unittest.TestCase):
    def test_done_ends_the_run(self):
        self.assertEqual(after_tool(_state([], done=True)), "end")

    def test_a_parked_action_ends_the_run(self):
        self.assertEqual(
            after_tool(_state([], pending_tool={"tool": "x", "params": {}})), "end",
        )

    def test_verified_rag_evidence_goes_back_to_reason(self):
        self.assertEqual(
            after_tool(_state([], synthesize_after_rag=True)), "reason",
        )

    def test_a_skipped_reflection_goes_back_to_reason(self):
        steps = [{"type": "reflection", "tool": "read_file"}]
        self.assertEqual(after_tool(_state(steps, skip_reflect=True)), "reason")

    def test_a_skip_flag_without_a_reflection_step_still_reflects(self):
        steps = [{"type": "tool_result", "tool": "read_file"}]
        self.assertEqual(after_tool(_state(steps, skip_reflect=True)), "reflect")

    def test_a_regular_result_goes_to_reflect(self):
        self.assertEqual(after_tool(_state([{"type": "tool_result"}])), "reflect")


if __name__ == "__main__":
    unittest.main()
