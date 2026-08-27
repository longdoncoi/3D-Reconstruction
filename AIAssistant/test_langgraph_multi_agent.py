import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LangGraphAgent import LocalAgentGraph


class LangGraphMultiAgentTests(unittest.TestCase):
    def test_delegation_and_verification_steps_are_emitted(self):
        replies = iter([
            '{"requires_plan": false, "plan": []}', "CALL",
            '{"passed": true, "decision": "continue", "reason": "ok"}', "DONE",
        ])
        graph = LocalAgentGraph(
            complete=lambda _messages, _temperature: next(replies),
            parse=lambda text: ("run_command", {"command": "echo test"}) if text == "CALL" else (None, None),
            execute=lambda _tool, _params: {"success": True},
            needs_approval=lambda _tool: False,
            max_iterations=3,
            select_specialist=lambda _tool, _params: {"specialist": "research", "idempotency_key": "key"},
            verify_result=lambda _tool, _params, _result: {"passed": True, "reason": "ok"},
        )
        state = graph.run([
            {"role": "system", "content": "test"},
            {"role": "user", "content": "short"},
        ], "multi-agent-test", 0.1)
        self.assertEqual([step["type"] for step in state["steps"]],
                         ["thinking", "delegation", "tool_call", "tool_result", "verification", "reflection", "final_answer"])

    def test_ui_ack_is_reflected_before_reasoning_resumes(self):
        replies = iter(['{"passed": true, "decision": "continue", "reason": "Qt confirmed"}', "DONE"])
        graph = LocalAgentGraph(
            complete=lambda _messages, _temperature: next(replies),
            parse=lambda _text: (None, None),
            execute=lambda _tool, _params: {"success": True},
            needs_approval=lambda _tool: False,
            max_iterations=3,
            reflect_complete=lambda _messages, _temperature: next(replies),
        )
        state = graph.run(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "load"}],
            "ui-ack-test", 0.1,
            steps=[
                {"type": "tool_call", "tool": "application_action", "params": {"action": "viewer.load_2d"}},
                {"type": "tool_result", "tool": "application_action", "result": {"success": True}},
            ],
            resume_with_reflection=True,
        )
        self.assertEqual(state["steps"][-2]["type"], "reflection")
        self.assertEqual(state["steps"][-1]["type"], "final_answer")

    def test_failed_ui_ack_reaches_deterministic_reflection(self):
        replies = iter(["not-json", "DONE"])
        graph = LocalAgentGraph(
            complete=lambda _messages, _temperature: next(replies),
            parse=lambda _text: (None, None),
            execute=lambda _tool, _params: {"success": True},
            needs_approval=lambda _tool: False,
            max_iterations=3,
            reflect_complete=lambda _messages, _temperature: next(replies),
            reflect_result=lambda _tool, _params, _result, verification: {
                "passed": verification["passed"], "decision": "revise",
                "reason": verification["reason"],
            },
        )
        state = graph.run(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "load"}],
            "ui-ack-failure-test", 0.1,
            steps=[
                {"type": "tool_call", "tool": "application_action", "params": {"action": "viewer.load_2d"}},
                {"type": "tool_result", "tool": "application_action", "result": {"success": False}},
                {"type": "verification", "tool": "application_action", "result": {"passed": False, "reason": "Qt failed"}},
            ],
            resume_with_reflection=True,
        )
        reflection = state["steps"][-2]["result"]
        self.assertFalse(reflection["passed"])
        self.assertEqual(reflection["decision"], "revise")

    def test_failed_reflection_does_not_advance_plan_progress(self):
        observed_messages = []

        def complete(messages, _temperature):
            observed_messages.extend(messages)
            return "DONE"

        graph = LocalAgentGraph(
            complete=complete,
            parse=lambda _text: (None, None),
            execute=lambda _tool, _params: {"success": True},
            needs_approval=lambda _tool: False,
            max_iterations=3,
        )
        graph.run(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "task"}],
            "plan-progress-test", 0.1,
            steps=[
                {"type": "plan", "steps": ["step 1", "step 2", "step 3"]},
                {"type": "reflection", "tool": "read_file", "result": {"passed": True}},
                {"type": "reflection", "tool": "read_file", "result": {"passed": False}},
            ],
        )
        self.assertIn('[Ke hoach con lai] ["step 2", "step 3"]',
                      [message["content"] for message in observed_messages])

    def test_verified_low_risk_tool_skips_critic_llm(self):
        replies = iter(["CALL", "DONE"])
        graph = LocalAgentGraph(
            complete=lambda _messages, _temperature: next(replies),
            parse=lambda text: ("read_file", {"path": "README.md"}) if text == "CALL" else (None, None),
            execute=lambda _tool, _params: {"success": True},
            needs_approval=lambda _tool: False,
            max_iterations=3,
            plan_complete=lambda _messages, _temperature: '{"requires_plan": false, "plan": []}',
            reflect_complete=lambda *_args: self.fail("low-risk tool must not call critic LLM"),
            verify_result=lambda _tool, _params, _result: {"passed": True, "reason": "ok"},
        )
        state = graph.run(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "task"}],
            "low-risk-test", 0.1,
        )
        self.assertTrue(any(step.get("type") == "reflection" for step in state["steps"]))

    def test_forced_ui_route_rejects_research_tool_before_execution(self):
        replies = iter(["RAG", "UI"])
        graph = LocalAgentGraph(
            complete=lambda _messages, _temperature: next(replies),
            parse=lambda text: (
                ("rag_search", {"query": "load 2d"}) if text == "RAG"
                else ("application_action", {"action": "viewer.load_2d"})
            ),
            execute=lambda _tool, _params: {
                "pending_ui_ack": True, "action": "viewer.load_2d",
            },
            needs_approval=lambda _tool: False,
            max_iterations=3,
            plan_complete=lambda _messages, _temperature: '{"requires_plan": false, "plan": []}',
        )
        state = graph.run(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "load"}],
            "forced-ui-route-test", 0.1,
            required_ui_actions=[{"action": "viewer.load_2d"}],
        )
        self.assertTrue(any(step.get("type") == "validation_error" for step in state["steps"]))
        self.assertEqual(
            [step["tool"] for step in state["steps"] if step.get("type") == "tool_result"],
            ["application_action"],
        )


if __name__ == "__main__":
    unittest.main()
