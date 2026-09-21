import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LangGraphAgent import LocalAgentGraph, _summarize_messages


class LangGraphMultiAgentTests(unittest.TestCase):
    def test_context_compactor_keeps_task_and_stays_bounded(self):
        messages = [
            {"role": "system", "content": "SYSTEM " + "x" * 9000},
            {"role": "user", "content": "original task"},
        ]
        messages.extend(
            {"role": "user", "content": "observation " + "y" * 5000}
            for _ in range(8)
        )
        compacted = _summarize_messages(messages)
        self.assertLessEqual(sum(len(item["content"]) for item in compacted), 14000)
        self.assertIn("original task", "\n".join(item["content"] for item in compacted))

    def test_context_compaction_returns_to_reasoning_without_a_tool_call(self):
        replies = iter(["Context too long", "DONE"])
        graph = LocalAgentGraph(
            complete=lambda _messages, _temperature: next(replies),
            parse=lambda _text: (None, None),
            execute=lambda _tool, _params: self.fail("tool must not run after context compaction"),
            needs_approval=lambda _tool: False,
            max_iterations=3,
            plan_complete=lambda _messages, _temperature: '{"requires_plan": false, "plan": []}',
        )
        state = graph.run(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "task"}],
            "context-compaction-routing-test", 0.1,
        )
        self.assertEqual(
            [step["type"] for step in state["steps"]],
            ["context_compacted", "final_answer"],
        )

    def test_large_rag_observation_is_compacted_before_next_reason_turn(self):
        context_sizes = []
        replies = iter(["SEARCH", "DONE"])

        def complete(messages, _temperature):
            context_sizes.append(sum(len(message.get("content", "")) for message in messages))
            return next(replies)

        graph = LocalAgentGraph(
            complete=complete,
            parse=lambda text: ("rag_search", {"query": "engineers"}) if text == "SEARCH" else (None, None),
            execute=lambda _tool, _params: {"found": True, "results": [{"content": "e" * 6000}]},
            needs_approval=lambda _tool: False,
            max_iterations=3,
            plan_complete=lambda _messages, _temperature: '{"requires_plan": false, "plan": []}',
        )
        state = graph.run(
            [{"role": "system", "content": "s" * 10000},
             {"role": "user", "content": "What evidence is available?"}],
            "rag-context-budget-test", 0.1,
        )

        self.assertEqual(state["steps"][-1]["type"], "final_answer")
        self.assertGreaterEqual(len(context_sizes), 2)
        self.assertLessEqual(context_sizes[-1], 14000)

    def test_project_information_is_synthesized_after_rag_search(self):
        calls = []

        def execute(tool, params):
            calls.append((tool, params))
            return {"found": True, "results": [{
                "content": "=== TÀI LIỆU THAM KHẢO ===\n[1] people.txt\nEngineer evidence"
            }]}

        def complete(messages, _temperature):
            prompt = messages[-1]["content"]
            self.assertIn("Engineer evidence", prompt)
            self.assertIn("Do not mention, quote, or reproduce source headings", prompt)
            return "=== TÀI LIỆU THAM KHẢO ===\n[1] people.txt\nNgười này là kỹ sư trong dự án. [1]"

        graph = LocalAgentGraph(
            complete=complete,
            parse=lambda _text: (None, None),
            execute=execute,
            needs_approval=lambda _tool: False,
            max_iterations=3,
            plan_complete=lambda *_args: self.fail("Planner must not run before project RAG search"),
        )
        state = graph.run(
            [{"role": "system", "content": "test"},
             {"role": "user", "content": "K\u1ef9 s\u01b0 trong d\u1ef1 \u00e1n l\u00e0 ai?"}],
            "project-information-rag-test", 0.1,
        )

        self.assertEqual(calls, [("rag_search", {"query": "K\u1ef9 s\u01b0 trong d\u1ef1 \u00e1n l\u00e0 ai?", "top_k": 5})])
        self.assertEqual(state["steps"][-1]["content"], "Người này là kỹ sư trong dự án.")
        self.assertNotIn("TÀI LIỆU THAM KHẢO", state["steps"][-1]["content"])

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

    def test_reflect_failure_returns_reason_for_alternate_tool_call(self):
        reason_replies = iter(["WRONG", "UI"])
        graph = LocalAgentGraph(
            complete=lambda _messages, _temperature: next(reason_replies),
            parse=lambda text: (
                ("run_command", {"command": "echo wrong"}) if text == "WRONG"
                else ("application_action", {"action": "viewer.load_2d"})
            ),
            execute=lambda tool, params: ({"success": True} if tool == "run_command"
                                           else {"pending_ui_ack": True, "action": params["action"]}),
            needs_approval=lambda _tool: False,
            max_iterations=3,
            plan_complete=lambda _messages, _temperature: '{"requires_plan": false, "plan": []}',
            reflect_complete=lambda _messages, _temperature: '{"passed": false, "decision": "revise", "reason": "wrong tool for current step"}',
        )
        state = graph.run(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "load"}],
            "reflect-reason-retry-test", 0.1,
            required_ui_actions=[{"action": "viewer.load_2d"}],
        )
        # A semantic failure is handled by Reflect, then the next Reason turn
        # receives the failure context and selects a new tool call.
        self.assertEqual(
            [step["tool"] for step in state["steps"] if step.get("type") == "tool_call"],
            ["run_command", "application_action"],
        )
        self.assertEqual(
            [step["tool"] for step in state["steps"] if step.get("type") == "tool_result"],
            ["run_command"],
        )
        self.assertTrue(any(
            step.get("type") == "reflection" and step.get("result", {}).get("passed") is False
            for step in state["steps"]
        ))

    def test_failed_second_plan_step_is_reselected_by_reason(self):
        reason_replies = iter(["STEP1", "WRONG_STEP2", "RIGHT_STEP2", "DONE"])
        reflect_replies = iter([
            '{"passed": true, "decision": "continue", "reason": "step 1 complete"}',
            '{"passed": false, "decision": "revise", "reason": "wrong tool for step 2"}',
            '{"passed": true, "decision": "continue", "reason": "step 2 complete"}',
        ])
        graph = LocalAgentGraph(
            complete=lambda _messages, _temperature: next(reason_replies),
            parse=lambda text: (None, None) if text == "DONE" else ("run_command", {"command": text}),
            execute=lambda _tool, params: {"success": True, "command": params["command"]},
            needs_approval=lambda _tool: False,
            max_iterations=6,
            plan_complete=lambda _messages, _temperature: '{"requires_plan": true, "plan": ["step 1", "step 2", "step 3"]}',
            reflect_complete=lambda _messages, _temperature: next(reflect_replies),
            verify_result=lambda _tool, _params, _result: {"passed": True, "reason": "structured result"},
        )
        state = graph.run(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "multi-step task"}],
            "plan-step-retry-test", 0.1,
        )
        calls = [step["params"]["command"] for step in state["steps"] if step.get("type") == "tool_call"]
        self.assertEqual(calls, ["STEP1", "WRONG_STEP2", "RIGHT_STEP2"])
        reflections = [step["result"] for step in state["steps"] if step.get("type") == "reflection"]
        self.assertEqual([item["passed"] for item in reflections], [True, False, True])
        self.assertTrue(any(
            message.get("content", "").startswith("[Independent review failed]")
            for message in state["messages"]
        ))

    def test_reflect_rejects_successful_but_unrelated_ui_action(self):
        critic_calls = []
        graph = LocalAgentGraph(
            complete=lambda _messages, _temperature: "DONE",
            parse=lambda _text: (None, None),
            execute=lambda _tool, _params: {"success": True},
            needs_approval=lambda _tool: False,
            max_iterations=3,
            reflect_complete=lambda messages, _temperature: critic_calls.append(messages) or (
                '{"passed": true, "decision": "continue", "reason": "looks successful"}'
            ),
            verify_result=lambda _tool, _params, _result: {"passed": True, "reason": "Qt acknowledged"},
        )
        state = graph.run(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "load 2d and 3d"}],
            "semantic-reflect-test", 0.1,
            steps=[
                {"type": "plan", "steps": ["Tải ảnh 2D", "Tải mô hình 3D"]},
                {"type": "reflection", "tool": "application_action", "result": {"passed": True}},
                {"type": "tool_call", "tool": "application_action", "params": {"action": "reconstruction.view_3d_model"}},
                {"type": "tool_result", "tool": "application_action", "result": {"success": True, "action": "reconstruction.view_3d_model"}},
                {"type": "verification", "tool": "application_action", "result": {"passed": True, "reason": "Qt acknowledged"}},
            ],
            resume_with_reflection=True,
        )
        reflection = next(step["result"] for step in reversed(state["steps"])
                          if step.get("type") == "reflection")
        self.assertFalse(reflection["passed"])
        self.assertIn("Tải mô hình 3D", reflection["reason"])
        self.assertTrue(critic_calls)

    def test_ui_sequence_dispatches_next_canonical_action(self):
        replies = iter(["CALL_2D"])
        graph = LocalAgentGraph(
            complete=lambda _messages, _temperature: next(replies),
            parse=lambda _text: ("application_action", {"action": "viewer.load_2d"}),
            execute=lambda _tool, params: {"pending_ui_ack": True, "action": params["action"]},
            needs_approval=lambda _tool: False,
            max_iterations=3,
            plan_complete=lambda _messages, _temperature: '{"requires_plan": true, "plan": ["Tải ảnh 2D", "Tải mô hình 3D", "Tải ảnh DICOM"]}',
        )
        state = graph.run(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "load all"}],
            "ui-sequence-test", 0.1,
            required_ui_actions=[
                {"action": "viewer.load_2d"},
                {"action": "viewer.load_3d"},
                {"action": "viewer.load_dicom"},
            ],
        )
        self.assertEqual(
            [step["params"]["action"] for step in state["steps"] if step.get("type") == "tool_call"],
            ["viewer.load_2d"],
        )

    def test_plan_reflect_regenerates_unexecutable_ui_plan(self):
        plans = iter([
            '{"requires_plan": true, "plan": ["open app and find storage", "tai 2d", "tai 3d", "tai dicom"]}',
            '{"requires_plan": true, "plan": ["tai 2d", "tai 3d", "tai dicom"]}',
        ])
        reviews = iter([
            '{"passed": false, "decision": "revise", "reason": "first step is not required"}',
            '{"passed": true, "decision": "continue", "reason": "concise"}',
        ])
        graph = LocalAgentGraph(
            complete=lambda _messages, _temperature: "CALL",
            parse=lambda _text: ("application_action", {"action": "viewer.load_2d"}),
            execute=lambda _tool, params: {"pending_ui_ack": True, "action": params["action"]},
            needs_approval=lambda _tool: False,
            max_iterations=2,
            plan_complete=lambda _messages, _temperature: next(plans),
            plan_reflect_complete=lambda _messages, _temperature: next(reviews),
        )
        state = graph.run(
            [{"role": "system", "content": "test"},
             {"role": "user", "content": "tai 2d tai 3d tai dicom"}],
            "plan-reflect-regeneration-test", 0.1,
        )
        reviews = [step for step in state["steps"] if step.get("type") == "plan_reflection"]
        self.assertEqual([review["result"]["passed"] for review in reviews], [False, True])
        self.assertEqual(state["plan"], ["tai 2d", "tai 3d", "tai dicom"])

    def test_code_citation_uses_complete_markdown_fence(self):
        replies = iter(["READ", "DONE"])
        source = "void ReconstructionPlugin::onRunReconstruction() {\n    return;\n}"
        graph = LocalAgentGraph(
            complete=lambda _messages, _temperature: next(replies),
            parse=lambda text: ("read_file", {"path": "src/ReconstructionPlugin.cpp", "symbol": "onRunReconstruction"})
            if text == "READ" else (None, None),
            execute=lambda _tool, _params: {
                "path": "src/ReconstructionPlugin.cpp", "showing": "lines 10-12", "content": source,
            },
            needs_approval=lambda _tool: False,
            max_iterations=3,
            plan_complete=lambda _messages, _temperature: '{"requires_plan": false, "plan": []}',
        )
        state = graph.run(
            [{"role": "system", "content": "test"},
             {"role": "user", "content": "Trích dẫn hàm onRunReconstruction"}],
            "code-citation-format-test", 0.1,
        )
        answer = state["steps"][-1]["content"]
        self.assertIn("```cpp", answer)
        self.assertIn("return;", answer)
        self.assertTrue(answer.rstrip().endswith("```"))

    def test_coding_task_cannot_finish_after_discovery_only(self):
        graph = LocalAgentGraph(
            complete=lambda _messages, _temperature: "DONE",
            parse=lambda _text: (None, None),
            execute=lambda _tool, _params: {"success": True},
            needs_approval=lambda _tool: False,
            max_iterations=1,
            plan_complete=lambda _messages, _temperature: '{"requires_plan": false, "plan": []}',
        )
        state = graph.run(
            [{"role": "system", "content": "test"},
             {"role": "user", "content": "Implement a feature in the project"}],
            "coding-completion-gate-test", 0.1,
            enforce_plan_completion=True,
        )
        self.assertTrue(any(step.get("type") == "coding_incomplete"
                            for step in state["steps"]))
        self.assertFalse(any(step.get("type") == "final_answer" and step.get("content") == "DONE"
                             for step in state["steps"]))

    def test_wrong_desktop_action_is_rejected_before_dispatch(self):
        replies = iter(["WRONG", "RIGHT"])
        dispatched = []

        def parse(text):
            action = "reconstruction.view_3d_model" if text == "WRONG" else "reconstruction.close_3d_model"
            return "application_action", {"action": action}

        def execute(_tool, params):
            dispatched.append(params["action"])
            return {"pending_ui_ack": True, "action": params["action"]}

        graph = LocalAgentGraph(
            complete=lambda *_: next(replies), parse=parse, execute=execute,
            needs_approval=lambda _tool: False, max_iterations=4,
            plan_complete=lambda *_: '{"requires_plan": true, "plan": ["Ẩn mô hình 3D"]}',
        )
        state = graph.run([{"role": "system", "content": "test"},
                           {"role": "user", "content": "ẩn mô hình 3D"}], "step-guard-test", 0.1)
        self.assertEqual(dispatched, ["reconstruction.close_3d_model"])
        self.assertTrue(any(s.get("reason") == "step_action_mismatch" for s in state["steps"]))

if __name__ == "__main__":
    unittest.main()
