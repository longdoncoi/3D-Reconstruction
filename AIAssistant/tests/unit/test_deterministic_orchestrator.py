"""Guard-path coverage for the framework-independent agent loop.

``adapters/orchestration/deterministic.py`` is the A2A lifecycle's fallback
engine — deliberately unwired from production, but it is the loop that must
keep working where LangGraph is unavailable (embeddings, CI, contract tests).
These tests drive its durable-continuation guards, its decision parser's four
rejection arms and both input-required exits, none of which the happy-path
platform tests reach.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_assistant.adapters.orchestration.deterministic import DeterministicAgentOrchestrator
from ai_assistant.domain.agents import AgentDecision, AgentDecisionKind
from ai_assistant.domain.security import Principal
from ai_assistant.domain.tasks import AgentTask


class _Result:
    """Stand-in for ``ToolResult``: only ``to_dict`` is read."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def to_dict(self) -> dict:
        return dict(self._payload)


class _FakeTools:
    """Duck-typed ``ToolExecutionService`` with a scripted payload per tool."""

    def __init__(self, payloads: dict[str, dict] | None = None) -> None:
        self.payloads = payloads or {}
        self.grants: list[tuple[str, str]] = []
        self.calls: list[tuple[str, dict, str]] = []

    def issue_approval_grant(self, tool_name, parameters, token) -> None:
        self.grants.append((tool_name, token))

    def execute(self, tool_name, parameters, principal, correlation_id=None, *, approval_token=""):
        self.calls.append((tool_name, dict(parameters), approval_token))
        return _Result(self.payloads.get(tool_name, {"success": True}))


def _task(metadata: dict | None = None, message: str = "do it") -> AgentTask:
    task = AgentTask.new("supervisor", message)
    if metadata:
        task = replace(task, metadata=dict(metadata))
    return task


def _orchestrator(complete, tools=None, max_iterations: int = 8) -> DeterministicAgentOrchestrator:
    return DeterministicAgentOrchestrator(
        complete=complete,
        tools=tools if tools is not None else _FakeTools(),
        tool_descriptions=lambda: "read_file: read a file",
        max_iterations=max_iterations,
    )


class ContinuationGuardTests(unittest.TestCase):
    def test_a_non_mapping_continuation_is_refused(self):
        agent = _orchestrator(lambda _m, _t: '{"kind":"final","content":"x"}')
        result = agent.run(_task({"agent_run.continuation": "not-a-dict", "agent_run.resume": {}}),
                           Principal("user", "id"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["content"], "Task continuation is invalid")

    def test_a_non_mapping_resume_is_refused(self):
        agent = _orchestrator(lambda _m, _t: '{"kind":"final","content":"x"}')
        result = agent.run(_task({"agent_run.continuation": {"messages": []}, "agent_run.resume": 7}),
                           Principal("user", "id"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["content"], "Task continuation is invalid")

    def test_a_malformed_continuation_cannot_be_restored(self):
        """Shape validation lives in ``_restore``; a bad field fails the run."""
        agent = _orchestrator(lambda _m, _t: '{"kind":"final","content":"x"}')
        continuation = {"messages": "not-a-list", "steps": [], "iteration": 1,
                        "pending": {"kind": "approval", "tool": "write", "params": {}, "raw": "r"}}
        result = agent.run(
            _task({"agent_run.continuation": continuation, "agent_run.resume": {}}),
            Principal("user", "id"),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["content"], "Task continuation cannot be restored")


class ApprovalResumeTests(unittest.TestCase):
    @staticmethod
    def _continuation(kind: str = "approval") -> dict:
        return {
            "messages": [{"role": "assistant", "content": "raw"}, {"role": "user", "content": "q"}],
            "steps": [{"type": "tool_call", "tool": "write", "params": {}, "iteration": 1}],
            "iteration": 1,
            "pending": {"kind": kind, "tool": "write", "params": {"path": "a"}, "raw": "raw"},
        }

    def test_a_refused_approval_records_the_rejection_and_ends(self):
        tools = _FakeTools()
        agent = _orchestrator(lambda _m, _t: '{"kind":"final","content":"x"}', tools)
        result = agent.run(
            _task({"agent_run.continuation": self._continuation(),
                   "agent_run.resume": {"approved": False}}),
            Principal("user", "id"),
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["content"], "The requested action was not approved")
        self.assertEqual(result["steps"][-1]["result"], {"rejected": True})
        self.assertEqual(tools.calls, [])          # never executed
        self.assertEqual(tools.grants, [])          # no grant issued

    def test_an_approved_action_is_bound_to_a_single_use_grant(self):
        tools = _FakeTools({"write": {"saved": True}})
        agent = _orchestrator(lambda _m, _t: '{"kind":"final","content":"x"}', tools)
        agent.run(
            _task({"agent_run.continuation": self._continuation(),
                   "agent_run.resume": {"approved": True}}),
            Principal("user", "id"),
        )
        self.assertTrue(tools.grants, "a grant must be issued before execution")
        tool, token = tools.grants[0]
        self.assertEqual(tool, "write")
        self.assertTrue(token.startswith("grant-"))
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(tools.calls[0][2], token)  # the same token is spent

    def test_a_missing_desktop_ack_result_fails_the_resume(self):
        agent = _orchestrator(lambda _m, _t: '{"kind":"final","content":"x"}')
        result = agent.run(
            _task({"agent_run.continuation": self._continuation("desktop_ack"),
                   "agent_run.resume": {"approved": True}}),
            Principal("user", "id"),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["content"], "Desktop acknowledgement result is required")

    def test_a_desktop_ack_result_is_merged_with_its_success_flag(self):
        agent = _orchestrator(lambda _m, _t: '{"kind":"final","content":"done"}')
        result = agent.run(
            _task({"agent_run.continuation": self._continuation("desktop_ack"),
                   "agent_run.resume": {"success": False, "result": {"opened": True}}}),
            Principal("user", "id"),
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps"][-1]["result"], {"success": False, "opened": True})


class DecisionRejectionTests(unittest.TestCase):
    def test_an_unparseable_decision_fails_the_run(self):
        agent = _orchestrator(lambda _m, _t: "this is not json at all")
        result = agent.run(_task(), Principal("user", "id"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["content"], "Model returned an invalid agent decision")

    def test_a_tool_decision_without_a_tool_name_is_refused(self):
        """A subclass may override ``_parse``; the name check is the backstop."""
        agent = _orchestrator(lambda _m, _t: '{"kind":"tool","tool":"read_file","params":{}}')
        agent._parse = lambda _raw: AgentDecision(AgentDecisionKind.TOOL, tool_name=None)
        result = agent.run(_task(), Principal("user", "id"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["content"], "Tool decision did not include a tool name")


class ParseArmTests(unittest.TestCase):
    def _parse(self, raw):
        return DeterministicAgentOrchestrator._parse(raw)

    def test_a_json_fence_is_tolerated(self):
        decision = self._parse('```json\n{"kind":"final","content":"ok"}\n```')
        self.assertIs(decision.kind, AgentDecisionKind.FINAL)
        self.assertEqual(decision.content, "ok")

    def test_a_tool_decision_is_decoded(self):
        decision = self._parse('{"kind":"tool","tool":"read_file","params":{"path":"a"}}')
        self.assertIs(decision.kind, AgentDecisionKind.TOOL)
        self.assertEqual(decision.tool_name, "read_file")
        self.assertEqual(decision.parameters, {"path": "a"})

    def test_malformed_json_is_rejected(self):
        self.assertIsNone(self._parse("{not json"))

    def test_a_json_array_is_rejected(self):
        self.assertIsNone(self._parse("[1, 2, 3]"))

    def test_json_scalar_is_rejected(self):
        self.assertIsNone(self._parse('"just a string"'))

    def test_a_tool_kind_without_a_string_tool_is_rejected(self):
        self.assertIsNone(self._parse('{"kind":"tool","tool":42,"params":{}}'))

    def test_a_tool_kind_without_object_params_is_rejected(self):
        self.assertIsNone(self._parse('{"kind":"tool","tool":"read_file","params":"nope"}'))

    def test_a_final_kind_without_a_string_content_is_rejected(self):
        self.assertIsNone(self._parse('{"kind":"final","content":42}'))


class InputRequiredExitTests(unittest.TestCase):
    def test_a_pending_ui_ack_parks_the_run_for_the_desktop(self):
        tools = _FakeTools({"application_action": {"pending_ui_ack": True, "success": False}})
        agent = _orchestrator(lambda _m, _t: '{"kind":"tool","tool":"application_action","params":{}}', tools)
        result = agent.run(_task(), Principal("user", "id"))
        self.assertEqual(result["status"], "input_required")
        self.assertEqual(result["content"], "Waiting for desktop action acknowledgement")
        pending = result["continuation"]["pending"]
        self.assertEqual(pending["kind"], "desktop_ack")
        self.assertEqual(pending["tool"], "application_action")

    def test_an_approval_gate_parks_the_run_with_its_parameters(self):
        tools = _FakeTools({"write": {"approval_required": True}})
        agent = _orchestrator(lambda _m, _t: '{"kind":"tool","tool":"write","params":{"path":"a"}}', tools)
        result = agent.run(_task(), Principal("user", "id"))
        self.assertEqual(result["status"], "input_required")
        self.assertEqual(result["content"], "Approval is required before this tool can run")
        pending = result["continuation"]["pending"]
        self.assertEqual(pending["kind"], "approval")
        self.assertEqual(pending["params"], {"path": "a"})

    def test_the_iteration_limit_stops_a_run_that_never_answers(self):
        agent = _orchestrator(
            lambda _m, _t: '{"kind":"tool","tool":"read_file","params":{}}',
            _FakeTools(), max_iterations=2,
        )
        result = agent.run(_task(), Principal("user", "id"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["content"], "Agent iteration limit reached")

    def test_the_prompt_embeds_the_registered_tool_descriptions(self):
        seen: list[str] = []

        def _complete(messages, _temperature):
            seen.append(messages[0]["content"])
            return '{"kind":"final","content":"done"}'

        agent = _orchestrator(_complete)
        agent.run(_task(), Principal("user", "id"))
        self.assertIn("read_file: read a file", seen[0])
        self.assertIn("Answer in vi.", seen[0])

    def test_task_metadata_supplies_language_and_temperature(self):
        seen: list[float] = []

        def _complete(_messages, temperature):
            seen.append(temperature)
            return '{"kind":"final","content":"done"}'

        agent = _orchestrator(_complete)
        agent.run(_task({"language": "en", "temperature": "0.9"}), Principal("user", "id"))
        self.assertEqual(seen, [0.9])


if __name__ == "__main__":
    unittest.main()
