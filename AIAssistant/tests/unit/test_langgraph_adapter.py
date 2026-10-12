"""Guard-path coverage for the canonical LangGraph adapter.

``adapters/orchestration/langgraph.py`` is the only bridge between the durable
A2A task lifecycle and the production decision loop. Two contracts live here
and neither is policy: metadata normalisation on the way in (language,
temperature, history) and Qt status translation on the way out. Both have
reject arms that no happy-path runner test reaches — a resume payload that is
not a mapping, an approval with no ``action_id``, a desktop acknowledgement
with no bridge or no result, an unknown continuation kind and the cancelled /
unrecognised status arms.
"""
from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any

from ai_assistant.adapters.orchestration.langgraph import LangGraphAgentOrchestrator
from ai_assistant.domain.security import Principal
from ai_assistant.domain.tasks import AgentTask


def _task(metadata: dict[str, Any] | None = None, message: str = "do it") -> AgentTask:
    task = AgentTask.new("supervisor", message)
    return replace(task, metadata=dict(metadata)) if metadata else task


class _Recorder:
    """Captures the request the adapter built and answers with a canned result."""

    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result or {"status": "completed", "steps": [
            {"type": "final_answer", "content": "done"},
        ]}
        self.requests: list[Any] = []

    def __call__(self, request) -> dict[str, Any]:
        self.requests.append(request)
        return self.result


def _orchestrator(execute_result=None, approve_result=None, desktop=None):
    execute = _Recorder(execute_result)
    approve = _Recorder(approve_result)
    orchestrator = LangGraphAgentOrchestrator(
        execute=execute, approve=approve, ui_action_result=desktop,
    )
    return orchestrator, execute, approve


def _principal() -> Principal:
    return Principal("user", "id")


class RequestNormalisationTests(unittest.TestCase):
    """Metadata must arrive at the engine in the shape the engine promises."""

    def test_an_unknown_language_falls_back_to_vietnamese(self):
        orchestrator, execute, _ = _orchestrator()
        orchestrator.run(_task({"language": "fr"}), _principal())
        self.assertEqual(execute.requests[0].language, "vi")

    def test_language_matching_is_case_insensitive(self):
        orchestrator, execute, _ = _orchestrator()
        orchestrator.run(_task({"language": "EN"}), _principal())
        self.assertEqual(execute.requests[0].language, "en")

    def test_a_non_numeric_temperature_falls_back_to_the_default(self):
        orchestrator, execute, _ = _orchestrator()
        orchestrator.run(_task({"temperature": "warm"}), _principal())
        self.assertEqual(execute.requests[0].temperature, 0.2)

    def test_a_non_numeric_temperature_does_not_raise(self):
        """``float(None)`` raises ``TypeError``, which must not escape ``run``."""
        orchestrator, execute, _ = _orchestrator()
        orchestrator.run(_task({"temperature": None}), _principal())
        self.assertEqual(execute.requests[0].temperature, 0.2)

    def test_temperature_is_clamped_to_the_engine_range(self):
        orchestrator, execute, _ = _orchestrator()
        orchestrator.run(_task({"temperature": 99}), _principal())
        orchestrator.run(_task({"temperature": -5}), _principal())
        self.assertEqual([r.temperature for r in execute.requests], [1.5, 0.0])

    def test_history_is_only_forwarded_when_it_is_a_list(self):
        orchestrator, execute, _ = _orchestrator()
        orchestrator.run(_task({"history": "not a list"}), _principal())
        orchestrator.run(_task({"history": [{"role": "user", "content": "hi"}]}), _principal())
        self.assertEqual(execute.requests[0].history, [])
        self.assertEqual(len(execute.requests[1].history), 1)

    def test_the_request_carries_the_task_message_and_id(self):
        orchestrator, execute, _ = _orchestrator()
        task = _task(message="rebuild the index")
        orchestrator.run(task, _principal())
        self.assertEqual(execute.requests[0].task, "rebuild the index")
        self.assertEqual(execute.requests[0].session_id, task.id)


class ApprovalResumeTests(unittest.TestCase):
    CONTINUATION = {"kind": "approval", "action_id": "act-1"}

    def test_a_resume_payload_that_is_not_a_mapping_is_refused(self):
        orchestrator, _, _ = _orchestrator()
        result = orchestrator.run(
            _task({"agent_run.continuation": self.CONTINUATION, "agent_run.resume": "nope"}),
            _principal(),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["content"], "Task continuation is invalid")

    def test_an_approval_without_an_action_id_is_refused(self):
        """``action_id`` is the join key to the pending Qt action."""
        orchestrator, _, approve = _orchestrator()
        result = orchestrator.run(
            _task({"agent_run.continuation": {"kind": "approval"},
                   "agent_run.resume": {}}),
            _principal(),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["content"], "Task continuation cannot be restored")
        self.assertEqual(approve.requests, [])

    def test_an_empty_action_id_is_refused(self):
        orchestrator, _, _ = _orchestrator()
        result = orchestrator.run(
            _task({"agent_run.continuation": {"kind": "approval", "action_id": ""},
                   "agent_run.resume": {}}),
            _principal(),
        )
        self.assertEqual(result["status"], "failed")

    def test_an_approval_forwards_the_users_decision(self):
        """Only an explicit ``True`` reads as consent."""
        orchestrator, _, approve = _orchestrator()
        approve.result = {"status": "completed", "steps": [
            {"type": "final_answer", "content": "granted"},
        ]}
        task = _task({"agent_run.continuation": self.CONTINUATION,
                      "agent_run.resume": {"approved": True}})
        result = orchestrator.run(task, _principal())
        request = approve.requests[0]
        self.assertEqual(request.action_id, "act-1")
        self.assertTrue(request.approved)
        self.assertEqual(request.session_id, task.id)
        self.assertEqual(result["content"], "granted")

    def test_a_refusal_is_forwarded_as_a_non_approval(self):
        orchestrator, _, approve = _orchestrator()
        orchestrator.run(
            _task({"agent_run.continuation": self.CONTINUATION,
                   "agent_run.resume": {"approved": False}}),
            _principal(),
        )
        self.assertFalse(approve.requests[0].approved)

    def test_an_unknown_continuation_kind_is_refused(self):
        orchestrator, _, _ = _orchestrator()
        result = orchestrator.run(
            _task({"agent_run.continuation": {"kind": "teleport"},
                   "agent_run.resume": {}}),
            _principal(),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["content"], "Task continuation cannot be restored")


class DesktopAckResumeTests(unittest.TestCase):
    CONTINUATION = {"kind": "desktop_ack", "request_id": "req-7"}

    def _desktop(self):
        recorder = _Recorder({"status": "completed", "steps": [
            {"type": "final_answer", "content": "opened"},
        ]})
        return LangGraphAgentOrchestrator(
            execute=_Recorder(), approve=_Recorder(), ui_action_result=recorder,
        ), recorder

    def test_a_desktop_ack_without_a_bridge_is_refused(self):
        """An orchestrator built without a desktop callback cannot ack."""
        orchestrator, _, _ = _orchestrator()
        result = orchestrator.run(
            _task({"agent_run.continuation": self.CONTINUATION,
                   "agent_run.resume": {"result": {"opened": True}}}),
            _principal(),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["content"], "Desktop acknowledgement result is required")

    def test_a_desktop_ack_without_a_request_id_is_refused(self):
        orchestrator, recorder = self._desktop()
        result = orchestrator.run(
            _task({"agent_run.continuation": {"kind": "desktop_ack"},
                   "agent_run.resume": {"result": {}}}),
            _principal(),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(recorder.requests, [])

    def test_a_desktop_ack_without_a_result_payload_is_refused(self):
        """``result`` is the observable outcome; without it there is nothing to ack."""
        orchestrator, recorder = self._desktop()
        result = orchestrator.run(
            _task({"agent_run.continuation": self.CONTINUATION,
                   "agent_run.resume": {"result": "not a mapping"}}),
            _principal(),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(recorder.requests, [])

    def test_a_desktop_ack_is_forwarded_with_its_payload(self):
        orchestrator, recorder = self._desktop()
        task = _task({"agent_run.continuation": self.CONTINUATION,
                      "agent_run.resume": {"success": True, "result": {"opened": True}}})
        result = orchestrator.run(task, _principal())
        request = recorder.requests[0]
        self.assertEqual(request.request_id, "req-7")
        self.assertTrue(request.success)
        self.assertEqual(request.result, {"opened": True})
        self.assertEqual(result["content"], "opened")

    def test_a_failed_desktop_action_is_forwarded_as_a_failure(self):
        orchestrator, recorder = self._desktop()
        orchestrator.run(
            _task({"agent_run.continuation": self.CONTINUATION,
                   "agent_run.resume": {"success": False, "result": {"error": "no"}}}),
            _principal(),
        )
        self.assertFalse(recorder.requests[0].success)


class StatusTranslationTests(unittest.TestCase):
    """Qt statuses become A2A statuses; every arm is a distinct user outcome."""

    def _map(self, result):
        orchestrator, _, _ = _orchestrator()
        return orchestrator._map(result)

    def test_a_pending_ui_action_without_a_descriptor_still_parks_the_run(self):
        mapped = self._map({"status": "pending_ui_action", "steps": []})
        self.assertEqual(mapped["status"], "input_required")
        self.assertEqual(mapped["continuation"]["kind"], "desktop_ack")
        self.assertEqual(mapped["continuation"]["request_id"], "")
        self.assertEqual(mapped["continuation"]["tool"], "application_action")
        self.assertEqual(mapped["continuation"]["params"], {})

    def test_a_pending_ui_action_carries_its_descriptor(self):
        mapped = self._map({
            "status": "pending_ui_action",
            "request_id": "req-9",
            "ui_action": {"action": "login", "params": {"username": "u"}},
            "steps": [],
        })
        self.assertEqual(mapped["continuation"]["request_id"], "req-9")
        self.assertEqual(mapped["continuation"]["tool"], "login")
        self.assertEqual(mapped["continuation"]["params"], {"username": "u"})

    def test_a_cancelled_task_is_a_failure(self):
        mapped = self._map({"status": "cancelled", "steps": []})
        self.assertEqual(mapped["status"], "failed")
        self.assertEqual(mapped["content"], "Agent task was cancelled")

    def test_a_rejected_action_reports_the_users_refusal(self):
        mapped = self._map({"status": "rejected", "steps": []})
        self.assertEqual(mapped["status"], "completed")
        self.assertEqual(mapped["content"], "The requested action was not approved")

    def test_a_pending_approval_carries_the_tool_and_its_parameters(self):
        mapped = self._map({
            "status": "pending_approval",
            "action_id": "act-3",
            "steps": [{"type": "pending_approval", "tool": "write_file",
                       "params": {"path": "a.txt"}}],
        })
        self.assertEqual(mapped["status"], "input_required")
        self.assertEqual(mapped["continuation"], {
            "kind": "approval", "action_id": "act-3",
            "tool": "write_file", "params": {"path": "a.txt"},
        })

    def test_a_pending_approval_without_a_marker_step_uses_defaults(self):
        mapped = self._map({"status": "pending_approval", "steps": []})
        self.assertEqual(mapped["continuation"]["tool"], None)
        self.assertEqual(mapped["continuation"]["params"], {})

    def test_an_unrecognised_status_reports_the_engine_error(self):
        mapped = self._map({"status": "exploded", "error": "kernel panic", "steps": []})
        self.assertEqual(mapped["status"], "failed")
        self.assertEqual(mapped["content"], "kernel panic")

    def test_an_unrecognised_status_without_an_error_has_a_generic_message(self):
        mapped = self._map({"status": "exploded", "steps": []})
        self.assertEqual(mapped["content"], "Agent execution failed")

    def test_a_completed_run_without_a_final_answer_reports_empty_content(self):
        mapped = self._map({"status": "completed", "steps": [
            {"type": "tool_call", "tool": "read_file", "params": {}},
        ]})
        self.assertEqual(mapped["status"], "completed")
        self.assertEqual(mapped["content"], "")

    def test_a_final_answer_with_non_text_content_is_not_used(self):
        """Only a string ``content`` is a usable answer for the A2A client."""
        mapped = self._map({"status": "completed", "steps": [
            {"type": "final_answer", "content": {"nested": True}},
        ]})
        self.assertEqual(mapped["content"], "")

    def test_the_latest_final_answer_wins(self):
        mapped = self._map({"status": "completed", "steps": [
            {"type": "final_answer", "content": "first"},
            {"type": "final_answer", "content": "second"},
        ]})
        self.assertEqual(mapped["content"], "second")


if __name__ == "__main__":
    import unittest

    unittest.main()
