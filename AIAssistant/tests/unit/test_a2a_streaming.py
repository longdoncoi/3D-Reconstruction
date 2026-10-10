"""Tests for A2A streaming edge cases and payload safety.

Covers SSE streaming, task payload serialization, and sensitive data redaction.
"""
from __future__ import annotations

import unittest
from typing import Any

from ai_assistant.adapters.a2a_payloads import _safe_value, task_payload
from ai_assistant.domain.security import DataClassification
from ai_assistant.domain.tasks import AgentTask, TaskStatus


class A2AStreamingTests(unittest.TestCase):
    """Test A2A streaming and payload construction edge cases."""

    def test_safe_value_redacts_sensitive_keys(self) -> None:
        """Sensitive keys must be redacted in A2A artifacts."""
        data = {
            "api_key": "secret123",
            "password": "hunter2",
            "token": "abc",
            "authorization": "Bearer xyz",
            "nested": {
                "secret": "hidden",
                "continuation": {"kind": "approval"},
            },
            "normal": "visible",
        }
        result = _safe_value(data)
        self.assertEqual(result["api_key"], "[redacted]")
        self.assertEqual(result["password"], "[redacted]")
        self.assertEqual(result["token"], "[redacted]")
        self.assertEqual(result["authorization"], "[redacted]")
        self.assertEqual(result["nested"]["secret"], "[redacted]")
        self.assertEqual(result["nested"]["continuation"], "[redacted]")
        self.assertEqual(result["normal"], "visible")

    def test_safe_value_handles_lists(self) -> None:
        """Lists containing sensitive data must be redacted."""
        data = [{"api_key": "secret"}, {"normal": "visible"}]
        result = _safe_value(data)
        self.assertEqual(result[0]["api_key"], "[redacted]")
        self.assertEqual(result[1]["normal"], "visible")

    def test_safe_value_handles_empty_structures(self) -> None:
        """Empty dicts and lists pass through unchanged."""
        self.assertEqual(_safe_value({}), {})
        self.assertEqual(_safe_value([]), [])

    def test_safe_value_handles_primitives(self) -> None:
        """Primitive values pass through unchanged."""
        self.assertEqual(_safe_value("hello"), "hello")
        self.assertEqual(_safe_value(42), 42)
        self.assertEqual(_safe_value(None), None)
        self.assertEqual(_safe_value(True), True)


class TaskPayloadTests(unittest.TestCase):
    """Test task_payload serialization for various task states."""

    def _make_task(self, **kwargs: Any) -> AgentTask:
        defaults: dict[str, Any] = {
            "id": "task-1",
            "message": "Test task",
            "capability": "supervisor",
            "status": TaskStatus.WORKING,
            "classification": DataClassification.INTERNAL,
            "context_id": "ctx-1",
            "metadata": {},
            "result": None,
            "error": None,
            "created_at": 1700000000.0,
            "updated_at": 1700000001.0,
        }
        defaults.update(kwargs)
        return AgentTask(**defaults)

    def test_task_payload_completed_with_result(self) -> None:
        """Completed tasks include artifacts with result data."""
        task = self._make_task(
            status=TaskStatus.COMPLETED,
            result={"content": "Done!", "steps": []},
        )
        payload = task_payload(task)
        self.assertEqual(payload["id"], "task-1")
        self.assertEqual(payload["status"]["state"], "completed")
        self.assertEqual(len(payload["artifacts"]), 1)
        self.assertEqual(payload["artifacts"][0]["artifactId"], "result-task-1")

    def test_task_payload_failed_with_error(self) -> None:
        """Failed tasks include error message in status."""
        task = self._make_task(
            status=TaskStatus.FAILED,
            error="Something went wrong",
        )
        payload = task_payload(task)
        self.assertEqual(payload["status"]["state"], "failed")
        self.assertIn("message", payload["status"])
        self.assertIn("Something went wrong", payload["status"]["message"]["parts"][0]["text"])

    def test_task_payload_input_required(self) -> None:
        """Input-required tasks include continuation message."""
        task = self._make_task(
            status=TaskStatus.INPUT_REQUIRED,
            result={"content": "Approval needed"},
        )
        payload = task_payload(task)
        self.assertEqual(payload["status"]["state"], "input-required")
        self.assertIn("message", payload["status"])

    def test_task_payload_redacts_agent_run_metadata(self) -> None:
        """Internal agent_run.* metadata must not leak into A2A artifacts."""
        task = self._make_task(
            status=TaskStatus.COMPLETED,
            result={"content": "Done"},
            metadata={
                "agent_run.continuation": {"kind": "approval"},
                "agent_run.resume": {"approved": True},
                "language": "vi",
            },
        )
        payload = task_payload(task)
        self.assertNotIn("agent_run.continuation", payload["metadata"])
        self.assertNotIn("agent_run.resume", payload["metadata"])
        self.assertEqual(payload["metadata"]["language"], "vi")

    def test_task_payload_includes_timestamp(self) -> None:
        """Task payload includes ISO 8601 timestamp."""
        task = self._make_task()
        payload = task_payload(task)
        self.assertIn("timestamp", payload["status"])


if __name__ == "__main__":
    unittest.main()
