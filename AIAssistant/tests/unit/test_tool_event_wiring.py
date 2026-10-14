"""The tool chokepoint must be observable: ``build_container`` wires a sink.

``ToolExecutionService`` has accepted an ``event_sink`` for a while and the
guard tests pin the mechanism — but the composition root built
``ToolExecutionService(plugins)`` with no sink, so every execution that did not
come from the agent loop (a HITL-approved invocation, a desktop action, the
A2A task worker) happened without leaving a trace. These tests pin both halves
of the fix:

* the default sink really fires for success, for a policy rejection and for an
  approval-gated call, and its parameters are redacted before they are logged;
* an injected sink replaces the default, so the wiring is a parameter of
  ``build_container`` rather than a copy someone can forget to use;
* the shared ``TaskCoordinator`` carries the cancel-to-stop recovery observer
  (SCORECARD #4), while an injected coordinator keeps its own configuration.
"""
from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from typing import Any

from ai_assistant.application.coordination import TaskCoordinator
from ai_assistant.bootstrap.container import PlatformContainer, build_container
from ai_assistant.domain.security import DataClassification, Principal
from ai_assistant.domain.tools import ToolRequest, ToolResult
from ai_assistant.observability.metrics import record_recovery
from ai_assistant.observability.tool_events import audit_tool_event, tool_event_payload
from ai_assistant.tools.definition import ToolDefinition

_LOGGER_NAME = "agent.tools"
_SECRET = "hunter2"


def _read_tool() -> ToolDefinition:
    return ToolDefinition(
        name="read_file",
        description="read a file",
        parameters={"path": {"type": "string", "required": True}},
        policy="read_only",
        handler=lambda params: {"content": "x"},
    )


def _desktop_tool() -> ToolDefinition:
    return ToolDefinition(
        name="desktop_login",
        description="sign in on the desktop app",
        parameters={
            "username": {"type": "string", "required": True},
            "password": {"type": "string", "required": True},
        },
        policy="desktop_ack",
        requires_approval=True,
        handler=lambda params: {"ack": True},
    )


def _principal() -> Principal:
    return Principal(
        "tester",
        frozenset({"project.read", "desktop.action"}),
        DataClassification.INTERNAL,
    )


def _events(records: list[logging.LogRecord]) -> list[dict[str, Any]]:
    """Parse the ``tool_event {json}`` messages a captured log produced."""
    events = []
    for record in records:
        message = record.getMessage()
        if message.startswith("tool_event "):
            events.append(json.loads(message.split(" ", 1)[1]))
    return events


class DefaultSinkTests(unittest.TestCase):
    def _build(self, entries: list[ToolDefinition]) -> PlatformContainer:
        container = build_container(
            Path(self._directory), list(entries), {},
            task_executor=lambda task: {},
        )
        self.addCleanup(container.tasks._store.close)
        return container

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._directory = self._tmp.name

    def test_a_successful_execution_is_recorded_on_the_chokepoint_logger(self) -> None:
        container = self._build([_read_tool()])

        with self.assertLogs(_LOGGER_NAME, level="INFO") as captured:
            result = container.tools.execute(
                "read_file", {"path": "src/app.cpp"}, _principal(), correlation_id="corr-1",
            )

        self.assertTrue(result.success)
        events = _events(captured.records)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["tool"], "read_file")
        self.assertEqual(events[0]["correlation_id"], "corr-1")
        self.assertTrue(events[0]["ok"])
        self.assertIsNone(events[0]["error_code"])
        self.assertEqual(events[0]["parameters"], {"path": "src/app.cpp"})

    def test_the_agent_gateway_path_is_recorded_too(self) -> None:
        """The LangGraph loop executes through the gateway; that path logs as well."""
        container = self._build([_read_tool()])

        with self.assertLogs(_LOGGER_NAME, level="INFO") as captured:
            result = container.gateway.execute("read_file", {"path": "src/app.cpp"})

        self.assertTrue(result["success"])
        events = _events(captured.records)
        self.assertEqual([event["tool"] for event in events], ["read_file"])
        self.assertTrue(events[0]["ok"])

    def test_an_approval_gated_execution_is_recorded_with_redacted_parameters(self) -> None:
        """HITL-approved runs are exactly the executions the agent loop never sees."""
        container = self._build([_desktop_tool()])
        parameters = {"username": "qt-user", "password": _SECRET}
        container.tools.issue_approval_grant("desktop_login", parameters, "tok-1")

        with self.assertLogs(_LOGGER_NAME, level="INFO") as captured:
            result = container.tools.execute(
                "desktop_login", parameters, _principal(),
                correlation_id="corr-2", approval_token="tok-1",
            )

        self.assertTrue(result.success)
        events = _events(captured.records)
        self.assertEqual(events[0]["error_code"], None)
        self.assertEqual(events[0]["parameters"]["username"], "qt-user")
        self.assertEqual(events[0]["parameters"]["password"], "[redacted]")
        self.assertNotIn(_SECRET, "\n".join(captured.output))

    def test_a_policy_rejection_is_recorded_as_an_event(self) -> None:
        """A refused call is the event an operator most needs to see."""
        container = self._build([_read_tool()])

        with self.assertLogs(_LOGGER_NAME, level="INFO") as captured:
            result = container.tools.execute(
                "no_such_tool", {}, _principal(), correlation_id="corr-3",
            )

        self.assertFalse(result.success)
        events = _events(captured.records)
        self.assertEqual(events[0]["error_code"], "tool_not_found")
        self.assertFalse(events[0]["ok"])


class InjectedSinkTests(unittest.TestCase):
    def test_an_injected_sink_replaces_the_default(self) -> None:
        seen: list[tuple[str, bool]] = []

        def collect(request: ToolRequest, result: ToolResult) -> None:
            seen.append((request.tool_name, result.success))

        with tempfile.TemporaryDirectory() as directory:
            container = build_container(
                Path(directory), [_read_tool()], {},
                task_executor=lambda task: {},
                tool_event_sink=collect,
            )
            try:
                container.tools.execute("read_file", {"path": "a.txt"}, _principal())
            finally:
                container.tasks._store.close()

        self.assertEqual(seen, [("read_file", True)])


class RecoveryObserverWiringTests(unittest.TestCase):
    """The coordinator must carry the cancel-to-stop observer (SCORECARD #4)."""

    def test_the_default_coordinator_observes_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            container = build_container(
                Path(directory), [_read_tool()], {}, task_executor=lambda task: {},
            )
            try:
                self.assertIs(container.coordinator._recovery_observer, record_recovery)
            finally:
                container.tasks._store.close()

    def test_an_injected_coordinator_is_left_alone(self) -> None:
        injected = TaskCoordinator()
        with tempfile.TemporaryDirectory() as directory:
            container = build_container(
                Path(directory), [_read_tool()], {}, task_executor=lambda task: {},
                coordinator=injected,
            )
            try:
                self.assertIs(container.coordinator, injected)
                self.assertIsNone(container.coordinator._recovery_observer)
            finally:
                container.tasks._store.close()


class PayloadTests(unittest.TestCase):
    def test_the_payload_is_json_safe_and_redacted(self) -> None:
        request = ToolRequest("desktop_login", {"username": "u", "password": _SECRET}, "c-9")
        result = ToolResult(True, {"ack": True})

        payload = tool_event_payload(request, result)

        self.assertEqual(json.loads(json.dumps(payload))["parameters"]["password"], "[redacted]")

    def test_the_default_sink_is_the_auditing_one(self) -> None:
        """``audit_tool_event`` is exported for the composition root to name."""
        self.assertTrue(callable(audit_tool_event))

    def test_a_preflight_record_is_not_mistaken_for_an_execution(self) -> None:
        """The remote leg authorises before it dispatches; without the field
        that record would read as a tool that had run in this process."""
        request = ToolRequest("run_command", {"command": "ls"}, "c-10")

        executed = tool_event_payload(request, ToolResult(True, {"stdout": "ok"}))
        authorised = tool_event_payload(
            request, ToolResult(True, {"tool": "run_command", "preflight": True}),
        )

        self.assertEqual(executed["dispatch"], "execute")
        self.assertEqual(authorised["dispatch"], "preflight")


if __name__ == "__main__":
    unittest.main()
