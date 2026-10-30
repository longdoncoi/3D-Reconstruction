"""Consumer-driven contract test — Qt desktop client perspective (ADR 0015).

Every payload below is copied from the *consumer*, ``src/services/ai/AIAssistant.cpp``:

* ``buildAgentPayload``    → POST /v1/agent/execute          {task, history, temperature, language, session_id}
* ``reportUiActionResult`` → POST /v1/agent/ui-action-result {request_id, success, result}
* ``approvePendingAction`` → POST /v1/agent/approve          {action_id, approved, session_id}
* ``onAgentReplyFinished`` → reads {status, request_id, steps[{type, tool, params}]}

The previous version of this file called ``create_app()`` without its required
arguments (so the class errored out and nothing ran) and asserted request
bodies the client never sends (``query``, ``approve``); a contract test that
matches neither side proves nothing.
"""
from __future__ import annotations

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ai_assistant.adapters.http.agent_routes import build_agent_router
from ai_assistant.agents.models import (
    AgentApproveRequest,
    AgentExecuteRequest,
    AgentUiActionResultRequest,
)
from ai_assistant.domain.memory import DEFAULT_MEMORY_OWNER

# The request bodies the Qt client actually sends.
QT_EXECUTE_PAYLOAD = {
    "task": "Xây dựng lại target Renderer",
    "history": [{"role": "user", "content": "xin chào"}, {"role": "assistant", "content": "ok"}],
    "temperature": 0.3,
    "language": "vi",
    "session_id": "qt-session-1",
}
QT_APPROVE_PAYLOAD = {"action_id": "act-1", "approved": True, "session_id": "qt-session-1"}
QT_UI_ACK_PAYLOAD = {"request_id": "req-1", "success": True, "result": {"handled_by": "desktop"}}

# The statuses ``onAgentReplyFinished`` branches on.
QT_PENDING_APPROVAL = "pending_approval"
QT_PENDING_UI_ACTION = "pending_ui_action"


class _RecordingAgentService:
    """Stand-in for ``AgentService``: records the parsed request per endpoint."""

    def __init__(self) -> None:
        self.requests: dict[str, object] = {}
        self.owners: list[str | None] = []

    def execute(self, request, *, owner=None, event_sink=None, client_host: str = "") -> dict:
        self.requests["execute"] = request
        self.owners.append(owner)
        return {"status": "completed", "content": "ok", "steps": []}

    def cancel(self, request) -> dict:
        self.requests["cancel"] = request
        return {"status": "cancelled", "session_id": request.session_id}

    def ui_action_result(self, request) -> dict:
        self.requests["ui_action_result"] = request
        return {"status": "completed", "request_id": request.request_id}

    def approve(self, request) -> dict:
        self.requests["approve"] = request
        return {"status": "completed", "action_id": request.action_id}


def _client() -> tuple[TestClient, _RecordingAgentService]:
    service = _RecordingAgentService()
    app = FastAPI()
    app.include_router(build_agent_router(agent_service=service))
    # The agent surface is loopback-only (ADR 0008) and TestClient's default
    # peer is the literal name "testclient", so the test speaks from loopback
    # exactly like the desktop does.
    return TestClient(app, client=("127.0.0.1", 54321)), service


class QtRequestSchemaTests(unittest.TestCase):
    """The consumer's JSON must parse into the server's request models."""

    def test_execute_payload_parses(self) -> None:
        request = AgentExecuteRequest(**QT_EXECUTE_PAYLOAD)
        self.assertEqual(request.task, QT_EXECUTE_PAYLOAD["task"])
        self.assertEqual(request.session_id, "qt-session-1")
        self.assertEqual(request.language, "vi")
        self.assertEqual(request.temperature, 0.3)
        self.assertEqual(len(request.history), 2)

    def test_execute_accepts_a_long_session_history(self) -> None:
        """``buildAgentPayload`` sends every prior message, with no cap.

        A 20-entry limit made every long desktop session answer HTTP 422.
        """
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
                   for i in range(60)]
        request = AgentExecuteRequest(**{**QT_EXECUTE_PAYLOAD, "history": history})
        self.assertEqual(len(request.history), 60)

    def test_execute_requires_task_and_vietnamese_or_english(self) -> None:
        with self.assertRaises(ValidationError):
            AgentExecuteRequest(**{key: value for key, value in QT_EXECUTE_PAYLOAD.items() if key != "task"})
        with self.assertRaises(ValidationError):
            AgentExecuteRequest(**{**QT_EXECUTE_PAYLOAD, "language": "de"})

    def test_approve_payload_parses(self) -> None:
        request = AgentApproveRequest(**QT_APPROVE_PAYLOAD)
        self.assertEqual((request.action_id, request.approved, request.session_id),
                         ("act-1", True, "qt-session-1"))

    def test_approve_requires_the_boolean_field_qt_sends(self) -> None:
        with self.assertRaises(ValidationError):
            AgentApproveRequest(**{key: value for key, value in QT_APPROVE_PAYLOAD.items() if key != "approved"})

    def test_ui_ack_payload_parses(self) -> None:
        request = AgentUiActionResultRequest(**QT_UI_ACK_PAYLOAD)
        self.assertEqual(request.request_id, "req-1")
        self.assertTrue(request.success)
        self.assertEqual(request.result, {"handled_by": "desktop"})


class QtTransportContractTests(unittest.TestCase):
    """Route-level: the endpoints exist, accept the Qt bodies and forward them."""

    def test_execute_endpoint_forwards_the_parsed_request(self) -> None:
        client, service = _client()
        response = client.post("/v1/agent/execute", json=QT_EXECUTE_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        forwarded = service.requests["execute"]
        self.assertIsInstance(forwarded, AgentExecuteRequest)
        self.assertEqual(forwarded.task, QT_EXECUTE_PAYLOAD["task"])
        self.assertEqual(service.owners, [DEFAULT_MEMORY_OWNER])
        self.assertEqual(response.json()["status"], "completed")

    def test_approve_endpoint_forwards_the_parsed_request(self) -> None:
        client, service = _client()
        response = client.post("/v1/agent/approve", json=QT_APPROVE_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(service.requests["approve"], AgentApproveRequest)

    def test_ui_action_ack_endpoint_forwards_the_parsed_request(self) -> None:
        client, service = _client()
        response = client.post("/v1/agent/ui-action-result", json=QT_UI_ACK_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(service.requests["ui_action_result"], AgentUiActionResultRequest)

    def test_cancel_endpoint_exists_for_the_envelope_shape(self) -> None:
        client, _service = _client()
        response = client.post("/v1/agent/cancel", json={"session_id": "qt-session-1"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())

    def test_a_non_loopback_peer_is_rejected(self) -> None:
        """ADR 0008: the Qt surface must not be reachable from another host."""
        service = _RecordingAgentService()
        app = FastAPI()
        app.include_router(build_agent_router(agent_service=service))
        remote = TestClient(app, client=("203.0.113.9", 51000))
        self.assertEqual(remote.post("/v1/agent/execute", json=QT_EXECUTE_PAYLOAD).status_code, 403)
        self.assertNotIn("execute", service.requests)


class QtResponseContractTests(unittest.TestCase):
    """``onAgentReplyFinished`` reads these keys; keep them present."""

    def test_completed_status_carries_steps_and_content(self) -> None:
        client, _ = _client()
        payload = client.post("/v1/agent/execute", json=QT_EXECUTE_PAYLOAD).json()
        self.assertEqual(payload["status"], "completed")
        for key in ("status", "steps", "content"):
            self.assertIn(key, payload)
        self.assertIsInstance(payload["steps"], list)
        # ``request_id`` is only meaningful for the pending envelopes; Qt reads
        # it with ``toString()`` and treats an empty value as "no follow-up", so
        # its absence here is part of the contract, not an omission.


if __name__ == "__main__":
    unittest.main()
