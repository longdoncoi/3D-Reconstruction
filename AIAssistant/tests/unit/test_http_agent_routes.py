"""Unit tests for the /v1/agent/* HTTP router.

The router is transport-only: a fake ``AgentService`` records the calls and
returns canned results, so no LangGraph infrastructure is needed.
"""
from __future__ import annotations

import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_assistant.adapters.http.agent_routes import build_agent_router


class _FakeAgentService:
    def __init__(self, execute_result=None, execute_error: Exception | None = None) -> None:
        self.execute_result = execute_result or {"status": "ok", "result": "hello"}
        self.execute_error = execute_error
        self.calls: list[tuple[str, object]] = []

    def execute(self, request, event_sink=None, client_host=None):
        self.calls.append(("execute", (request, event_sink, client_host)))
        if self.execute_error is not None:
            raise self.execute_error
        if event_sink is not None:
            event_sink({"node": "tools"})
        return self.execute_result

    def cancel(self, request):
        self.calls.append(("cancel", request))
        return {"status": "cancelled"}

    def ui_action_result(self, request):
        self.calls.append(("ui_action_result", request))
        return {"status": "ok"}

    def approve(self, request):
        self.calls.append(("approve", request))
        return {"status": "approved"}


def _build_client(service: _FakeAgentService | None = None) -> tuple[TestClient, _FakeAgentService]:
    service = service or _FakeAgentService()
    app = FastAPI()
    app.include_router(build_agent_router(service))
    return TestClient(app, client=("127.0.0.1", 50000)), service


class AgentRoutesTest(unittest.TestCase):
    def test_execute_returns_json_result(self) -> None:
        client, service = _build_client()
        response = client.post(
            "/v1/agent/execute",
            json={"task": "summarize the repo"},
            headers={"accept": "application/json"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        (name, (request, event_sink, _)) = service.calls[0]
        self.assertEqual(name, "execute")
        self.assertIsNone(event_sink)
        self.assertEqual(request.task, "summarize the repo")

    def test_execute_streams_sse_events(self) -> None:
        client, _ = _build_client()
        response = client.post(
            "/v1/agent/execute",
            json={"task": "do something"},
            headers={"accept": "text/event-stream"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        body = response.text
        self.assertIn("event: status", body)
        self.assertIn("event: step", body)
        self.assertIn("event: done", body)
        self.assertIn('"status": "ok"', body)

    def test_execute_streams_error_event_when_service_fails(self) -> None:
        failing = _FakeAgentService(execute_error=RuntimeError("boom"))
        client, _ = _build_client(failing)
        response = client.post(
            "/v1/agent/execute",
            json={"task": "do something"},
            headers={"accept": "text/event-stream"},
        )
        body = response.text
        self.assertIn("event: error", body)
        self.assertIn("boom", body)

    def test_default_service_is_constructed_when_none_given(self) -> None:
        with mock.patch(
            "ai_assistant.agents.service.AgentService", _FakeAgentService
        ):
            client = TestClient(FastAPI(), client=("127.0.0.1", 50000))
            client.app.include_router(build_agent_router())
        response = client.post(
            "/v1/agent/execute",
            json={"task": "ping"},
            headers={"accept": "application/json"},
        )
        self.assertEqual(response.status_code, 200)

    def test_cancel_route(self) -> None:
        client, service = _build_client()
        response = client.post("/v1/agent/cancel", json={"session_id": "s1"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "cancelled")
        self.assertEqual(service.calls[0][0], "cancel")

    def test_ui_action_result_route(self) -> None:
        client, service = _build_client()
        response = client.post(
            "/v1/agent/ui-action-result",
            json={"request_id": "r1", "success": True},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.calls[0][0], "ui_action_result")

    def test_approve_route(self) -> None:
        client, service = _build_client()
        response = client.post(
            "/v1/agent/approve",
            json={"action_id": "a1", "approved": True},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "approved")
        self.assertEqual(service.calls[0][0], "approve")

    def test_request_validation_rejects_bad_language(self) -> None:
        client, _ = _build_client()
        response = client.post(
            "/v1/agent/execute",
            json={"task": "x", "language": "fr"},
            headers={"accept": "application/json"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
