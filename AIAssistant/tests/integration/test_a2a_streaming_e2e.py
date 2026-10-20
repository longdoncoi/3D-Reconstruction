"""End-to-end integration tests for A2A streaming and task lifecycle.

Covers the full A2A 0.3 flow: task submission, status polling, streaming
subscription, cancellation, and approval resume.
"""
from __future__ import annotations

import json
import threading
import time
import unittest
from typing import Any
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_assistant.adapters.a2a import build_a2a_router
from ai_assistant.application.tasks import TaskService
from ai_assistant.domain.tasks import AgentTask, TaskStatus

# ``build_a2a_router`` owns its own paths and the composition root mounts it at
# the application root, so the 0.3 endpoints are ``/message:send`` and
# ``/tasks/{id}`` while the pre-0.3 endpoints stay under ``/a2a/tasks``.
LEGACY_EVENTS_PATH = "/a2a/tasks"


class _MockTaskStore:
    """In-memory task store for testing."""

    def __init__(self) -> None:
        self._tasks: dict[str, AgentTask] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        # Cursors the service was asked for — proves the router resumes from a
        # cursor instead of re-reading and re-slicing the whole log each poll.
        self.event_reads: list[tuple[str, int]] = []

    def commit(self, task: AgentTask, *, expected: TaskStatus | None = None,
               events: tuple[tuple[str, dict[str, Any]], ...] = ()) -> bool:
        """ADR 0032: the port's single unit of work — state and events land
        together, and a refused compare-and-set writes neither."""
        with self._lock:
            if expected is not None:
                current = self._tasks.get(task.id)
                if current is None or current.status != expected:
                    return False
            self._tasks[task.id] = task
            recorded = self._events.setdefault(task.id, [])
            for kind, data in events:
                # Round-trip through JSON like the SQLite store does, so domain
                # enums arrive in the SSE stream as their string form.
                payload = json.loads(json.dumps(data, default=str))
                recorded.append({"seq": len(recorded) + 1, "kind": kind, "data": payload})
            return True

    def get(self, task_id: str) -> AgentTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def append_event(self, task_id: str, kind: str, data: dict) -> None:
        with self._lock:
            recorded = self._events.get(task_id)
            if recorded is not None:
                # Round-trip through JSON like the SQLite store does, so domain
                # enums arrive in the SSE stream as their string form.
                payload = json.loads(json.dumps(data, default=str))
                recorded.append({"seq": len(recorded) + 1, "kind": kind, "data": payload})

    def events(self, task_id: str, *, since: int = 0) -> list[dict[str, Any]]:
        """Cursor form of the port: ``since`` is a store-native ordering key.

        Events carry the 1-based ``seq`` the SQLite store exposes, so the router
        resumes from a cursor instead of re-reading and re-slicing the log.
        """
        with self._lock:
            recorded = self._events.get(task_id, [])
            self.event_reads.append((task_id, since))
            return [event for event in recorded if event["seq"] > since]

    def list(self, *, context_id: str | None = None, limit: int = 100,
             owner: str | None = None) -> list[AgentTask]:
        with self._lock:
            tasks = list(self._tasks.values())
            if context_id:
                tasks = [t for t in tasks if t.context_id == context_id]
            if owner is not None:
                tasks = [t for t in tasks if t.owner == owner]
            return tasks[:limit]


class A2AStreamingE2ETests(unittest.TestCase):
    """End-to-end A2A streaming and lifecycle tests."""

    def setUp(self) -> None:
        self.store = _MockTaskStore()
        self.service = TaskService(
            self.store,
            MagicMock(return_value={"status": "completed", "content": "Done", "steps": []}),
            frozenset({"supervisor", "research"}),
        )
        self.router = build_a2a_router(self.service, "Test Agent", "1.0.0")
        self.app = FastAPI()
        # Mounted the way the composition root mounts it: the router owns its
        # paths (no prefix), otherwise every 0.3 endpoint moves and the legacy
        # ``/a2a/tasks`` tree ends up double-prefixed.
        self.app.include_router(self.router)
        self.client = TestClient(self.app, client=("127.0.0.1", 50000))

    def test_submit_task_returns_task_payload(self) -> None:
        """POST /message:send returns a valid A2A task payload."""
        response = self.client.post(
            "/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"kind": "text", "text": "Hello"}],
                },
                "configuration": {"returnImmediately": True},
            },
            headers={"A2A-Version": "0.3"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("task", data)
        self.assertEqual(data["task"]["status"]["state"], "submitted")
        self.assertIn("id", data["task"])

    def test_get_task_by_id(self) -> None:
        """GET /tasks/{id} returns the task."""
        # Submit a task
        submit_resp = self.client.post(
            "/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"kind": "text", "text": "Hello"}],
                },
            },
            headers={"A2A-Version": "0.3"},
        )
        task_id = submit_resp.json()["task"]["id"]

        # Get the task
        get_resp = self.client.get(f"/tasks/{task_id}", headers={"A2A-Version": "0.3"})
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["task"]["id"], task_id)

    def test_list_tasks(self) -> None:
        """GET /tasks returns a list of tasks."""
        # Submit two tasks
        for i in range(2):
            self.client.post(
                "/message:send",
                json={
                    "message": {
                        "role": "ROLE_USER",
                        "parts": [{"kind": "text", "text": f"Task {i}"}],
                    },
                },
                headers={"A2A-Version": "0.3"},
            )

        # List tasks
        list_resp = self.client.get("/tasks", headers={"A2A-Version": "0.3"})
        self.assertEqual(list_resp.status_code, 200)
        self.assertEqual(len(list_resp.json()["tasks"]), 2)

    def test_cancel_task(self) -> None:
        """POST /tasks/{id}:cancel cancels a running task."""
        # Use a slow executor to ensure the task is still running when we cancel
        self.service._executor = MagicMock(side_effect=lambda task: (time.sleep(0.5), {"status": "completed", "content": "Done", "steps": []})[1])

        # Submit a task
        submit_resp = self.client.post(
            "/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"kind": "text", "text": "Long task"}],
                },
            },
            headers={"A2A-Version": "0.3"},
        )
        task_id = submit_resp.json()["task"]["id"]

        # Cancel it while it's still running
        cancel_resp = self.client.post(f"/tasks/{task_id}:cancel", headers={"A2A-Version": "0.3"})
        self.assertEqual(cancel_resp.status_code, 200)
        self.assertEqual(cancel_resp.json()["task"]["status"]["state"], "canceled")

    def test_cancel_nonexistent_task_returns_409(self) -> None:
        """Cancelling a non-existent task returns 409."""
        response = self.client.post("/tasks/nonexistent:cancel", headers={"A2A-Version": "0.3"})
        self.assertEqual(response.status_code, 409)

    def test_get_nonexistent_task_returns_404(self) -> None:
        """Getting a non-existent task returns 404."""
        response = self.client.get("/tasks/nonexistent", headers={"A2A-Version": "0.3"})
        self.assertEqual(response.status_code, 404)

    def test_version_mismatch_returns_400(self) -> None:
        """Sending wrong A2A-Version returns 400."""
        response = self.client.post(
            "/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"kind": "text", "text": "Hello"}],
                },
            },
            headers={"A2A-Version": "0.2"},
        )
        self.assertEqual(response.status_code, 400)

    def test_agent_card_discovery(self) -> None:
        """GET /.well-known/agent.json returns valid Agent Card."""
        response = self.client.get("/.well-known/agent.json")
        self.assertEqual(response.status_code, 200)
        card = response.json()
        self.assertEqual(card["protocolVersion"], "0.3")
        self.assertIn("skills", card)
        self.assertGreater(len(card["skills"]), 0)

    def test_json_rpc_send_message(self) -> None:
        """JSON-RPC message/send works."""
        response = self.client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "ROLE_USER",
                        "parts": [{"kind": "text", "text": "Hello via RPC"}],
                    },
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["jsonrpc"], "2.0")
        self.assertEqual(data["id"], 1)
        self.assertIn("result", data)
        self.assertIn("task", data["result"])

    def test_json_rpc_unsupported_method_returns_error(self) -> None:
        """JSON-RPC with unsupported method returns error."""
        response = self.client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "unsupported/method",
                "params": {},
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], -32602)

    def test_subscribe_to_nonexistent_task_returns_404(self) -> None:
        """Subscribing to a non-existent task returns 404."""
        response = self.client.post("/tasks/nonexistent:subscribe", headers={"A2A-Version": "0.3"})
        self.assertEqual(response.status_code, 404)

    def test_resume_nonexistent_task_returns_409(self) -> None:
        """Resuming a non-existent task returns 409."""
        response = self.client.post(
            "/tasks/nonexistent:resume",
            json={"input": {}},
            headers={"A2A-Version": "0.3"},
        )
        self.assertEqual(response.status_code, 409)


    def test_task_events_stream_resumes_from_the_cursor(self) -> None:
        """GET /a2a/tasks/{id}/events streams the log and never re-reads it.

        The store now exposes an ``events(task_id, *, since)`` cursor (SQLite
        ``rowid``); a stream that re-read and re-sliced the whole log on every
        poll would show up here as a second read starting back at cursor 0.
        """
        # A slow executor keeps the task non-terminal, so the stream has to poll
        # more than once and each poll must resume after the last delivered event.
        self.service._executor = MagicMock(
            side_effect=lambda task: (time.sleep(0.4), {"status": "completed", "content": "Done", "steps": []})[1]
        )
        submit_resp = self.client.post(
            "/message:send",
            json={"message": {"role": "ROLE_USER", "parts": [{"kind": "text", "text": "Stream me"}]}},
            headers={"A2A-Version": "0.3"},
        )
        task_id = submit_resp.json()["task"]["id"]
        self.store.append_event(task_id, "message", {"text": "step-1"})

        with self.client.stream("GET", f"{LEGACY_EVENTS_PATH}/{task_id}/events") as stream:
            body = "".join(stream.iter_text())

        self.assertIn("event: message", body)
        self.assertIn('"text": "step-1"', body)
        self.assertIn("event: done", body)

        reads = [cursor for owner, cursor in self.store.event_reads if owner == task_id]
        self.assertTrue(reads)
        self.assertEqual(reads[0], 0)
        self.assertEqual(reads, sorted(reads), "the cursor must never move backwards")
        self.assertGreater(max(reads), 0, "a later poll must resume, not re-read from the start")


if __name__ == "__main__":
    unittest.main()
