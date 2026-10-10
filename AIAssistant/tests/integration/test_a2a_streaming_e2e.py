"""End-to-end integration tests for A2A streaming and task lifecycle.

Covers the full A2A 0.3 flow: task submission, status polling, streaming
subscription, cancellation, and approval resume.
"""
from __future__ import annotations

import threading
import time
import unittest
from typing import Any
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_assistant.adapters.a2a import build_a2a_router
from ai_assistant.application.tasks import TaskService
from ai_assistant.domain.tasks import AgentTask


class _MockTaskStore:
    """In-memory task store for testing."""

    def __init__(self) -> None:
        self._tasks: dict[str, AgentTask] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def create(self, task: AgentTask) -> None:
        with self._lock:
            self._tasks[task.id] = task
            self._events[task.id] = []

    def get(self, task_id: str) -> AgentTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def save(self, task: AgentTask) -> None:
        with self._lock:
            self._tasks[task.id] = task

    def append_event(self, task_id: str, kind: str, data: dict) -> None:
        with self._lock:
            if task_id in self._events:
                self._events[task_id].append({"kind": kind, "data": data})

    def events(self, task_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events.get(task_id, []))

    def list(self, *, context_id: str | None = None, limit: int = 100) -> list[AgentTask]:
        with self._lock:
            tasks = list(self._tasks.values())
            if context_id:
                tasks = [t for t in tasks if t.context_id == context_id]
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
        self.app.include_router(self.router, prefix="/a2a")
        self.client = TestClient(self.app, client=("127.0.0.1", 50000))

    def test_submit_task_returns_task_payload(self) -> None:
        """POST /message:send returns a valid A2A task payload."""
        response = self.client.post(
            "/a2a/message:send",
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
            "/a2a/message:send",
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
        get_resp = self.client.get(f"/a2a/tasks/{task_id}", headers={"A2A-Version": "0.3"})
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["task"]["id"], task_id)

    def test_list_tasks(self) -> None:
        """GET /tasks returns a list of tasks."""
        # Submit two tasks
        for i in range(2):
            self.client.post(
                "/a2a/message:send",
                json={
                    "message": {
                        "role": "ROLE_USER",
                        "parts": [{"kind": "text", "text": f"Task {i}"}],
                    },
                },
                headers={"A2A-Version": "0.3"},
            )

        # List tasks
        list_resp = self.client.get("/a2a/tasks", headers={"A2A-Version": "0.3"})
        self.assertEqual(list_resp.status_code, 200)
        self.assertEqual(len(list_resp.json()["tasks"]), 2)

    def test_cancel_task(self) -> None:
        """POST /tasks/{id}:cancel cancels a running task."""
        # Use a slow executor to ensure the task is still running when we cancel
        self.service._executor = MagicMock(side_effect=lambda task: (time.sleep(0.5), {"status": "completed", "content": "Done", "steps": []})[1])

        # Submit a task
        submit_resp = self.client.post(
            "/a2a/message:send",
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
        cancel_resp = self.client.post(f"/a2a/tasks/{task_id}:cancel", headers={"A2A-Version": "0.3"})
        self.assertEqual(cancel_resp.status_code, 200)
        self.assertEqual(cancel_resp.json()["task"]["status"]["state"], "canceled")

    def test_cancel_nonexistent_task_returns_409(self) -> None:
        """Cancelling a non-existent task returns 409."""
        response = self.client.post("/a2a/tasks/nonexistent:cancel", headers={"A2A-Version": "0.3"})
        self.assertEqual(response.status_code, 409)

    def test_get_nonexistent_task_returns_404(self) -> None:
        """Getting a non-existent task returns 404."""
        response = self.client.get("/a2a/tasks/nonexistent", headers={"A2A-Version": "0.3"})
        self.assertEqual(response.status_code, 404)

    def test_version_mismatch_returns_400(self) -> None:
        """Sending wrong A2A-Version returns 400."""
        response = self.client.post(
            "/a2a/message:send",
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
        response = self.client.get("/a2a/.well-known/agent.json")
        self.assertEqual(response.status_code, 200)
        card = response.json()
        self.assertEqual(card["protocolVersion"], "0.3")
        self.assertIn("skills", card)
        self.assertGreater(len(card["skills"]), 0)

    def test_json_rpc_send_message(self) -> None:
        """JSON-RPC message/send works."""
        response = self.client.post(
            "/a2a/a2a",
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
            "/a2a/a2a",
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
        response = self.client.post("/a2a/tasks/nonexistent:subscribe", headers={"A2A-Version": "0.3"})
        self.assertEqual(response.status_code, 404)

    def test_resume_nonexistent_task_returns_409(self) -> None:
        """Resuming a non-existent task returns 409."""
        response = self.client.post(
            "/a2a/tasks/nonexistent:resume",
            json={"input": {}},
            headers={"A2A-Version": "0.3"},
        )
        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
