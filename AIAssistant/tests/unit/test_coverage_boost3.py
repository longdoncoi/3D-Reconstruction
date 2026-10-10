"""Third coverage batch: AgentService facade lifecycle and TaskService worker lifecycle.

These tests exercise the durable A2A task worker (submit -> working -> completed,
input_required -> resume, cancel, failure) and the AgentService facade edge
paths (cancel miss, execute without LLM, reset_state).
"""
from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from ai_assistant.agents import service as agents_service
from ai_assistant.agents.models import AgentCancelRequest, AgentExecuteRequest
from ai_assistant.agents.service import AgentService
from ai_assistant.application.tasks import TaskService
from ai_assistant.domain.errors import ModelNotLoadedError, NotFoundError
from ai_assistant.domain.tasks import AgentTask, TaskStatus


class _MemStore:
    """In-memory TaskStore double supporting the TaskService lifecycle."""

    def __init__(self) -> None:
        self._tasks: dict[str, AgentTask] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self.closed = False

    def create(self, task: AgentTask) -> None:
        with self._lock:
            self._tasks[task.id] = task
            self._events.setdefault(task.id, [])

    def get(self, task_id: str) -> AgentTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def save(self, task: AgentTask) -> None:
        with self._lock:
            self._tasks[task.id] = task

    def append_event(self, task_id: str, kind: str, data: dict) -> None:
        with self._lock:
            self._events.setdefault(task_id, []).append({"kind": kind, "data": data})

    def events(self, task_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events.get(task_id, []))

    def list(self, *, context_id: str | None = None, limit: int = 100) -> list[AgentTask]:
        with self._lock:
            tasks = list(self._tasks.values())
            if context_id:
                tasks = [t for t in tasks if t.context_id == context_id]
            return tasks[:limit]

    def close(self) -> None:
        self.closed = True


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class TaskServiceLifecycleTests(unittest.TestCase):
    """End-to-end worker lifecycle for durable A2A tasks."""

    def setUp(self) -> None:
        self.store = _MemStore()
        self.executor = MagicMock()
        self.service = TaskService(
            self.store, self.executor, frozenset({"supervisor", "research"})
        )

    def tearDown(self) -> None:
        self.service.close()

    def test_submit_unsupported_capability_rejected(self) -> None:
        task = self.service.submit("nonexistent", "hello")
        self.assertEqual(task.status, TaskStatus.REJECTED)
        self.assertIn("Unsupported capability", task.error or "")
        events = self.service.events(task.id)
        self.assertEqual(events[-1]["kind"], "status")

    def test_submit_to_completed(self) -> None:
        self.executor.return_value = {"status": "completed", "content": "Done"}
        task = self.service.submit("supervisor", "hello")
        self.assertEqual(task.status, TaskStatus.SUBMITTED)
        self.assertTrue(
            _wait_until(lambda: self.service.get(task.id).status == TaskStatus.COMPLETED)
        )
        self.assertEqual(self.service.get(task.id).result["content"], "Done")
        kinds = [event["kind"] for event in self.service.events(task.id)]
        self.assertIn("artifact", kinds)

    def test_executor_failure_becomes_observable(self) -> None:
        self.executor.side_effect = RuntimeError("backend exploded")
        task = self.service.submit("research", "query")
        self.assertTrue(
            _wait_until(lambda: self.service.get(task.id).status == TaskStatus.FAILED)
        )
        self.assertIn("backend exploded", self.service.get(task.id).error or "")

    def test_input_required_then_resume(self) -> None:
        first = {"status": "input_required", "continuation": {"step": 2}, "content": "Ask AI"}
        second = {"status": "completed", "content": "After approval"}
        self.executor.side_effect = [first, second]

        task = self.service.submit("supervisor", "apply patch")
        self.assertTrue(
            _wait_until(lambda: self.service.get(task.id).status == TaskStatus.INPUT_REQUIRED)
        )
        waiting = self.service.get(task.id)
        self.assertEqual(waiting.metadata["agent_run.continuation"], {"step": 2})

        resumed = self.service.resume(task.id, {"approved": True})
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed.status, TaskStatus.WORKING)
        self.assertTrue(
            _wait_until(lambda: self.service.get(task.id).status == TaskStatus.COMPLETED)
        )
        self.assertEqual(self.service.get(task.id).result["content"], "After approval")

    def test_resume_invalid_state_returns_none(self) -> None:
        task = self.service.submit("supervisor", "hello")
        self.assertTrue(
            _wait_until(lambda: self.service.get(task.id).status == TaskStatus.COMPLETED)
        )
        self.assertIsNone(self.service.resume(task.id, {"input": 1}))
        self.assertIsNone(self.service.resume("missing-task", {"input": 1}))

    def test_cancel_running_task(self) -> None:
        self.executor.side_effect = lambda task: time.sleep(1.0) or {"status": "completed"}
        task = self.service.submit("supervisor", "long task")
        cancelled = self.service.cancel(task.id)
        self.assertIsNotNone(cancelled)
        self.assertEqual(cancelled.status, TaskStatus.CANCELED)
        # Terminal cancellation means the late executor result is ignored.
        self.assertTrue(
            _wait_until(lambda: self.service.get(task.id).status == TaskStatus.CANCELED)
        )

    def test_cancel_unknown_or_terminal_returns_none(self) -> None:
        self.assertIsNone(self.service.cancel("missing"))
        self.executor.return_value = {"status": "completed"}
        task = self.service.submit("supervisor", "hello")
        self.assertTrue(
            _wait_until(lambda: self.service.get(task.id).status == TaskStatus.COMPLETED)
        )
        self.assertIsNone(self.service.cancel(task.id))

    def test_close_drains_workers_and_closes_store(self) -> None:
        self.executor.side_effect = lambda task: time.sleep(0.4) or {"status": "completed"}
        task = self.service.submit("supervisor", "slow")
        self.service.close(timeout_seconds=2.0)
        self.assertTrue(self.store.closed)
        self.assertTrue(
            _wait_until(lambda: self.service.get(task.id).status == TaskStatus.COMPLETED)
        )

    def test_submit_after_close_raises(self) -> None:
        self.service.close()
        with self.assertRaises(RuntimeError):
            self.service.submit("supervisor", "too late")

    def test_list_and_context_filtering(self) -> None:
        self.executor.return_value = {"status": "completed"}
        self.service.submit("supervisor", "hello", context_id="ctx-1")
        self.service.submit("research", "query", context_id="ctx-2")
        all_tasks = self.service.list()
        self.assertEqual(len(all_tasks), 2)
        only_ctx1 = self.service.list(context_id="ctx-1")
        self.assertEqual(len(only_ctx1), 1)
        self.assertEqual(only_ctx1[0].context_id, "ctx-1")


class AgentServiceFacadeTests(unittest.TestCase):
    """Coverage for the AgentService facade edge paths."""

    def test_cancel_unknown_session_raises_not_found(self) -> None:
        service = AgentService()
        with self.assertRaises(NotFoundError):
            service.cancel(AgentCancelRequest(session_id="does-not-exist"))

    def test_cancel_known_session_returns_cancelled(self) -> None:
        service = AgentService()
        session = f"facade-cancel-{time.time()}"
        agents_service.task_coordinator.start(session, task="t")
        result = service.cancel(AgentCancelRequest(session_id=session))
        self.assertEqual(result["status"], "cancelled")
        agents_service.task_coordinator.finish(session, success=False)

    def test_execute_without_llm_raises_model_not_loaded(self) -> None:
        service = AgentService(llm_runtime=None)
        with self.assertRaises(ModelNotLoadedError):
            service.execute(AgentExecuteRequest(task="do something"))

    def test_execute_delegates_to_agent_runner(self) -> None:
        mock_llm = MagicMock()
        mock_llm.llm = MagicMock()
        service = AgentService(llm_runtime=mock_llm)
        session = f"facade-exec-{time.time()}"
        with patch.object(
            agents_service, "run_langgraph_agent", return_value={"status": "completed", "content": "ok"}
        ) as mock_run:
            result = service.execute(
                AgentExecuteRequest(
                    task="summarise", session_id=session, retry_message_index=3,
                    history=[{"role": "user", "content": "  "}, {"role": "assistant", "content": "hi"}],
                    attachments=["a.txt"],
                )
            )
        mock_run.assert_called_once()
        self.assertEqual(result["retry_message_index"], 3)
        agents_service.task_coordinator.finish(session, success=True)

    def test_reset_state_clears_pending_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            from ai_assistant.adapters.persistence import PendingActionStore
            store = PendingActionStore(str(Path(tmpdir) / "pending.json"))
            store["key-1"] = {"tool": "read_file"}
            store.save()

            service = AgentService(pending_actions=store)
            self.assertIn("key-1", service.pending_actions)
            service.reset_state()
            self.assertNotIn("key-1", service.pending_actions)
            self.assertFalse(Path(store.path).exists())


if __name__ == "__main__":
    unittest.main()
