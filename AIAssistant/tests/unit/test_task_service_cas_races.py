"""Guard-path coverage for the durable task lifecycle's compare-and-swap.

``TaskService`` writes through a store that may be updated by another writer
(a cancel landing on a different thread, a UI acknowledgement, an A2A
subscriber). Every mutation is therefore a compare-and-swap: when the CAS
loses the service must *return* instead of overwriting the newer status with
an unconditional write.

Each arm below is one lost race, pinned with a store double whose
``save_if_status`` refuses the write, plus the duck-typed fallback used by
stores that do not implement the CAS at all.
"""
from __future__ import annotations

import unittest
from typing import Any

from ai_assistant.application.coordination import TaskCoordinator
from ai_assistant.application.tasks import TaskService
from ai_assistant.domain.tasks import AgentTask, TaskStatus


class _CasStore:
    """Store double with controllable compare-and-swap outcomes.

    ``cas_ok=False`` loses every race; ``fail_after=N`` lets the first N writes
    land (the run starts) and loses everything after them, which is how a cancel
    arriving *while the executor runs* looks from the store's side.
    """

    def __init__(self, task: AgentTask, *, cas_ok: bool = True, fail_after: int | None = None) -> None:
        self._task = task
        self.cas_ok = cas_ok
        self.fail_after = fail_after
        self.cas_calls = 0
        self.saved: list[AgentTask] = []
        self.events: list[tuple[str, Any]] = []

    def create(self, task: AgentTask) -> None:
        self._task = task

    def get(self, task_id: str) -> AgentTask | None:
        return self._task

    def save(self, task: AgentTask) -> None:
        self.saved.append(task)
        self._task = task

    def save_if_status(self, task: AgentTask, expected: TaskStatus) -> bool:
        self.cas_calls += 1
        lost = not self.cas_ok or (self.fail_after is not None and self.cas_calls >= self.fail_after)
        if lost:
            return False
        if self._task is None or self._task.status != expected:
            return False
        self._task = task
        return True

    def append_event(self, task_id: str, kind: str, data: dict) -> None:
        self.events.append((kind, data))

    def events_for(self, task_id: str) -> list[tuple[str, Any]]:
        return list(self.events)


class _PlainStore:
    """Store without ``save_if_status``: the service falls back to get/compare/save."""

    def __init__(self, task: AgentTask) -> None:
        self._task = task
        self.saves = 0

    def get(self, task_id: str) -> AgentTask | None:
        return self._task

    def save(self, task: AgentTask) -> None:
        self.saves += 1
        self._task = task


def _submitted(message: str = "hi") -> AgentTask:
    return AgentTask.new("chat", message)


def _working() -> AgentTask:
    return _submitted().transition(TaskStatus.WORKING)


def _waiting() -> AgentTask:
    task = _working().with_metadata({"agent_run.continuation": {"kind": "approval"}})
    return task.transition(TaskStatus.INPUT_REQUIRED)


class LostRaceTests(unittest.TestCase):
    def test_a_cancel_that_loses_the_cas_returns_none(self) -> None:
        store = _CasStore(_working(), cas_ok=False)
        service = TaskService(store, lambda task: {}, frozenset({"chat"}))

        self.assertIsNone(service.cancel("task-1"))
        self.assertEqual(store.get("task-1").status, TaskStatus.WORKING)
        self.assertEqual(store.events, [])

    def test_a_won_cancel_notifies_the_shared_coordinator(self) -> None:
        """The A2A cancel must flip the flag the runner's cancel_checker polls."""
        coordinator = TaskCoordinator()
        store = _CasStore(_working(), cas_ok=True)
        service = TaskService(store, lambda task: {}, frozenset({"chat"}), coordinator=coordinator)

        cancelled = service.cancel("task-1")
        self.assertIsNotNone(cancelled)
        self.assertTrue(coordinator.is_cancelled("task-1"))

    def test_a_run_that_loses_the_cas_never_executes(self) -> None:
        store = _CasStore(_submitted(), cas_ok=False)
        calls: list[AgentTask] = []
        service = TaskService(store, lambda task: calls.append(task) or {"status": "completed"}, frozenset({"chat"}))

        service.submit("chat", "hi")
        service.close(timeout_seconds=5)

        self.assertEqual(calls, [])

    def test_a_resume_that_loses_the_cas_returns_none(self) -> None:
        store = _CasStore(_waiting(), cas_ok=False)
        service = TaskService(store, lambda task: {}, frozenset({"chat"}))

        self.assertIsNone(service.resume("task-1", {"choice": "approve"}))

    def test_an_input_required_write_that_loses_the_cas_stays_working(self) -> None:
        # The run starts (CAS #1 lands), then a cancel wins the race before the
        # INPUT_REQUIRED row can be written.
        store = _CasStore(_submitted(), fail_after=2)
        service = TaskService(
            store,
            lambda task: {"status": "input_required", "continuation": {"kind": "approval"}},
            frozenset({"chat"}),
        )

        service.submit("chat", "hi")
        service.close(timeout_seconds=5)

        self.assertEqual(store.get("task-1").status, TaskStatus.WORKING)
        statuses = [data["status"] for kind, data in store.events if kind == "status"]
        self.assertEqual(statuses, [TaskStatus.SUBMITTED, TaskStatus.WORKING])

    def test_a_completion_write_that_loses_the_cas_stays_put(self) -> None:
        store = _CasStore(_submitted(), fail_after=2)
        service = TaskService(
            store, lambda task: {"status": "completed", "content": "ok"}, frozenset({"chat"})
        )

        service.submit("chat", "hi")
        service.close(timeout_seconds=5)

        self.assertEqual(store.get("task-1").status, TaskStatus.WORKING)
        statuses = [data["status"] for kind, data in store.events if kind == "status"]
        self.assertEqual(statuses, [TaskStatus.SUBMITTED, TaskStatus.WORKING])

    def test_a_failure_write_that_loses_the_cas_stays_put(self) -> None:
        def boom(task: AgentTask) -> dict:
            msg = "executor exploded"
            raise RuntimeError(msg)

        store = _CasStore(_submitted(), fail_after=2)
        service = TaskService(store, boom, frozenset({"chat"}))

        service.submit("chat", "hi")
        service.close(timeout_seconds=5)

        self.assertEqual(store.get("task-1").status, TaskStatus.WORKING)
        statuses = [data["status"] for kind, data in store.events if kind == "status"]
        self.assertEqual(statuses, [TaskStatus.SUBMITTED, TaskStatus.WORKING])


class FallbackCasTests(unittest.TestCase):
    """Stores that do not implement ``save_if_status`` get get/compare/save."""

    def test_a_status_mismatch_refuses_the_write(self) -> None:
        store = _PlainStore(_working())
        service = TaskService(store, lambda task: {}, frozenset({"chat"}))

        self.assertFalse(service._save_if_status(_working(), TaskStatus.SUBMITTED))
        self.assertEqual(store.saves, 0)

    def test_a_matching_status_persists(self) -> None:
        store = _PlainStore(_working())
        service = TaskService(store, lambda task: {}, frozenset({"chat"}))
        finished = _working().transition(TaskStatus.COMPLETED)

        self.assertTrue(service._save_if_status(finished, TaskStatus.WORKING))
        self.assertEqual(store.saves, 1)
        self.assertEqual(store.get("task-1").status, TaskStatus.COMPLETED)

    def test_a_vanished_task_refuses_the_write(self) -> None:
        store = _PlainStore(None)
        service = TaskService(store, lambda task: {}, frozenset({"chat"}))

        self.assertFalse(service._save_if_status(_working(), TaskStatus.SUBMITTED))
        self.assertEqual(store.saves, 0)


if __name__ == "__main__":
    unittest.main()
