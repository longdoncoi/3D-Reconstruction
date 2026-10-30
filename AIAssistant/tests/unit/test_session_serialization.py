"""Per-session run serialization (architecture review, finding 3).

``AgentService.execute`` must let at most one run occupy a session at a time:
concurrent runs for one session share a checkpointer ``thread_id`` and clobber
the coordinator's running record.  Distinct sessions must stay independent —
the lock is per session, never global — and ``cancel`` must keep working while
a run holds its session's lock (it takes no session lock at all).

The two paths that *resume* a run — HITL approval and the desktop-ack
continuation — re-enter the same per-session lock: they share the checkpointer
``thread_id`` and the coordinator record with any concurrent execute of their
session, so letting them run outside the lock would recreate the very overlap
the lock exists to prevent.
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

import ai_assistant.agents.context as agents_context
from ai_assistant.agents.models import (
    AgentApproveRequest,
    AgentCancelRequest,
    AgentExecuteRequest,
    AgentUiActionResultRequest,
)
from ai_assistant.agents.service import AgentService
from ai_assistant.agents.session_locks import SessionLockPool
from ai_assistant.domain.errors import ServiceUnavailableError
from ai_assistant.domain.memory import DEFAULT_MEMORY_OWNER

# Generous on purpose: a gate timeout means the lock itself broke, not that
# the machine is slow.
_GATE_TIMEOUT = 30.0


class SessionSerializationTests(unittest.TestCase):
    def _service(self) -> AgentService:
        llm = MagicMock()
        llm.llm = MagicMock()
        return AgentService(llm_runtime=llm)

    def test_two_runs_for_one_session_never_overlap(self):
        service = self._service()
        first_inside = threading.Event()
        release_first = threading.Event()
        self.addCleanup(release_first.set)
        guard = threading.Lock()
        state = {"active": 0, "runs": 0, "max_active": 0}

        def fake_run(**kwargs):
            with guard:
                state["runs"] += 1
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            try:
                if state["runs"] == 1:
                    first_inside.set()
                    if not release_first.wait(timeout=_GATE_TIMEOUT):
                        msg = "gate was never released"
                        raise RuntimeError(msg)
            finally:
                with guard:
                    state["active"] -= 1
            return {"status": "completed"}

        with patch.object(agents_context, "run_langgraph_agent", side_effect=fake_run):
            first = threading.Thread(
                target=service.execute,
                args=(AgentExecuteRequest(task="first", session_id="s1"),),
                kwargs={"owner": DEFAULT_MEMORY_OWNER},
            )
            first.start()
            self.assertTrue(first_inside.wait(timeout=_GATE_TIMEOUT))
            second = threading.Thread(
                target=service.execute,
                args=(AgentExecuteRequest(task="second", session_id="s1"),),
                kwargs={"owner": DEFAULT_MEMORY_OWNER},
            )
            second.start()
            # Give the second run every chance to break in: while the first
            # still holds the session's lock it must stay outside.  A slow
            # scheduler can only make this assertion pass, never fail it.
            time.sleep(0.2)
            with guard:
                self.assertEqual(state["max_active"], 1)
            release_first.set()
            first.join(timeout=_GATE_TIMEOUT)
            second.join(timeout=_GATE_TIMEOUT)
        with guard:
            self.assertEqual((state["runs"], state["max_active"]), (2, 1))
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())

    def test_a_second_session_runs_while_the_first_is_held(self) -> None:
        service = self._service()
        first_inside = threading.Event()
        release_first = threading.Event()
        self.addCleanup(release_first.set)
        second_finished = threading.Event()

        def fake_run(**kwargs):
            if kwargs["session_id"] == "s1":
                first_inside.set()
                if not release_first.wait(timeout=_GATE_TIMEOUT):
                    msg = "gate was never released"
                    raise RuntimeError(msg)
                return {"status": "completed"}
            second_finished.set()
            return {"status": "completed"}

        with patch.object(agents_context, "run_langgraph_agent", side_effect=fake_run):
            first = threading.Thread(
                target=service.execute,
                args=(AgentExecuteRequest(task="blocked", session_id="s1"),),
                kwargs={"owner": DEFAULT_MEMORY_OWNER},
            )
            first.start()
            self.assertTrue(first_inside.wait(timeout=_GATE_TIMEOUT))
            second = threading.Thread(
                target=service.execute,
                args=(AgentExecuteRequest(task="parallel", session_id="s2"),),
                kwargs={"owner": DEFAULT_MEMORY_OWNER},
            )
            second.start()
            second.join(timeout=_GATE_TIMEOUT)
            # s2 completed while s1 is provably still inside its run: the
            # serialization is per session, not a global gate.
            self.assertTrue(second_finished.is_set())
            self.assertTrue(first.is_alive(), "s1 must still hold its run")
            release_first.set()
            first.join(timeout=_GATE_TIMEOUT)
        self.assertFalse(first.is_alive())

    def test_approve_resumes_under_the_sessions_run_lock(self) -> None:
        """An approval *resumes the run*: it serializes with execute.

        The resumed graph shares the checkpointer ``thread_id`` and the
        coordinator record with a concurrent execute of the same session, so
        the approval must queue on the session's lock instead of interleaving.
        """
        service = self._service()
        run_inside = threading.Event()
        release_run = threading.Event()
        self.addCleanup(release_run.set)
        guard = threading.Lock()
        state = {"active": 0, "max_active": 0, "approved": False}

        def enter() -> None:
            with guard:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])

        def leave() -> None:
            with guard:
                state["active"] -= 1

        def fake_run(**kwargs):
            enter()
            try:
                run_inside.set()
                if not release_run.wait(timeout=_GATE_TIMEOUT):
                    msg = "gate was never released"
                    raise RuntimeError(msg)
            finally:
                leave()
            return {"status": "completed"}

        def fake_approve(request):
            enter()
            try:
                with guard:
                    state["approved"] = True
            finally:
                leave()
            return {"status": "completed"}

        with patch.object(agents_context, "run_langgraph_agent", side_effect=fake_run):
            with patch.object(service._approval_service, "approve", side_effect=fake_approve):
                run_thread = threading.Thread(
                    target=service.execute,
                    args=(AgentExecuteRequest(task="block", session_id="s1"),),
                    kwargs={"owner": DEFAULT_MEMORY_OWNER},
                )
                run_thread.start()
                self.assertTrue(run_inside.wait(timeout=_GATE_TIMEOUT))
                approve_thread = threading.Thread(
                    target=service.approve,
                    args=(AgentApproveRequest(action_id="a-1", approved=True, session_id="s1"),),
                )
                approve_thread.start()
                # While the run provably holds the session's lock the resumed
                # run must stay outside it.
                time.sleep(0.2)
                with guard:
                    self.assertFalse(state["approved"])
                    self.assertEqual(state["max_active"], 1)
                release_run.set()
                run_thread.join(timeout=_GATE_TIMEOUT)
                approve_thread.join(timeout=_GATE_TIMEOUT)
        self.assertTrue(state["approved"])
        self.assertEqual(state["max_active"], 1)
        self.assertFalse(approve_thread.is_alive())

    def test_the_desktop_ack_resumes_under_the_owning_sessions_lock(self) -> None:
        """The ack request carries only the request id, so the owning session
        is read from the parked action and its lock is the one taken."""
        service = self._service()
        service.pending_actions["req-1"] = {"session_id": "s1", "ui_ack": True}
        run_inside = threading.Event()
        release_run = threading.Event()
        self.addCleanup(release_run.set)
        ack_done = threading.Event()

        def fake_run(**kwargs):
            run_inside.set()
            if not release_run.wait(timeout=_GATE_TIMEOUT):
                msg = "gate was never released"
                raise RuntimeError(msg)
            return {"status": "completed"}

        def fake_ack(request):
            ack_done.set()
            return {"status": "completed"}

        with patch.object(agents_context, "run_langgraph_agent", side_effect=fake_run):
            run_thread = threading.Thread(
                target=service.execute,
                args=(AgentExecuteRequest(task="block", session_id="s1"),),
                kwargs={"owner": DEFAULT_MEMORY_OWNER},
            )
            run_thread.start()
            self.assertTrue(run_inside.wait(timeout=_GATE_TIMEOUT))
            with patch.object(service._ui_action_service, "ui_action_result", side_effect=fake_ack):
                ack_thread = threading.Thread(
                    target=service.ui_action_result,
                    args=(AgentUiActionResultRequest(request_id="req-1", success=True),),
                )
                ack_thread.start()
                time.sleep(0.2)
                self.assertFalse(ack_done.is_set(), "the ack must queue on s1's run lock")
                release_run.set()
                ack_thread.join(timeout=_GATE_TIMEOUT)
            run_thread.join(timeout=_GATE_TIMEOUT)
        self.assertTrue(ack_done.is_set())
        self.assertFalse(ack_thread.is_alive())


class SessionLockPoolTests(unittest.TestCase):
    """The pool is bounded by live sessions, not by session ids ever seen."""

    def test_the_entry_is_dropped_after_the_last_user_leaves(self) -> None:
        pool = SessionLockPool()
        for index in range(10_000):
            with pool.hold(f"session-{index}"):
                self.assertEqual(pool.active_sessions(), 1)
        self.assertEqual(pool.active_sessions(), 0)

    def test_a_queued_thread_keeps_its_entry_until_it_runs(self) -> None:
        pool = SessionLockPool()
        first_inside = threading.Event()
        release_first = threading.Event()
        second_inside = threading.Event()
        self.addCleanup(release_first.set)

        def first() -> None:
            with pool.hold("s1"):
                first_inside.set()
                if not release_first.wait(timeout=_GATE_TIMEOUT):
                    msg = "gate was never released"
                    raise RuntimeError(msg)

        def second() -> None:
            with pool.hold("s1"):
                second_inside.set()

        first_thread = threading.Thread(target=first)
        first_thread.start()
        self.assertTrue(first_inside.wait(timeout=_GATE_TIMEOUT))
        second_thread = threading.Thread(target=second)
        second_thread.start()

        # The second thread increments the entry's user count *before* it
        # blocks. Wait until both users are registered; the entry must still
        # be the one the first thread holds (no second lock was created).
        deadline = time.monotonic() + _GATE_TIMEOUT
        while time.monotonic() < deadline:
            with pool._guard:
                entry = pool._entries.get("s1")
                if entry is not None and entry.users == 2:
                    break
            time.sleep(0.01)
        else:
            self.fail("the queued thread never registered as a user")
        self.assertFalse(second_inside.is_set())

        release_first.set()
        first_thread.join(timeout=_GATE_TIMEOUT)
        second_thread.join(timeout=_GATE_TIMEOUT)
        self.assertTrue(second_inside.is_set())
        self.assertEqual(pool.active_sessions(), 0)

    def test_reentry_uses_the_same_lock_and_distinct_sessions_do_not(self) -> None:
        pool = SessionLockPool()
        with pool.hold("s1"):
            with pool._guard:
                first = pool._entries["s1"].lock
            with pool.hold("s1"):
                with pool._guard:
                    self.assertIs(pool._entries["s1"].lock, first)
        with pool.hold("s2"):
            with pool._guard:
                self.assertIsNot(pool._entries["s2"].lock, first)


class SessionIdBoundsTests(unittest.TestCase):
    """A session id becomes a checkpointer thread id and a lock-pool key."""

    def test_execute_rejects_an_oversized_session_id(self) -> None:
        with self.assertRaises(ValidationError):
            AgentExecuteRequest(task="x", session_id="s" * 129)

    def test_approve_rejects_an_oversized_session_id(self) -> None:
        with self.assertRaises(ValidationError):
            AgentApproveRequest(action_id="a", approved=True, session_id="s" * 129)

    def test_cancel_rejects_an_oversized_session_id(self) -> None:
        with self.assertRaises(ValidationError):
            AgentCancelRequest(session_id="s" * 129)

    def test_a_128_character_session_id_is_accepted(self) -> None:
        self.assertEqual(
            AgentExecuteRequest(task="x", session_id="s" * 128).session_id,
            "s" * 128,
        )


class GlobalRunBudgetTests(unittest.TestCase):
    """ADR 0024: ``AGENT_MAX_CONCURRENT_RUNS`` bounds total concurrency.

    The per-session lock keeps one run per session, but a burst of sessions can
    still stack runs on the single local model. The global budget refuses extra
    runs with a retryable ``ServiceUnavailableError`` and releases the slot on
    every exit path, including a crashing run.
    """

    def _service(self, budget: int) -> AgentService:
        llm = MagicMock()
        llm.llm = MagicMock()
        return AgentService(llm_runtime=llm, max_concurrent_runs=budget)

    def test_slots_are_global_across_sessions_and_released_afterward(self) -> None:
        service = self._service(2)
        entered = threading.Semaphore(0)
        release = threading.Event()
        self.addCleanup(release.set)
        guard = threading.Lock()
        state = {"active": 0, "max_active": 0}

        def fake_run(**kwargs):
            with guard:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            entered.release()
            try:
                if not release.wait(timeout=_GATE_TIMEOUT):
                    msg = "gate was never released"
                    raise RuntimeError(msg)
            finally:
                with guard:
                    state["active"] -= 1
            return {"status": "completed"}

        with patch.object(agents_context, "run_langgraph_agent", side_effect=fake_run):
            threads = [
                threading.Thread(
                    target=service.execute,
                    args=(AgentExecuteRequest(task=f"t{idx}", session_id=f"s{idx}"),),
                    kwargs={"owner": DEFAULT_MEMORY_OWNER},
                )
                for idx in range(2)
            ]
            threads[0].start()
            self.assertTrue(entered.acquire(timeout=_GATE_TIMEOUT))
            threads[1].start()
            self.assertTrue(entered.acquire(timeout=_GATE_TIMEOUT))
            with guard:
                self.assertEqual(state["max_active"], 2)
            # A third session is refused while both slots are occupied…
            with self.assertRaises(ServiceUnavailableError):
                service.execute(AgentExecuteRequest(task="overflow", session_id="s3"),
                                owner=DEFAULT_MEMORY_OWNER)
            release.set()
            for thread in threads:
                thread.join(timeout=_GATE_TIMEOUT)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            # …and admitted once a slot frees up (the budget is released).
            result = service.execute(AgentExecuteRequest(task="after", session_id="s4"),
                                     owner=DEFAULT_MEMORY_OWNER)
        self.assertEqual(result["status"], "completed")

    def test_a_failed_run_releases_its_slot(self) -> None:
        service = self._service(1)
        with patch.object(
            agents_context, "run_langgraph_agent",
            side_effect=[RuntimeError("boom"), {"status": "completed"}],
        ):
            with self.assertRaises(RuntimeError):
                service.execute(AgentExecuteRequest(task="explode", session_id="s1"),
                                owner=DEFAULT_MEMORY_OWNER)
            result = service.execute(AgentExecuteRequest(task="retry", session_id="s2"),
                                     owner=DEFAULT_MEMORY_OWNER)
        self.assertEqual(result["status"], "completed")

    def test_a_resume_is_refused_while_the_budget_is_full(self) -> None:
        """ADR 0024 follows the ADR 0040 rule: a resume is a run.

        Both resume paths used to skip the global budget, so a burst of
        approvals/desktop acks could stack graphs on the single local model
        even while ``execute`` was capped.
        """
        service = self._service(1)
        slots = service._run_slots
        self.assertIsNotNone(slots)
        slots.acquire()
        self.addCleanup(slots.release)
        with self.assertRaises(ServiceUnavailableError):
            service.approve(
                AgentApproveRequest(action_id="a1", approved=True, session_id="s1")
            )
        with self.assertRaises(ServiceUnavailableError):
            service.ui_action_result(
                AgentUiActionResultRequest(request_id="r1", success=True, result={})
            )

    def test_a_failed_resume_releases_its_slot(self) -> None:
        """A resume that crashes must return its slot, like a fresh run does."""
        service = self._service(1)
        with patch.object(service._approval_service, "approve", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                service.approve(
                    AgentApproveRequest(action_id="a1", approved=True, session_id="s1")
                )
        with patch.object(
            service._ui_action_service, "ui_action_result", side_effect=RuntimeError("boom"),
        ):
            with self.assertRaises(RuntimeError):
                service.ui_action_result(
                    AgentUiActionResultRequest(request_id="r1", success=True, result={})
                )
        # Both failures returned their slot: the cap did not leak to the
        # next run (a leak would surface as a 503 after two bad resumes).
        self.assertTrue(service._run_slots.acquire(blocking=False))
        service._run_slots.release()

    def test_a_zero_budget_keeps_the_facade_unlimited(self) -> None:
        self.assertIsNone(self._service(0)._run_slots)


if __name__ == "__main__":
    unittest.main()
