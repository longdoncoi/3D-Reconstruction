"""Per-session run serialization (architecture review, finding 3).

``AgentService.execute`` must let at most one run occupy a session at a time:
concurrent runs for one session share a checkpointer ``thread_id`` and clobber
the coordinator's running record.  Distinct sessions must stay independent —
the lock is per session, never global — and ``cancel``/``approve`` must keep
working while a run holds its session's lock (they take no session lock at all).
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import ai_assistant.agents.context as agents_context
from ai_assistant.agents.models import AgentExecuteRequest
from ai_assistant.agents.service import AgentService

# Generous on purpose: a gate timeout means the lock itself broke, not that
# the machine is slow.
_GATE_TIMEOUT = 30.0


class SessionSerializationTests(unittest.TestCase):
    def _service(self) -> AgentService:
        llm = MagicMock()
        llm.llm = MagicMock()
        return AgentService(llm_runtime=llm)

    def test_the_lock_is_reused_per_session_and_never_shared(self):
        service = self._service()
        self.assertIs(service._session_lock("s1"), service._session_lock("s1"))
        self.assertIsNot(service._session_lock("s1"), service._session_lock("s2"))

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
            )
            first.start()
            self.assertTrue(first_inside.wait(timeout=_GATE_TIMEOUT))
            second = threading.Thread(
                target=service.execute,
                args=(AgentExecuteRequest(task="second", session_id="s1"),),
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

    def test_a_second_session_runs_while_the_first_is_held(self):
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
            )
            first.start()
            self.assertTrue(first_inside.wait(timeout=_GATE_TIMEOUT))
            second = threading.Thread(
                target=service.execute,
                args=(AgentExecuteRequest(task="parallel", session_id="s2"),),
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


if __name__ == "__main__":
    unittest.main()
