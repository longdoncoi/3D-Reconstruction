"""Guard-path coverage for the in-process ``TaskCoordinator`` registry.

The coordinator is what makes a cancel that races the worker thread actually
stick, and what keeps a long-lived desktop session from leaking records. Both
behaviours are load-bearing for the A2A/Qt surfaces, so they are pinned here:

* pending cancel survives ``start`` (the worker registering its run);
* terminal records are evicted oldest-first while live runs are never evicted;
* cancelling an already finished run is a no-op instead of resurrecting it;
* the cancel-to-stop window (SCORECARD #4) is reported through the injected
  recovery observer, and only for runs that were actually cancelled.
"""
from __future__ import annotations

import time
import unittest

from ai_assistant.application.coordination import TaskCoordinator


class PendingCancelTests(unittest.TestCase):
    def test_cancel_before_start_survives_the_run_registering(self) -> None:
        """``POST /tasks/{id}:cancel`` racing the worker must not be lost."""
        coordinator = TaskCoordinator()
        parked = coordinator.cancel("sess-1", create_if_missing=True)
        self.assertIsNotNone(parked)
        self.assertTrue(parked["cancelled"])

        started = coordinator.start("sess-1")
        self.assertTrue(started.cancelled)
        self.assertEqual(started.status, "cancelled")
        self.assertTrue(coordinator.is_cancelled("sess-1"))

    def test_cancel_without_request_id_matches_the_sessions_run(self) -> None:
        coordinator = TaskCoordinator()
        coordinator.start("sess-2", "req-1")
        snapshot = coordinator.cancel("sess-2")
        self.assertIsNotNone(snapshot)
        self.assertTrue(snapshot["cancelled"])
        self.assertTrue(coordinator.is_cancelled("sess-2", "req-1"))

    def test_unknown_session_without_the_opt_in_returns_none(self) -> None:
        coordinator = TaskCoordinator()
        self.assertIsNone(coordinator.cancel("no-such-session"))

    def test_cancelling_a_finished_run_is_a_no_op(self) -> None:
        """A terminal, non-cancelled record must stay terminal."""
        coordinator = TaskCoordinator()
        coordinator.finish("sess-3", success=True)
        self.assertIsNone(coordinator.cancel("sess-3"))
        self.assertEqual(
            coordinator.update("sess-3")["status"],
            "succeeded",
        )


class EvictionTests(unittest.TestCase):
    def test_terminal_records_are_evicted_oldest_first(self) -> None:
        max_records = 16
        coordinator = TaskCoordinator(max_records=max_records)
        for index in range(20):
            coordinator.update(f"sess-{index}", status="succeeded")

        keys = [TaskCoordinator.key(f"sess-{index}") for index in range(20)]
        surviving = [key for key in keys if key in coordinator._records]
        self.assertLessEqual(len(coordinator._records), max_records)
        # Insertion order is preserved, so the four oldest are the evicted ones.
        self.assertEqual(surviving, keys[4:])

    def test_live_runs_are_never_evicted(self) -> None:
        coordinator = TaskCoordinator(max_records=16)
        coordinator.update("live", status="running")
        for index in range(20):
            coordinator.update(f"sess-{index}", status="succeeded")

        self.assertEqual(
            coordinator._records[TaskCoordinator.key("live")].status,
            "running",
        )
        self.assertLessEqual(len(coordinator._records), 16)

    def test_eviction_gives_up_when_only_live_runs_remain(self) -> None:
        """Over capacity with nothing terminal: the registry keeps growing."""
        coordinator = TaskCoordinator(max_records=16)
        for index in range(17):
            coordinator.update(f"live-{index}", status="running")
        self.assertEqual(len(coordinator._records), 17)
        self.assertTrue(
            all(record.status == "running" for record in coordinator._records.values())
        )

    def test_eviction_keeps_deleting_until_it_fits(self) -> None:
        """Parked cancels arrive without a pass through the evictor."""
        coordinator = TaskCoordinator(max_records=16)
        for index in range(20):
            self.assertIsNotNone(
                coordinator.cancel(f"parked-{index}", create_if_missing=True)
            )
        self.assertEqual(len(coordinator._records), 20)

        coordinator.start("fresh-run")
        self.assertLessEqual(len(coordinator._records), 16)
        # Oldest parked entries are the ones that went; the newest survived.
        self.assertFalse(coordinator.is_cancelled("parked-0"))
        self.assertTrue(coordinator.is_cancelled("parked-19"))

    def test_cancelled_runs_report_through_finish(self) -> None:
        coordinator = TaskCoordinator()
        coordinator.start("sess-4")
        coordinator.cancel("sess-4")
        time.sleep(0.001)
        self.assertEqual(coordinator.finish("sess-4")["status"], "cancelled")


class RecoveryObserverTests(unittest.TestCase):
    """SCORECARD #4 — cancel request to runner stop, in seconds."""

    def test_a_cancelled_run_reports_its_recovery_time(self) -> None:
        observed: list[float] = []
        coordinator = TaskCoordinator(recovery_observer=observed.append)
        coordinator.start("sess-r1")
        coordinator.cancel("sess-r1")
        time.sleep(0.001)

        coordinator.finish("sess-r1")

        self.assertEqual(len(observed), 1)
        self.assertGreaterEqual(observed[0], 0.001)

    def test_a_run_that_was_never_cancelled_reports_nothing(self) -> None:
        observed: list[float] = []
        coordinator = TaskCoordinator(recovery_observer=observed.append)
        coordinator.start("sess-r2")

        coordinator.finish("sess-r2", success=True)

        self.assertEqual(observed, [])

    def test_a_cancel_that_raced_the_worker_still_reports_recovery(self) -> None:
        """The pending-cancel carry-over must keep the cancel timestamp."""
        observed: list[float] = []
        coordinator = TaskCoordinator(recovery_observer=observed.append)
        coordinator.cancel("sess-r3", create_if_missing=True)
        time.sleep(0.001)

        coordinator.start("sess-r3")
        coordinator.finish("sess-r3")

        self.assertEqual(len(observed), 1)
        self.assertGreaterEqual(observed[0], 0.001)

    def test_a_run_without_a_record_reports_nothing(self) -> None:
        observed: list[float] = []
        coordinator = TaskCoordinator(recovery_observer=observed.append)

        coordinator._observe_recovery("never-started")

        self.assertEqual(observed, [])


if __name__ == "__main__":
    unittest.main()
