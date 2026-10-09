"""Durability contract: pending approvals survive a process restart (ADR 0003).

The A2A lifecycle defers to the shared LangGraph engine, so an approval pause is
held in the engine's pending-action store. If that store is not rehydrated on
startup, a resume after a restart fails even though the on-disk state exists.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_assistant.adapters.persistence import PendingActionStore
from ai_assistant.agents.service import AgentService


class PendingDurabilityTests(unittest.TestCase):
    def test_service_rehydrates_persisted_pending_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending_agent_actions.json"
            first = PendingActionStore(path)
            first["act-1"] = {
                "tool": "write_file", "params": {"path": "a.txt"},
                "session_id": "s1", "task": "write", "messages": [],
                "steps": [], "iteration": 0, "created_at": 1.0,
            }
            first.save()

            # A fresh store simulates a new process starting up.
            second = PendingActionStore(path)
            service = AgentService(pending_actions=second)

            self.assertIn("act-1", service.pending_actions)
            self.assertEqual(service.pending_actions["act-1"]["tool"], "write_file")

    def test_startup_tolerates_a_missing_store(self):
        with tempfile.TemporaryDirectory() as directory:
            service = AgentService(pending_actions=PendingActionStore(Path(directory) / "absent.json"))
            self.assertEqual(len(service.pending_actions), 0)


if __name__ == "__main__":
    unittest.main()
