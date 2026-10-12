"""Desktop UI action continuation through the canonical ``AgentService``.

This test used to drive the removed ``adapters.legacy_agent`` bridge, which
built its own ``AgentService`` on a process-global store. It now exercises the
same lifecycle through an explicitly injected :class:`PendingActionStore`
backed by a temporary file, so no repository state is touched (ADR 0013).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_assistant.agents.models import AgentUiActionResultRequest
from ai_assistant.agents.pending_store import PendingActionStore
from ai_assistant.agents.service import AgentService


class UiContinuationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.service = AgentService(
            pending_actions=PendingActionStore(str(Path(self._tmp.name) / "pending.json")),
        )

    def test_follow_up_response_contains_cursor_after_previous_snapshot(self) -> None:
        request_id = "ui-test-current"
        previous_steps = [
            {"type": "tool_call", "tool": "application_action",
             "params": {"action": "viewer.load_2d", "request_id": request_id}},
            {"type": "tool_result", "tool": "application_action",
             "result": {"success": True, "action": "viewer.load_2d"}},
        ]
        self.service.pending_actions[request_id] = {
            "ui_ack": True,
            "params": {"action": "viewer.load_2d", "request_id": request_id},
            "task": "load 2d, 3d and dicom",
            "session_id": "ui-test-session",
            "steps": previous_steps,
            "messages": [],
            "next_actions": [{"action": "viewer.load_3d"}],
            "iteration": 0,
        }
        try:
            request = AgentUiActionResultRequest(
                request_id=request_id, success=True, result={"action": "viewer.load_2d"},
            )
            # ``ui_action_result`` persists through ``AgentService._save_pending``;
            # patch it so the test never writes pending state even to the temp file.
            with patch.object(self.service, "_save_pending"):
                response = self.service.ui_action_result(request)
            self.assertEqual(response["prior_step_count"], len(previous_steps))
            self.assertEqual(response["steps"][:len(previous_steps)], previous_steps)
            self.assertEqual(response["steps"][-1]["params"]["action"], "viewer.load_3d")
            next_id = response["request_id"]
            self.assertEqual(
                self.service.pending_actions[next_id]["steps"], response["steps"],
            )
        finally:
            self.service.pending_actions.pop(request_id, None)


if __name__ == "__main__":
    unittest.main()
