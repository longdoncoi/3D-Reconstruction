import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_assistant.adapters import legacy_agent as agent_impl
from ai_assistant.adapters import legacy_agent as agent_module


class UiContinuationTests(unittest.TestCase):
    def test_follow_up_response_contains_cursor_after_previous_snapshot(self):
        request_id = "ui-test-current"
        previous_steps = [
            {"type": "tool_call", "tool": "application_action",
             "params": {"action": "viewer.load_2d", "request_id": request_id}},
            {"type": "tool_result", "tool": "application_action",
             "result": {"success": True, "action": "viewer.load_2d"}},
        ]
        agent_module._pending_actions[request_id] = {
            "ui_ack": True,
            "params": {"action": "viewer.load_2d", "request_id": request_id},
            "task": "load 2d, 3d and dicom",
            "session_id": "ui-test-session",
            "steps": previous_steps,
            "messages": [],
            "next_actions": [{"action": "viewer.load_3d"}],
            "iteration": 0,
        }
        next_id = None
        try:
            request = agent_module.AgentUiActionResultRequest(
                request_id=request_id, success=True, result={"action": "viewer.load_2d"},
            )
            # ``ui_action_result`` persists through ``AgentService._save_pending``;
            # patch that method (not the module-level shim) so this test never
            # rewrites the real AIAssistant/pending_agent_actions.json file.
            with patch.object(agent_impl._default_service, "_save_pending"):
                response = agent_module.agent_ui_action_result(request)
            self.assertEqual(response["prior_step_count"], len(previous_steps))
            self.assertEqual(response["steps"][:len(previous_steps)], previous_steps)
            self.assertEqual(response["steps"][-1]["params"]["action"], "viewer.load_3d")
            next_id = response["request_id"]
            self.assertEqual(
                agent_module._pending_actions[next_id]["steps"], response["steps"],
            )
        finally:
            agent_module._pending_actions.pop(request_id, None)
            if next_id:
                agent_module._pending_actions.pop(next_id, None)


if __name__ == "__main__":
    unittest.main()
