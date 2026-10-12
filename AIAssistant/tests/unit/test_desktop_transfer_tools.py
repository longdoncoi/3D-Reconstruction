"""Coverage for the transfer and desktop builtin tools.

The desktop action tool parks UI work for acknowledgement instead of
executing it, and the transfer tools hand a task to a sibling specialist
with a status marker. (Moved verbatim from ``test_coverage_boost2.py``
during the coverage-file split.)
"""
from __future__ import annotations

import unittest

from ai_assistant.tools.builtin import desktop_tools, transfer_tools


class BuiltinToolTests(unittest.TestCase):
    """Coverage for the transfer and desktop builtin tools."""

    def test_desktop_action_valid(self) -> None:
        result = desktop_tools.tool_application_action({"action": "viewer.load_2d"})
        self.assertTrue(result["pending_ui_ack"])
        self.assertEqual(result["action"], "viewer.load_2d")

    def test_desktop_action_invalid(self) -> None:
        result = desktop_tools.tool_application_action({})
        self.assertIn("error", result)

    def test_transfer_tools(self) -> None:
        self.assertEqual(transfer_tools.tool_transfer_to_code_agent({"intent": "fix"})["status"], "transferred_to_code")
        self.assertEqual(
            transfer_tools.tool_transfer_to_toolapp_agent({"intent": "ui"})["status"], "transferred_to_toolapp"
        )
        self.assertEqual(
            transfer_tools.tool_transfer_to_chatbot_agent({"intent": "chat"})["status"], "transferred_to_chatbot"
        )


if __name__ == "__main__":
    unittest.main()
