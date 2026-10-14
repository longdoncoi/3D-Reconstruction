import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_assistant.orchestration.specialists.code import CodingTaskContext, instruction, is_coding_task
from ai_assistant.orchestration.supervisor import Specialist, authorise, delegate, verify_result


class MultiAgentPolicyTests(unittest.TestCase):
    def test_routes_read_tools_to_research(self):
        item = delegate("find this symbol", "session-1", "search_text", {"query": "Symbol"})
        self.assertEqual(item.specialist, Specialist.RESEARCH)
        self.assertTrue(authorise(item, False)[0])

    def test_routes_desktop_action_to_workflow(self):
        item = delegate("open mail", "session-1", "application_action", {"action": "mail.open"})
        self.assertEqual(item.specialist, Specialist.WORKFLOW)
        self.assertFalse(verify_result(item, {"pending_ui_ack": True})["passed"] is False)

    def test_code_tools_require_approval(self):
        item = delegate("change code", "session-1", "write_file", {"path": "x.py"})
        self.assertEqual(item.specialist, Specialist.CODE)
        self.assertFalse(authorise(item, False)[0])
        self.assertTrue(authorise(item, True)[0])

    def test_unknown_tool_is_denied(self):
        item = delegate("do something", "session-1", "unknown_tool", {})
        self.assertEqual(item.specialist, Specialist.SUPERVISOR)
        self.assertFalse(authorise(item, False)[0])

    def test_coding_task_uses_isolated_code_context_for_read_and_write(self):
        self.assertTrue(is_coding_task("fix bug and add unit test"))
        read = delegate("fix bug", "session-1", "read_file", prefer_code=True)
        self.assertEqual(read.specialist, Specialist.CODE)
        self.assertTrue(authorise(read, False)[0])
        write = delegate("fix bug", "session-1", "patch_file", prefer_code=True)
        self.assertFalse(authorise(write, False)[0])
        self.assertIn("CODING AGENT", instruction(CodingTaskContext("fix bug", "en", ".")))


if __name__ == "__main__":
    unittest.main()
