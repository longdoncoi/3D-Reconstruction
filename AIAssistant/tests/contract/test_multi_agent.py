import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_assistant.orchestration.specialists.code import CodingTaskContext, instruction, is_coding_task
from ai_assistant.orchestration.supervisor import Specialist, authorise, delegate, route_task, verify_result


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

    def test_code_symbol_citation_routes_to_coding_agent(self):
        task = "Tr\u00edch d\u1eab\u006e h\u00e0m ReconstructionPlugin::onRunReconstruction()"
        self.assertEqual(route_task(task), Specialist.CODE)

    def test_coding_task_wins_over_matching_desktop_action(self):
        task = "Fix l\u1ed7i khi t\u1ea3i \u1ea3nh DICOM, b\u1ed5 sung unit test v\u00e0 ki\u1ec3m tra build"
        self.assertEqual(route_task(task), Specialist.CODE)

    def test_feature_request_routes_to_coding_agent(self):
        task = "B\u1ed5 sung ch\u1ee9c n\u0103ng export reconstruction sang file OBJ"
        self.assertEqual(route_task(task), Specialist.CODE)

    def test_analysis_and_debug_intents_route_to_coding_agent(self):
        self.assertEqual(
            route_task("Ph\u00e2n t\u00edch v\u00e0 gi\u1ea3i th\u00edch ch\u1ee9c n\u0103ng c\u1ee7a h\u00e0m"),
            Specialist.CODE,
        )
        self.assertEqual(route_task("S\u1eeda l\u1ed7i trong ch\u1ee9c n\u0103ng DICOM"), Specialist.CODE)


if __name__ == "__main__":
    unittest.main()
