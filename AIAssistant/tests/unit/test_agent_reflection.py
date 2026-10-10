import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_assistant.orchestration.supervisor import delegate, reflect_result, specialist_instruction, verify_result


class AgentReflectionTests(unittest.TestCase):
    def test_failed_tool_requires_revision(self):
        item = delegate("inspect", "session", "read_file", {"path": "missing.txt"})
        review = reflect_result(item, {"error": "not found"}, {"passed": False, "reason": "tool failed"})
        self.assertFalse(review["passed"])
        self.assertEqual(review["decision"], "revise")

    def test_successful_tool_continues(self):
        item = delegate("search", "session", "search_text", {"query": "main"})
        review = reflect_result(item, {"matches": []}, {"passed": True, "reason": "valid"})
        self.assertTrue(review["passed"])
        self.assertEqual(review["decision"], "continue")

    def test_each_specialist_has_a_compact_handoff(self):
        item = delegate("verify", "session", "validate_file", {"path": "x.py"})
        self.assertIn("Independently", specialist_instruction(item))

    def test_empty_discovery_result_is_not_accepted_as_evidence(self):
        item = delegate("inspect", "session", "search_text", {"query": "symbol"})
        review = verify_result(item, {"query": "symbol", "count": 0, "results": []})
        self.assertFalse(review["passed"])


if __name__ == "__main__":
    unittest.main()
