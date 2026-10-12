"""Guard-path coverage for history compaction and citation formatting.

``summarize_messages`` protects roles and recency first, then enforces the
total budget by shrinking from the front: the no-user-message shape, the
middle-summary injection and the over-budget shrink loop each need a wide
history to appear. ``format_code_citation`` skips errored reads and answers
``None`` when nothing citable survived — both arms are pinned.
"""
from __future__ import annotations

import unittest

from ai_assistant.orchestration.nodes.summarize import (
    format_code_citation,
    summarize_messages,
)


def _history(count: int, size: int = 500, role: str = "assistant") -> list[dict]:
    return [{"role": role, "content": "x" * size} for _ in range(count)]


class WideHistoryTests(unittest.TestCase):
    def test_a_history_without_a_user_turn_keeps_system_and_recency(self):
        """No user slot to fill, but the middle summary still appears and the
        total still fits the budget."""
        messages = [{"role": "system", "content": "sys"}, *_history(24)]
        result = summarize_messages(messages)
        roles = [m["role"] for m in result]
        self.assertIn("system", roles)
        self.assertTrue(any("Tom tat" in m.get("content", "") for m in result))
        total = sum(len(m.get("content", "")) for m in result)
        self.assertLessEqual(total, 11000)

    def test_a_short_history_passes_through(self):
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(summarize_messages(messages), messages)

    def test_an_unshrinkable_anchor_keeps_the_total_over_budget(self):
        """With no system/user turn, the middle summary lands at index 0 —
        which the shrink loop never touches — so the loop trims every recent
        turn and still exits over budget instead of looping forever."""
        result = summarize_messages(_history(60))
        self.assertEqual(result[0]["role"], "system")
        self.assertIn("Tom tat", result[0]["content"])
        total = sum(len(m.get("content", "")) for m in result)
        self.assertGreater(total, 11000)


class CitationTests(unittest.TestCase):
    def test_an_errored_read_is_skipped(self):
        messages = [{"role": "user", "content": "trich dan ham main"}]
        steps = [{"type": "tool_result", "tool": "read_file", "result": {"error": "boom"}}]
        self.assertIsNone(format_code_citation(messages, steps))

    def test_a_non_citation_question_returns_none(self):
        messages = [{"role": "user", "content": "what does this do"}]
        steps = [{"type": "tool_result", "tool": "read_file",
                  "result": {"content": "def main(): pass", "path": "a.py"}}]
        self.assertIsNone(format_code_citation(messages, steps))

    def test_a_citable_read_produces_a_fenced_citation(self):
        messages = [{"role": "user", "content": "trich dan ham main"}]
        steps = [{"type": "tool_result", "tool": "read_file",
                  "result": {"content": "def main(): pass", "path": "a.py"}}]
        citation = format_code_citation(messages, steps)
        self.assertIn("```python", citation)
        self.assertIn("def main(): pass", citation)


if __name__ == "__main__":
    unittest.main()
