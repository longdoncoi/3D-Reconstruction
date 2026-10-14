"""Unit tests for the shared action-identifier helper.

HITL approval binds an ``action_id`` to one specific desktop/UI action, so the
identifier has to be unpredictable and URL-safe: a guessable id would let one
desktop session mint an approval another session could consume.
"""
from __future__ import annotations

import unittest

from ai_assistant.agents.ids import generate_action_id


class GenerateActionIdTests(unittest.TestCase):
    def test_ids_are_unpredictable_and_url_safe(self) -> None:
        first = generate_action_id()
        second = generate_action_id()
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 22)
        for character in first:
            self.assertTrue(character.isalnum() or character in "-_", character)


if __name__ == "__main__":
    unittest.main()
