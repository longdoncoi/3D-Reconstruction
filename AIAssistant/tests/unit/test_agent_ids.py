"""Unit tests for the shared action-identifier helper.

HITL approval binds an ``action_id`` to one specific desktop/UI action, so the
identifier has to be unpredictable and the implicit ``agent_default`` session —
which is what an empty Qt ``session_id`` degrades to — must never mint one:
several concurrent desktop sessions would otherwise share a single approval
queue.
"""
from __future__ import annotations

import unittest

from ai_assistant.agents.ids import generate_action_id


class GenerateActionIdTests(unittest.TestCase):
    def test_ids_are_unpredictable_and_url_safe(self) -> None:
        first = generate_action_id("session-1")
        second = generate_action_id("session-1")
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 22)
        for character in first:
            self.assertTrue(character.isalnum() or character in "-_", character)

    def test_empty_session_id_is_accepted(self) -> None:
        self.assertTrue(generate_action_id())

    def test_the_implicit_session_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            generate_action_id("agent_default")
        self.assertEqual(str(caught.exception), "bad_session_id")


if __name__ == "__main__":
    unittest.main()
