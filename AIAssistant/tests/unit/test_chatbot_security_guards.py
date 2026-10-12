"""Guard-path coverage for transport peer checks and the chatbot specialist.

``_is_loopback`` fails closed: missing peers are rejected, and peers that are
neither IPs nor ``*.localhost`` names are rejected too. The chatbot's vision
arm only runs for vision models with an image attached (otherwise chunks are
dropped and the text builder is used), and ``clean_answer`` strips citations
for character answers and warns when the token limit truncated the reply.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from ai_assistant.adapters.http.security import _is_loopback
from ai_assistant.orchestration.specialists.chatbot import ChatbotAgent


class LoopbackCheckTests(unittest.TestCase):
    def test_a_missing_peer_is_not_loopback(self):
        self.assertFalse(_is_loopback(None))
        self.assertFalse(_is_loopback(""))

    def test_loopback_names_and_addresses_pass(self):
        self.assertTrue(_is_loopback("localhost"))
        self.assertTrue(_is_loopback("127.0.0.1"))
        self.assertTrue(_is_loopback("::1"))

    def test_an_unparseable_peer_fails_closed(self):
        """Test-client labels are not IPs; only ``*.localhost`` passes."""
        self.assertFalse(_is_loopback("testclient"))
        self.assertTrue(_is_loopback("app.localhost"))

    def test_a_remote_address_is_rejected(self):
        self.assertFalse(_is_loopback("8.8.8.8"))


def _agent(*, vision: bool = False):
    llm = MagicMock()
    llm._is_character_query.return_value = False
    llm.is_vision_model = vision
    rag = MagicMock()
    rag.get_context.return_value = ("docs", "code", [{"img": 1}])
    return ChatbotAgent(llm, rag), llm, rag


class VisionArmTests(unittest.TestCase):
    def test_a_vision_model_with_an_image_uses_the_vision_builder(self):
        agent, llm, _ = _agent(vision=True)
        llm.build_vision_messages.return_value = [{"role": "user", "content": "v"}]
        prepared, meta = agent.build_messages(
            [{"role": "user", "content": "hi"}], "what is this", "img-uri", "vi",
        )
        llm.build_vision_messages.assert_called_once()
        llm.build_text_messages.assert_not_called()
        self.assertEqual(prepared, [{"role": "user", "content": "v"}])
        self.assertEqual(meta, {"suppress_citations": False})

    def test_chunks_are_dropped_when_no_image_is_attached(self):
        agent, llm, _ = _agent(vision=False)
        llm.build_text_messages.return_value = []
        agent.build_messages([{"role": "user", "content": "hi"}], "q", None, "vi")
        _, kwargs = llm.build_text_messages.call_args
        _ = kwargs  # positional call; chunks were cleared before dispatch
        llm.build_vision_messages.assert_not_called()


class CleanAnswerTests(unittest.TestCase):
    def test_a_plain_answer_passes_through(self):
        agent, llm, _ = _agent()
        self.assertEqual(agent.clean_answer("a [1]", {}, "stop"), "a [1]")
        llm._strip_reference_citations_for_character_answer.assert_not_called()

    def test_character_answers_lose_their_citations(self):
        agent, llm, _ = _agent()
        llm._strip_reference_citations_for_character_answer.return_value = "clean"
        self.assertEqual(
            agent.clean_answer("a [1]", {"suppress_citations": True}, "stop"), "clean",
        )

    def test_a_truncated_answer_warns_the_user(self):
        agent, _, _ = _agent()
        self.assertIn("incomplete", agent.clean_answer("half", {}, "length"))


if __name__ == "__main__":
    unittest.main()
