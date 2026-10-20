import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_assistant.orchestration.specialists.chatbot import ChatbotAgent


class _FakeRag:
    def get_context(self, query, query_image_b64=None):
        return (f"docs:{query}", "code", ["image"] if query_image_b64 else [])


class _FakeLlm:
    is_vision_model = False


class _FakePrompts:
    """The prompt builder is a separate collaborator from the LLM handle."""

    @staticmethod
    def is_character_query(_query):
        return True

    @staticmethod
    def build_messages(messages, doc, code, suppress_citations, language):
        return [{"messages": messages, "doc": doc, "code": code,
                 "suppress": suppress_citations, "language": language}]

    @staticmethod
    def strip_citations(answer):
        return answer.replace("[1]", "")


class ChatbotAgentTests(unittest.TestCase):
    def test_chatbot_owns_rag_prompt_and_answer_cleanup(self):
        agent = ChatbotAgent(_FakeLlm(), _FakeRag(), _FakePrompts())
        prepared, metadata = agent.build_messages([{"role": "user", "content": "hello"}], "hello", None, "vi")
        self.assertEqual(prepared[0]["doc"], "docs:hello")
        self.assertTrue(metadata["suppress_citations"])
        self.assertEqual(agent.clean_answer("answer[1]", metadata, "stop"), "answer")


class RealWiringTests(unittest.TestCase):
    """The production objects must work together — not only the mocks.

    The chat route used to reach the prompt-building methods through the LLM
    runtime handle. That worked while both were one object and broke the
    endpoint outright once they were split, and **every** other case in this
    file mocks the collaborator, so the mismatch was invisible to the suite.
    These cases wire the real handle, the real prompt builder and the real
    composition-root holder, so a missing method fails here instead of on the
    first chat request.
    """

    @staticmethod
    def _llm():
        from ai_assistant.bootstrap.modules import LlmModule

        llm = LlmModule(registry=MagicMock(), paths=MagicMock(), model_idx=0,
                        use_vision_llm=False)
        llm.llm = MagicMock()
        llm.is_vision_model = False
        return llm

    def test_the_real_handle_and_prompt_builder_build_a_message_list(self):
        from ai_assistant.llm.prompts import DefaultPromptBuilder

        agent = ChatbotAgent(self._llm(), _FakeRag(), DefaultPromptBuilder())
        prepared, metadata = agent.build_messages(
            [{"role": "user", "content": "hello"}], "hello", None, "vi")

        self.assertTrue(prepared)
        self.assertIn("suppress_citations", metadata)

    def test_the_composition_root_holder_builds_a_working_agent(self):
        from ai_assistant.bootstrap.modules import ChatbotAgentHolder
        from ai_assistant.llm.prompts import DefaultPromptBuilder

        holder = ChatbotAgentHolder(self._llm(), _FakeRag(), DefaultPromptBuilder())
        prepared, _metadata = holder.get().build_messages(
            [{"role": "user", "content": "hello"}], "hello", None, "vi")

        self.assertTrue(prepared)

    def test_the_vision_arm_works_with_the_real_builder(self):
        from ai_assistant.llm.prompts import DefaultPromptBuilder

        llm = self._llm()
        llm.is_vision_model = True
        agent = ChatbotAgent(llm, _FakeRag(), DefaultPromptBuilder())
        prepared, _metadata = agent.build_messages(
            [{"role": "user", "content": "what is this"}], "what is this", "img-uri", "vi")

        self.assertTrue(prepared)


if __name__ == "__main__":
    unittest.main()
