import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_assistant.orchestration.specialists.chatbot import ChatbotAgent
from ai_assistant.orchestration.specialists.desktop_workflow import ToolAppAgent


class _FakeRag:
    def get_context(self, query, query_image_b64=None):
        return (f"docs:{query}", "code", ["image"] if query_image_b64 else [])


class _FakeLlm:
    is_vision_model = False

    @staticmethod
    def _is_character_query(_query):
        return True

    @staticmethod
    def build_text_messages(messages, doc, code, suppress_citations, language):
        return [{"messages": messages, "doc": doc, "code": code,
                 "suppress": suppress_citations, "language": language}]

    @staticmethod
    def _strip_reference_citations_for_character_answer(answer):
        return answer.replace("[1]", "")


class ChatbotAgentTests(unittest.TestCase):
    def test_chatbot_owns_rag_prompt_and_answer_cleanup(self):
        agent = ChatbotAgent(_FakeLlm(), _FakeRag())
        prepared, metadata = agent.build_messages([{"role": "user", "content": "hello"}], "hello", None, "vi")
        self.assertEqual(prepared[0]["doc"], "docs:hello")
        self.assertTrue(metadata["suppress_citations"])
        self.assertEqual(agent.clean_answer("answer[1]", metadata, "stop"), "answer")

    def test_toolapp_agent_owns_desktop_matching(self):
        agent = ToolAppAgent(
            lambda task: {"action": "viewer.load_2d"} if "2d" in task else None,
            lambda task: [{"action": "viewer.load_2d"}, {"action": "viewer.load_3d"}]
            if "both" in task else None,
        )
        action, sequence = agent.match("both")
        self.assertEqual(action["action"], "viewer.load_2d")
        self.assertEqual(len(sequence), 2)
        self.assertIn("application_action", ToolAppAgent.instruction())


if __name__ == "__main__":
    unittest.main()
