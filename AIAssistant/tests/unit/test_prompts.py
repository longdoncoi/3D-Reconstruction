"""Unit tests for prompt construction and history management."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_assistant.llm.prompts import DefaultPromptBuilder, estimate_tokens, trim_history

CHARS_PER_TOKEN = 2.2


class EstimateTokensTests(unittest.TestCase):
    def test_default_chars_per_token(self) -> None:
        self.assertEqual(estimate_tokens("hello world"), int(len("hello world") / CHARS_PER_TOKEN))
        self.assertEqual(estimate_tokens("hello world", 2.0), int(len("hello world") / 2.0))

    def test_empty_text(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)

    def test_returns_floor_int(self) -> None:
        self.assertIsInstance(estimate_tokens("a" * 100), int)


class TrimHistoryTests(unittest.TestCase):
    def test_under_budget_is_unchanged(self) -> None:
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        result = trim_history(messages, max_tokens=1000)
        self.assertEqual(result, messages)

    def test_drops_oldest_when_over_budget(self) -> None:
        messages = []
        for i in range(20):
            messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 100})
        result = trim_history(messages, max_tokens=300)
        self.assertLessEqual(len(result), len(messages))
        # The most recent message must survive.
        self.assertEqual(result[-1], messages[-1])
        budget = sum(estimate_tokens(m["content"]) for m in result)
        # While > 1 messages remain, trimming stops only when within budget.
        self.assertTrue(budget <= 300 or len(result) <= 1)

    def test_keeps_at_least_one_message(self) -> None:
        messages = [{"role": "user", "content": "z" * 5000}]
        result = trim_history(messages, max_tokens=10)
        self.assertEqual(result, messages)

    def test_non_string_content_is_skipped(self) -> None:
        messages = [
            {"role": "user", "content": "a" * 200},
            {"role": "user", "content": {"parts": ["x"]}},
        ]
        result = trim_history(messages, max_tokens=50)
        # The oldest string message is dropped; the non-string message survives
        # (the caller keeps at least one message).
        self.assertEqual(result, messages[1:])

    def test_does_not_mutate_input(self) -> None:
        messages = [{"role": "user", "content": "a" * 300}, {"role": "user", "content": "b" * 300}]
        original = list(messages)
        trim_history(messages, max_tokens=100)
        self.assertEqual(messages, original)


class PromptBuilderCharacterQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = DefaultPromptBuilder()

    def test_character_marker_words(self) -> None:
        for query in (
            "Nhân vật John trong dự án là ai?",
            "danh sách thành viên của nhóm",
            "ai là kỹ sư phát triển?",
            "team lead của dự án là ai",
            "đội ngũ phát triển gồm những ai",
        ):
            self.assertTrue(self.builder.is_character_query(query), query)

    def test_name_plus_role_qualifies(self) -> None:
        self.assertTrue(self.builder.is_character_query("Vai trò của Carpenter trong dự án"))
        self.assertTrue(self.builder.is_character_query("John có vai trò gì?"))

    def test_non_character_queries(self) -> None:
        for query in (
            "Cách build module RAG như thế nào?",
            "giải thích khái niệm point cloud",
            "hướng dẫn cài đặt Qt Creator",
            "tại sao đồ án chậm",
            "",
            None,
        ):
            self.assertFalse(self.builder.is_character_query(query), query)


class PromptBuilderCitationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = DefaultPromptBuilder()

    def test_removes_numeric_citations(self) -> None:
        cleaned = self.builder.strip_citations("Kết quả theo [1] và [2, 3] xác nhận điều đó.")
        self.assertNotIn("[1]", cleaned)
        self.assertNotIn("[2, 3]", cleaned)
        self.assertIn("Kết quả theo và xác nhận điều đó.", cleaned)

    def test_removes_source_intro_lines(self) -> None:
        cleaned = self.builder.strip_citations(
            "Theo tài liệu: hướngdẫn.md.\nNội dung trả lời chính [1].\n"
        )
        self.assertNotIn("Theo tài liệu", cleaned)

    def test_removes_source_labels(self) -> None:
        cleaned = self.builder.strip_citations("Tài liệu tham khảo: [1] abc.md\nNội dung.")
        self.assertNotIn("Tài liệu tham khảo", cleaned)
        self.assertIn("Nội dung.", cleaned)

    def test_collapses_blank_lines_and_whitespace(self) -> None:
        cleaned = self.builder.strip_citations("Đoạn 1.\n\n\n\nĐoạn 2.\n")
        self.assertNotIn("\n\n\n", cleaned)

    def test_empty_answer(self) -> None:
        self.assertEqual(self.builder.strip_citations(""), "")
        self.assertEqual(self.builder.strip_citations(None), None)


class PromptBuilderMessageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = DefaultPromptBuilder()

    def test_build_messages_shapes_conversation(self) -> None:
        messages = [
            {"role": "user", "content": "xin chào"},
            {"role": "assistant", "content": "chào bạn"},
            {"role": "user", "content": "giới thiệu dự án"},
        ]
        result = self.builder.build_messages(messages, doc_ctx="Tài liệu: ...", code_ctx="")
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[-1]["role"], "user")
        self.assertEqual(result[-1]["content"], "giới thiệu dự án")
        roles = [m["role"] for m in result]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_build_messages_appends_attachment_note(self) -> None:
        messages = [{"role": "user", "content": "xem ảnh này", "attachments": ["a.png"]}]
        result = self.builder.build_messages(messages, doc_ctx="", code_ctx="")
        self.assertIn("a.png", result[-1]["content"])
        self.assertIn("Model text-only", result[-1]["content"])

    def test_build_messages_no_context_falls_back_to_refusal(self) -> None:
        result = self.builder.build_messages(
            [{"role": "user", "content": "hi"}], doc_ctx="", code_ctx="", language="en"
        )
        self.assertIn("Respond exclusively in English", result[0]["content"])
        self.assertIn("Không tìm thấy ngữ cảnh", result[0]["content"])

    def test_build_messages_suppress_citations(self) -> None:
        result = self.builder.build_messages(
            [{"role": "user", "content": "hi"}],
            doc_ctx="ctx",
            code_ctx="",
            suppress_citations=True,
        )
        self.assertIn("CHẾ ĐỘ CÂU HỎI NHÂN VẬT/VAI TRÒ", result[0]["content"])


class PromptBuilderVisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = DefaultPromptBuilder()

    def test_vision_messages_with_pre_encoded_chunks(self) -> None:
        messages = [{"role": "user", "content": "mô tả"}]
        result = self.builder.build_vision_messages(
            messages, doc_ctx="", code_ctx="", image_chunks=["data:image/png;base64,AAAA"]
        )
        last = result[-1]["content"]
        self.assertIsInstance(last, list)
        self.assertTrue(any(part.get("type") == "image_url" for part in last))

    def test_vision_messages_image_attachment_encoded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "pic.png"
            image_path.write_bytes(b"fake-png-bytes")
            messages = [{"role": "user", "content": "xem ảnh", "attachments": [str(image_path)]}]
            with (
                mock.patch("ai_assistant.rag.image_utils.is_image_file", return_value=True),
                mock.patch("ai_assistant.rag.image_utils.image_to_data_uri", return_value="data:image/png;base64,XXXX"),
            ):
                result = self.builder.build_vision_messages(messages, doc_ctx="", code_ctx="")
        last = result[-1]["content"]
        self.assertTrue(any(p.get("type") == "image_url" for p in last))

    def test_vision_messages_non_image_attachment(self) -> None:
        messages = [{"role": "user", "content": "xem file", "attachments": ["readme.txt"]}]
        with mock.patch("ai_assistant.rag.image_utils.is_image_file", return_value=False):
            result = self.builder.build_vision_messages(messages, doc_ctx="", code_ctx="")
        last = result[-1]["content"]
        texts = "".join(p["text"] for p in last if p.get("type") == "text")
        self.assertIn("readme.txt", texts)

    def test_vision_messages_empty_content(self) -> None:
        result = self.builder.build_vision_messages([{"role": "user", "content": ""}], "", "")
        last = result[-1]["content"]
        self.assertTrue(any(p.get("type") == "text" for p in last))

    def test_vision_history_list_content_unpacked(self) -> None:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "phần 1"}, {"type": "image_url", "image_url": {}}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "tiếp"},
        ]
        result = self.builder.build_vision_messages(messages, "", "")
        # The list-part history message is flattened to text before trimming.
        user_msgs = [m["content"] for m in result if m["role"] == "user"]
        self.assertIn("phần 1", user_msgs[0] if isinstance(user_msgs[0], str) else "")


if __name__ == "__main__":
    unittest.main()
