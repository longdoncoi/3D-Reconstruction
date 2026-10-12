"""Guard-path coverage for the chat prompt builders.

The uncovered arms all sit where model input meets user files: history
compaction that must skip non-string content, the code-context slot of the
system prompt, attachment annotations on history turns, and the vision path
where an attached image exists on disk but cannot be encoded. Each is a
distinct user-visible behaviour, not an implementation detail.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_assistant.llm.prompts import DefaultPromptBuilder, trim_history


class TrimHistoryTests(unittest.TestCase):
    def test_non_string_content_is_skipped_not_measured(self):
        """A ``None`` content pops without touching the token budget."""
        messages = [
            {"role": "user", "content": None},
            {"role": "user", "content": "y" * 100},
        ]
        trimmed = trim_history(messages, max_tokens=10)
        self.assertEqual(len(trimmed), 1)
        self.assertEqual(trimmed[0]["content"], "y" * 100)

    def test_a_short_history_passes_through(self):
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(trim_history(messages), messages)


class SystemPromptTests(unittest.TestCase):
    def test_code_context_is_injected_into_the_system_prompt(self):
        builder = DefaultPromptBuilder()
        result = builder.build_messages(
            [{"role": "user", "content": "hi"}], doc_ctx="", code_ctx="def f(): pass",
        )
        self.assertIn("def f(): pass", result[0]["content"])

    def test_an_empty_context_refuses_by_principle(self):
        builder = DefaultPromptBuilder()
        result = builder.build_messages([{"role": "user", "content": "hi"}], doc_ctx="", code_ctx="")
        self.assertIn("Từ chối", result[0]["content"])


class AttachmentAnnotationTests(unittest.TestCase):
    def test_history_turns_carry_their_upload_note(self):
        builder = DefaultPromptBuilder()
        result = builder.build_messages(
            [
                {"role": "user", "content": "see this", "attachments": ["plan.png"]},
                {"role": "user", "content": "and this"},
            ],
            doc_ctx="docs",
            code_ctx="",
        )
        self.assertIn("plan.png", result[1]["content"])
        self.assertIn("text-only", result[1]["content"])


class VisionMessageTests(unittest.TestCase):
    def test_an_unreadable_image_becomes_an_error_note_not_a_crash(self):
        """The file exists and has an image extension, but no decoder reads it."""
        builder = DefaultPromptBuilder()
        with tempfile.TemporaryDirectory() as directory:
            bad_image = Path(directory) / "broken.png"
            bad_image.write_bytes(b"this is not a png file at all" * 10)
            with patch(
                "ai_assistant.rag.image_utils.image_to_data_uri",
                side_effect=RuntimeError("decoder gone"),
            ):
                result = builder.build_vision_messages(
                    [{"role": "user", "content": "look", "attachments": [str(bad_image)]}],
                    doc_ctx="",
                    code_ctx="",
                )
            texts = [
                part.get("text", "")
                for part in result[-1]["content"]
                if part.get("type") == "text"
            ]
            self.assertTrue(
                any("đọc ảnh" in text for text in texts),
                texts,
            )


if __name__ == "__main__":
    unittest.main()
