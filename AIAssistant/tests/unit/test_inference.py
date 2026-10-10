"""Unit tests for inference backend selection, PII scrubbing and completion."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from ai_assistant.domain.governance import scrub_messages, scrub_pii
from ai_assistant.llm.inference import (
    backend_mode,
    cloud_allowed,
    openai_compatible_completion,
    strip_think_tags,
)


class ScrubPiiTests(unittest.TestCase):
    def test_masks_api_keys(self) -> None:
        self.assertIn("[REDACTED_SECRET]", scrub_pii("token=abcdef1234567890"))
        self.assertIn("[REDACTED_SECRET]", scrub_pii('password: "hunter2secretvalue"'))

    def test_masks_ipv4(self) -> None:
        self.assertIn("[REDACTED_IP]", scrub_pii("server at 192.168.1.10 is up"))
        self.assertNotIn("192.168.1.10", scrub_pii("server at 192.168.1.10"))

    def test_masks_emails(self) -> None:
        self.assertIn("[REDACTED_EMAIL]", scrub_pii("contact person@example.com now"))
        self.assertNotIn("person@example.com", scrub_pii("contact person@example.com now"))

    def test_non_string_returns_unchanged(self) -> None:
        self.assertEqual(scrub_pii(None), None)
        self.assertEqual(scrub_pii(12345), 12345)

    def test_scrub_messages_preserves_structure(self) -> None:
        messages = [
            {"role": "user", "content": "ip 10.0.0.5 and email a@b.co"},
            {"role": "assistant", "content": 42},
            {"role": "user"},
        ]
        result = scrub_messages(messages)
        self.assertEqual(len(result), 3)
        self.assertIn("[REDACTED_IP]", result[0]["content"])
        self.assertIn("[REDACTED_EMAIL]", result[0]["content"])
        self.assertEqual(result[1]["content"], 42)  # non-str content untouched
        self.assertNotIn("content", result[2])      # absent key not added
        self.assertEqual([m["role"] for m in result], ["user", "assistant", "user"])


class StripThinkTagsTests(unittest.TestCase):
    def test_removes_think_block(self) -> None:
        text = "Hmm <think>let me reason about this deeply</think> Final answer here."
        result = strip_think_tags(text)
        self.assertNotIn("<think>", result)
        self.assertIn("Final answer here.", result)

    def test_removes_multiline_think_block(self) -> None:
        text = "Hoặc là ta\n<think>suy nghĩ\nqua nhiều dòng</think>\nkết luận."
        result = strip_think_tags(text)
        self.assertNotIn("<think>", result)
        self.assertIn("kết luận.", result)

    def test_untouched_when_no_think_block(self) -> None:
        text = "Plain answer without reasoning tags."
        self.assertEqual(strip_think_tags(text), text)

    def test_empty(self) -> None:
        self.assertEqual(strip_think_tags(""), "")


class BackendModeTests(unittest.TestCase):
    def test_default_is_llama_cpp(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_INFERENCE_BACKEND", None)
            self.assertEqual(backend_mode(), "llama_cpp")

    def test_configured_value_lowercased(self) -> None:
        with mock.patch.dict(os.environ, {"AGENT_INFERENCE_BACKEND": "OpenAI"}):
            self.assertEqual(backend_mode(), "openai")


class CloudAllowedTests(unittest.TestCase):
    def test_local_only_default_rejects_even_public(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_HYBRID_POLICY", None)
            self.assertFalse(cloud_allowed("public"))

    def test_cloud_allowed_policy_permits_public(self) -> None:
        with mock.patch.dict(os.environ, {"AGENT_HYBRID_POLICY": "cloud_allowed"}):
            self.assertTrue(cloud_allowed("public"))
            self.assertTrue(cloud_allowed("non_sensitive"))
            self.assertFalse(cloud_allowed("confidential"))
            self.assertFalse(cloud_allowed("internal"))


class OpenAiCompatibleCompletionTests(unittest.TestCase):
    def test_missing_url_raises(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_INFERENCE_URL", None)
            with self.assertRaises(RuntimeError):
                openai_compatible_completion([{"role": "user", "content": "hi"}])

    def test_returns_parsed_payload_and_scrubs_messages(self) -> None:
        env = {
            "AGENT_INFERENCE_URL": "http://inference:8000",
            "AGENT_INFERENCE_MODEL": "qwen",
            "AGENT_INFERENCE_TIMEOUT": "5",
        }
        payload = b'{"choices": [{"message": {"role": "assistant", "content": "ok"}}]}'
        with mock.patch.dict(os.environ, env):
            with mock.patch("ai_assistant.llm.inference.urlopen") as m_urlopen:
                response = mock.MagicMock()
                response.__enter__.return_value.read.return_value = payload
                m_urlopen.return_value = response
                result = openai_compatible_completion(
                    [{"role": "user", "content": "ip 192.168.1.1"}],
                    temperature=0.1,
                )
        self.assertEqual(result["choices"][0]["message"]["content"], "ok")
        request = m_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://inference:8000/v1/chat/completions")
        body = request.data.decode("utf-8")
        self.assertIn('"model": "qwen"', body)
        self.assertIn("temperature", body)
        self.assertNotIn("192.168.1.1", body)
        self.assertIn("[REDACTED_IP]", body)

    def test_invalid_json_still_parsed_by_caller(self) -> None:
        env = {"AGENT_INFERENCE_URL": "http://inference:8000"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("ai_assistant.llm.inference.urlopen") as m_urlopen:
                response = mock.MagicMock()
                response.__enter__.return_value.read.return_value = b"not-json"
                m_urlopen.return_value = response
                with self.assertRaises(Exception):
                    openai_compatible_completion([{"role": "user", "content": "x"}])


if __name__ == "__main__":
    unittest.main()
