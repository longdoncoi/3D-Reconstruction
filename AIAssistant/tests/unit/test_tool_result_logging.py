"""Unit tests for ``ai_assistant.domain.tools.describe_tool_result``.

ADR 0032: a tool result reaches the caller (and the sealed task event)
intact, but the *log* must not become a second, plaintext copy of whatever
the tool read. The log states what happened and how much of it — never the
payload.
"""
from __future__ import annotations

import unittest

from ai_assistant.domain.tools import describe_tool_result


class DescribeToolResultTests(unittest.TestCase):
    def test_a_non_mapping_result_is_described_by_its_type(self) -> None:
        self.assertEqual(describe_tool_result("plain text"), "str")
        self.assertEqual(describe_tool_result(None), "NoneType")

    def test_a_payload_key_is_summarised_by_size_not_by_content(self) -> None:
        described = describe_tool_result({"content": "the secret document body"})
        self.assertNotIn("secret document body", described)
        self.assertIn("content=<24 chars>", described)

    def test_every_payload_key_is_covered(self) -> None:
        described = describe_tool_result(
            {"results": ["a", "b"], "matches": {"x": 1}, "diff": "patch text"}
        )
        self.assertNotIn("patch text", described)
        self.assertIn("results=<2 items>", described)
        self.assertIn("matches=<1 items>", described)
        self.assertIn("diff=<10 chars>", described)

    def test_a_container_value_is_summarised_by_size(self) -> None:
        described = describe_tool_result({"entries": ["a", "b", "c"]})
        self.assertIn("entries=<3 items>", described)

    def test_a_scalar_fact_survives(self) -> None:
        described = describe_tool_result({"return_code": 0, "success": True})
        self.assertIn("return_code=0", described)
        self.assertIn("success=True", described)

    def test_a_secret_echoed_back_under_an_innocuous_key_is_masked(self) -> None:
        described = describe_tool_result({"note": "api_key=abcdefghijklmnop"})
        self.assertNotIn("abcdefghijklmnop", described)
        self.assertIn("[REDACTED_SECRET]", described)

    def test_a_non_string_payload_is_described_by_its_type(self) -> None:
        described = describe_tool_result({"data": 42})
        self.assertIn("data=<int>", described)

    def test_the_keys_are_reported_in_a_stable_order(self) -> None:
        described = describe_tool_result({"b": 1, "a": 2})
        self.assertEqual(described, "a=2 b=1")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
