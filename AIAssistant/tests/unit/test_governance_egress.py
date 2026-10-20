"""Egress policy for payloads that leave the process (ADR 0010).

``redact_value`` masks a value so it may still be *useful* in a log or a task
event. A payload dispatched to a remote agent to be executed is not that case:
masking it hands the peer a falsified payload, sending it as-is puts the
secret on the wire. ``has_sensitive_keys`` is the predicate that lets the
dispatch refuse instead.
"""
from __future__ import annotations

import unittest

from ai_assistant.domain.governance import has_sensitive_keys, redact_value, scrub_pii


class OrdinaryPayloadTests(unittest.TestCase):
    def test_a_plain_parameter_set_is_allowed_on_the_wire(self) -> None:
        self.assertFalse(has_sensitive_keys({"query": "kien truc du an", "top_k": 5}))

    def test_a_bare_scalar_carries_no_keys(self) -> None:
        self.assertFalse(has_sensitive_keys("just a string"))
        self.assertFalse(has_sensitive_keys(42))
        self.assertFalse(has_sensitive_keys(None))

    def test_an_empty_container_is_allowed(self) -> None:
        self.assertFalse(has_sensitive_keys({}))
        self.assertFalse(has_sensitive_keys([]))


class SensitivePayloadTests(unittest.TestCase):
    def test_a_top_level_credential_is_detected(self) -> None:
        self.assertTrue(has_sensitive_keys({"password": "s3cret"}))

    def test_detection_folds_case_like_the_redaction_policy(self) -> None:
        self.assertTrue(has_sensitive_keys({"Password": "s3cret"}))
        self.assertTrue(has_sensitive_keys({"API_KEY": "abcdef123456"}))

    def test_credentials_nested_inside_the_payload_are_detected(self) -> None:
        payload = {"action": "admin.login", "params": {"username": "alice", "password": "s3cret"}}
        self.assertTrue(has_sensitive_keys(payload))

    def test_credentials_inside_a_list_or_tuple_are_detected(self) -> None:
        self.assertTrue(has_sensitive_keys({"items": [{"token": "abcdef123456"}]}))
        self.assertTrue(has_sensitive_keys([{"cookie": "session=1"}]))
        self.assertTrue(has_sensitive_keys(({"secret": "abcdef123456"},)))

    def test_the_hitl_continuation_counts_as_sensitive(self) -> None:
        """It carries the approval action id: leaking it would forge a grant."""
        self.assertTrue(has_sensitive_keys({"continuation": "internal-context"}))

    def test_session_correlation_ids_are_redacted_but_not_egress_secrets(self) -> None:
        """Task events must not let a log reader join sessions, but a remote
        delegation still needs the correlation ids for routing/deduplication."""
        for key in ("session_id", "request_id", "idempotency_key", "context_id"):
            with self.subTest(key=key):
                self.assertFalse(has_sensitive_keys({key: "abc"}))
                self.assertEqual(redact_value({key: "abc"})[key], "[redacted]")

    def test_the_real_remote_delegation_payload_is_allowed(self) -> None:
        """Regression for the shape ``agents/graph_wiring`` actually sends.

        The egress classifier used to reject ``session_id``/``idempotency_key``,
        so every remote A2A delegation was classified RESTRICTED and refused by
        the client before it left the process.
        """
        payload = {
            "tool": "read_file",
            "parameters": {"path": "notes.txt"},
            "session_id": "s1",
            "idempotency_key": "k1",
        }
        self.assertFalse(has_sensitive_keys(payload))


class ValueScrubTests(unittest.TestCase):
    def test_secret_pasted_under_a_plain_key_is_still_masked(self) -> None:
        redacted = redact_value({"query": "contact me at a@b.com"})
        self.assertNotIn("a@b.com", str(redacted))
        self.assertIn("[REDACTED_EMAIL]", str(redacted))

    def test_plain_text_without_pii_passes_through(self) -> None:
        self.assertEqual(redact_value({"query": "kien truc du an"}),
                         {"query": "kien truc du an"})
        self.assertEqual(redact_value("src/app.cpp"), "src/app.cpp")

    def test_scrub_pii_masks_ip_and_token_shapes(self) -> None:
        self.assertIn("[REDACTED_IP]", scrub_pii("server 10.0.0.1 down"))
        self.assertIn("[REDACTED_SECRET]", scrub_pii("api_key=abcdefghij12345"))


if __name__ == "__main__":
    unittest.main()
