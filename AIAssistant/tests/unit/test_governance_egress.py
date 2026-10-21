"""Egress policy for payloads that leave the process (ADR 0010).

``redact_value`` masks a value so it may still be *useful* in a log or a task
event. A payload dispatched to a remote agent to be executed is not that case:
masking it hands the peer a falsified payload, sending it as-is puts the
secret on the wire. ``has_sensitive_keys`` is the predicate that lets the
dispatch refuse instead.
"""
from __future__ import annotations

import unittest
from unittest import mock

from ai_assistant.domain.governance import (
    has_sensitive_content,
    has_sensitive_keys,
    redact_value,
    scrub_credentials,
    scrub_pii,
)


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
        self.assertFalse(has_sensitive_content(payload))


class ContentShapeEgressTests(unittest.TestCase):
    """The wider egress rule: a *value* can carry a credential too.

    The key-only rule shipped a pasted secret in clear: the user's own
    prompt travels as the A2A message, and it carries no sensitive key at
    all, so the payload was classified ``internal`` and dispatched.
    """

    def test_a_credential_under_a_plain_key_is_an_egress_secret(self) -> None:
        self.assertTrue(has_sensitive_content({"note": "api_key=sk-abcdef0123456789"}))
        self.assertTrue(has_sensitive_content({"note": "password: correcthorsebattery"}))
        self.assertTrue(has_sensitive_content({"items": [{"x": "AKIAIOSFODNN7EXAMPLE"}]}))
        self.assertTrue(has_sensitive_content("token=abcdefghijklmnop"))

    def test_a_credential_shaped_token_is_detected_anywhere(self) -> None:
        self.assertTrue(has_sensitive_content("here is my key ghp_abcdefghijklmnopqrst"))
        self.assertTrue(has_sensitive_content({"a": {"b": ["ghp_abcdefghijklmnopqrst"]}}))

    def test_ordinary_prose_is_not_an_egress_secret(self) -> None:
        """A boundary that refused on prose would refuse ordinary requests."""
        for text in (
            "the password is stored in the vault",
            "fix the password validation bug",
            "build the 3D reconstruction pipeline",
            "please review the token refresh logic",
        ):
            with self.subTest(text=text):
                self.assertFalse(has_sensitive_content(text))
                self.assertFalse(has_sensitive_content({"query": text}))

    def test_the_message_is_masked_rather_than_refused(self) -> None:
        """Free text is redacted: the peer never needed the user's password."""
        scrubbed = scrub_credentials("my key is ghp_abcdefghijklmnopqrst")
        self.assertNotIn("ghp_abcdefghijklmnopqrst", scrubbed)
        self.assertIn("[REDACTED_SECRET]", scrubbed)

    def test_the_masker_leaves_a_non_string_alone(self) -> None:
        self.assertEqual(scrub_credentials(42), 42)
        self.assertIsNone(scrub_credentials(None))

    def test_the_client_refuses_a_payload_that_carries_a_credential(self) -> None:
        from ai_assistant.adapters.a2a_client import TrustedA2AClient

        client = TrustedA2AClient(frozenset({"https://peer.example"}))
        outcome = client.submit(
            "https://peer.example", "chat",
            "hello", metadata={"note": "api_key=sk-abcdef0123456789"},
        )
        self.assertEqual(outcome["routing"], "rejected_policy")

    def test_the_client_masks_a_credential_in_the_message(self) -> None:
        """The message is the user's own prompt — free text, so it is masked."""
        from ai_assistant.adapters.a2a_client import TrustedA2AClient

        captured: list[bytes] = []

        class _Response:
            def read(self, _size: int) -> bytes:
                return b'{"name":"R","skills":[{"id":"chat"}]}'

            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

        def _fake_urlopen(request: object, timeout: float = 0.0) -> _Response:
            data = getattr(request, "data", None)
            if data is not None:
                captured.append(data if isinstance(data, bytes) else bytes(data))
            return _Response()

        client = TrustedA2AClient(frozenset({"https://peer.example"}))
        with mock.patch("ai_assistant.adapters.a2a_client.urlopen", side_effect=_fake_urlopen):
            client.submit("https://peer.example", "chat", "my key is ghp_abcdefghijklmnopqrst")
        self.assertEqual(len(captured), 1)
        body = captured[0]
        self.assertNotIn(b"ghp_abcdefghijklmnopqrst", body)
        self.assertIn(b"[REDACTED_SECRET]", body)


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
