"""Egress policy for payloads that leave the process (ADR 0010).

``redact_value`` masks a value so it may still be *useful* in a log or a task
event. A payload dispatched to a remote agent to be executed is not that case:
masking it hands the peer a falsified payload, sending it as-is puts the
secret on the wire. ``has_sensitive_keys`` is the predicate that lets the
dispatch refuse instead.
"""
from __future__ import annotations

import unittest

from ai_assistant.domain.governance import has_sensitive_keys


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


if __name__ == "__main__":
    unittest.main()
