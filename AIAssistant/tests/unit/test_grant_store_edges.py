"""Edge coverage for the single-use approval grant store.

``covers`` is deliberately single-use and ``peek`` is deliberately not, so
the two paths diverge exactly where the remaining uncovered lines sit: the
multi-use decrement arm of ``covers``, the empty/unknown/expired probes of
``peek`` and the bulk ``revoke_all`` used when a run finishes. Expiry is
driven by injecting entries directly — waiting ten minutes is not a test
strategy.
"""
from __future__ import annotations

import unittest
from time import monotonic

from ai_assistant.application.approvals import InMemoryApprovalGrantStore


class CoversTests(unittest.TestCase):
    def test_a_multi_use_grant_decrements_instead_of_popping(self):
        """The decrement arm exists for refreshed grants carrying >1 use."""
        store = InMemoryApprovalGrantStore()
        store._grants["tok"] = ("fp", monotonic() + 600.0, 2)
        self.assertTrue(store.covers("tok", "fp"))
        self.assertEqual(store._grants["tok"][2], 1)
        self.assertTrue(store.covers("tok", "fp"))
        self.assertNotIn("tok", store._grants)

    def test_a_wrong_fingerprint_does_not_spend_the_grant(self):
        store = InMemoryApprovalGrantStore()
        store.issue("tok", "fp")
        self.assertFalse(store.covers("tok", "other"))
        self.assertTrue(store.covers("tok", "fp"))

    def test_an_expired_grant_is_dropped(self):
        store = InMemoryApprovalGrantStore()
        store._grants["tok"] = ("fp", monotonic() - 1.0, 1)
        self.assertFalse(store.covers("tok", "fp"))
        self.assertNotIn("tok", store._grants)


class PeekTests(unittest.TestCase):
    def test_an_empty_token_probes_false(self):
        self.assertFalse(InMemoryApprovalGrantStore().peek("", "fp"))

    def test_an_unknown_token_probes_false(self):
        self.assertFalse(InMemoryApprovalGrantStore().peek("nope", "fp"))

    def test_an_expired_grant_probes_false_and_is_dropped(self):
        store = InMemoryApprovalGrantStore()
        store._grants["tok"] = ("fp", monotonic() - 1.0, 1)
        self.assertFalse(store.peek("tok", "fp"))
        self.assertNotIn("tok", store._grants)

    def test_a_live_grant_probes_without_spending(self):
        store = InMemoryApprovalGrantStore()
        store.issue("tok", "fp")
        self.assertTrue(store.peek("tok", "fp"))
        self.assertTrue(store.peek("tok", "fp"))
        self.assertTrue(store.covers("tok", "fp"))


class RevokeTests(unittest.TestCase):
    def test_revoke_all_drops_every_grant(self):
        store = InMemoryApprovalGrantStore()
        store.issue("a", "fp-a")
        store.issue("b", "fp-b")
        store.revoke_all()
        self.assertFalse(store.peek("a", "fp-a"))
        self.assertFalse(store.peek("b", "fp-b"))

    def test_revoke_drops_one_grant(self):
        store = InMemoryApprovalGrantStore()
        store.issue("a", "fp-a")
        store.issue("b", "fp-b")
        store.revoke("a")
        self.assertFalse(store.peek("a", "fp-a"))
        self.assertTrue(store.peek("b", "fp-b"))


if __name__ == "__main__":
    unittest.main()
