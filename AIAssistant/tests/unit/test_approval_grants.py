"""Regression tests for single-use, invocation-bound approval grants (ADR 0009).

These pin the fix for the HITL escalation hole: one approval used to set a
run-level boolean that unlocked *every* approval-gated tool. A grant is now
bound to the exact ``(tool, params)`` the user saw, is spent on execution and
can not be replayed.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_assistant.application.approvals import InMemoryApprovalGrantStore
from ai_assistant.domain.approvals import approval_fingerprint
from ai_assistant.orchestration.nodes.reason import ReasonContext, approval_covers_run


class ApprovalFingerprintTests(unittest.TestCase):
    def test_is_stable_and_order_independent(self) -> None:
        self.assertEqual(
            approval_fingerprint("write_file", {"a": 1, "b": 2}),
            approval_fingerprint("write_file", {"b": 2, "a": 1}),
        )

    def test_distinguishes_tool_and_params(self) -> None:
        base = approval_fingerprint("write_file", {"path": "a.txt"})
        self.assertNotEqual(base, approval_fingerprint("patch_file", {"path": "a.txt"}))
        self.assertNotEqual(base, approval_fingerprint("write_file", {"path": "b.txt"}))


class ApprovalGrantStoreTests(unittest.TestCase):
    def test_grant_is_single_use(self) -> None:
        store = InMemoryApprovalGrantStore()
        store.issue("t1", "fp-1")
        self.assertTrue(store.covers("t1", "fp-1"))
        self.assertFalse(store.covers("t1", "fp-1"), "the grant must be spent")

    def test_grant_never_covers_a_different_invocation(self) -> None:
        store = InMemoryApprovalGrantStore()
        store.issue("t1", "fp-1")
        self.assertFalse(store.covers("t1", "fp-2"))
        # a mismatched probe must not burn the grant
        self.assertTrue(store.covers("t1", "fp-1"))

    def test_peek_does_not_consume(self) -> None:
        store = InMemoryApprovalGrantStore()
        store.issue("t1", "fp-1")
        self.assertTrue(store.peek("t1", "fp-1"))
        self.assertTrue(store.peek("t1", "fp-1"))
        self.assertTrue(store.covers("t1", "fp-1"))

    def test_unknown_and_empty_tokens_are_refused(self) -> None:
        store = InMemoryApprovalGrantStore()
        self.assertFalse(store.covers("missing", "fp"))
        self.assertFalse(store.covers("", "fp"))
        store.issue("", "fp")
        self.assertFalse(store.covers("", "fp"))

    def test_grant_expires(self) -> None:
        store = InMemoryApprovalGrantStore(ttl_seconds=-1.0)
        store.issue("t1", "fp-1")
        self.assertFalse(store.covers("t1", "fp-1"))

    def test_revoke(self) -> None:
        store = InMemoryApprovalGrantStore()
        store.issue("t1", "fp-1")
        store.revoke("t1")
        self.assertFalse(store.covers("t1", "fp-1"))


class ApprovalCoversRunTests(unittest.TestCase):
    """The reason node must ask again for anything but the approved call."""

    def _ctx(self, covers=None) -> ReasonContext:
        return ReasonContext(
            complete=lambda *_: "{}",
            parse=lambda _text: (None, None),
            needs_approval=lambda _name: True,
            max_iterations=5,
            cancel_checker=lambda: False,
            select_specialist=None,
            approval_covers=covers,
        )

    def _state(self, tool: str, params: dict) -> dict:
        return {
            "approval_granted": True,
            "granted_fingerprint": approval_fingerprint(tool, params),
        }

    def test_exact_invocation_is_covered(self) -> None:
        state = self._state("write_file", {"path": "a.txt"})
        self.assertTrue(approval_covers_run(state, "write_file", {"path": "a.txt"}, self._ctx()))

    def test_other_tool_is_not_covered(self) -> None:
        state = self._state("write_file", {"path": "a.txt"})
        self.assertFalse(approval_covers_run(state, "run_command", {"command": "rm -rf /"}, self._ctx()))

    def test_other_params_are_not_covered(self) -> None:
        state = self._state("write_file", {"path": "a.txt"})
        self.assertFalse(approval_covers_run(state, "write_file", {"path": "secrets.txt"}, self._ctx()))

    def test_without_a_grant_nothing_is_covered(self) -> None:
        self.assertFalse(approval_covers_run({"approval_granted": False}, "write_file", {}, self._ctx()))

    def test_live_probe_can_withdraw_coverage(self) -> None:
        state = self._state("write_file", {"path": "a.txt"})
        ctx = self._ctx(covers=lambda _tool, _params: False)
        self.assertFalse(approval_covers_run(state, "write_file", {"path": "a.txt"}, ctx))


if __name__ == "__main__":
    unittest.main()
