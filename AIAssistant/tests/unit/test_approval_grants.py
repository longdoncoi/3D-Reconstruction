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
from ai_assistant.domain.approvals import (
    approval_covers_invocation,
    approval_fingerprint,
    build_approval_preview,
    derive_approval_scope,
    issue_grant_pair,
)
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


class IssueGrantPairTests(unittest.TestCase):
    """One approval mints exactly two namespaced single-use grants (review 8)."""

    def test_direct_token_is_namespaced_off_the_run_token(self) -> None:
        run_token, direct_token = issue_grant_pair("write_file", {"path": "a"}, run_id="act-1")
        self.assertEqual(run_token, "act-1")
        self.assertEqual(direct_token, "act-1#direct")

    def test_falls_back_to_a_fingerprint_anchored_run_token(self) -> None:
        run_token, direct_token = issue_grant_pair("write_file", {"path": "a"})
        fingerprint = approval_fingerprint("write_file", {"path": "a"})
        self.assertEqual(run_token, f"approval-{fingerprint[:16]}")
        self.assertEqual(direct_token, f"{run_token}#direct")

    def test_pair_is_deterministic_for_the_same_run(self) -> None:
        self.assertEqual(
            issue_grant_pair("write_file", {"path": "a"}, run_id="r"),
            issue_grant_pair("write_file", {"path": "a"}, run_id="r"),
        )

    def test_fallback_tokens_differ_per_invocation(self) -> None:
        """With no run id the token is anchored to the invocation fingerprint."""
        a = issue_grant_pair("write_file", {"path": "a"})
        b = issue_grant_pair("write_file", {"path": "b"})
        self.assertNotEqual(a[0], b[0])
        self.assertNotEqual(a[1], b[1])


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

    def test_a_granted_run_without_a_pinned_fingerprint_falls_back(self) -> None:
        """A graph built outside the runner has no pinned fingerprint.

        It falls back to the run-level grant; the gateway still refuses the
        execution without a live capability, so this is not an escalation.
        """
        state = {"approval_granted": True, "granted_fingerprint": ""}
        self.assertTrue(approval_covers_run(state, "write_file", {"path": "a.txt"}, self._ctx()))

    def test_a_wrong_pinned_fingerprint_is_never_treated_as_run_level(self) -> None:
        state = {"approval_granted": True, "granted_fingerprint": "not-the-real-fingerprint"}
        self.assertFalse(approval_covers_run(state, "write_file", {"path": "a.txt"}, self._ctx()))

    def test_live_probe_can_withdraw_coverage(self) -> None:
        state = self._state("write_file", {"path": "a.txt"})
        ctx = self._ctx(covers=lambda _tool, _params: False)
        self.assertFalse(approval_covers_run(state, "write_file", {"path": "a.txt"}, ctx))


class ApprovalCoversInvocationTests(unittest.TestCase):
    """The domain half of the grant check, independent of graph state."""

    def test_a_matching_fingerprint_covers_the_invocation(self) -> None:
        fp = approval_fingerprint("write_file", {"path": "a.txt"})
        self.assertTrue(approval_covers_invocation(fp, "write_file", {"path": "a.txt"}))

    def test_a_different_tool_is_not_covered(self) -> None:
        fp = approval_fingerprint("write_file", {"path": "a.txt"})
        self.assertFalse(approval_covers_invocation(fp, "run_command", {"path": "a.txt"}))

    def test_different_parameters_are_not_covered(self) -> None:
        fp = approval_fingerprint("write_file", {"path": "a.txt"})
        self.assertFalse(approval_covers_invocation(fp, "write_file", {"path": "secrets.txt"}))

    def test_an_empty_grant_falls_back_to_the_run_level_approval(self) -> None:
        """A graph built outside the runner has no pinned fingerprint.

        It falls back to the run-level grant, which the gateway still refuses
        without a live capability, so this is not an escalation.
        """
        for empty in ("", None):
            with self.subTest(granted=empty):
                self.assertTrue(approval_covers_invocation(empty, "write_file", {"path": "a.txt"}))

    def test_a_non_empty_but_wrong_grant_never_falls_back(self) -> None:
        """Only an *absent* fingerprint may fall back — a wrong one is a mismatch."""
        self.assertFalse(
            approval_covers_invocation("deadbeef", "write_file", {"path": "a.txt"})
        )

    def test_none_parameters_match_the_fingerprint_of_no_parameters(self) -> None:
        fp = approval_fingerprint("list_files")
        self.assertTrue(approval_covers_invocation(fp, "list_files", None))
        self.assertTrue(approval_covers_invocation(fp, "list_files", {}))


class DeriveApprovalScopeTests(unittest.TestCase):
    """An approval is bound to the plan it was granted against."""

    def test_same_plan_and_spec_yield_the_same_scope(self) -> None:
        plan, spec = ["step 1", "step 2"], {"goal": "ship"}
        self.assertEqual(derive_approval_scope(plan, spec), derive_approval_scope(plan, spec))

    def test_a_different_plan_yields_a_different_scope(self) -> None:
        base = derive_approval_scope(["step 1"], {"goal": "ship"})
        self.assertNotEqual(base, derive_approval_scope(["step 1", "step 2"], {"goal": "ship"}))

    def test_a_different_goal_yields_a_different_scope(self) -> None:
        base = derive_approval_scope(["step 1"], {"goal": "ship"})
        self.assertNotEqual(base, derive_approval_scope(["step 1"], {"goal": "delete everything"}))

    def test_spec_key_order_does_not_change_the_scope(self) -> None:
        self.assertEqual(
            derive_approval_scope(["s"], {"goal": "g", "areas": ["a"]}),
            derive_approval_scope(["s"], {"areas": ["a"], "goal": "g"}),
        )

    def test_an_existing_scope_wins(self) -> None:
        """A resumed graph keeps the id its approval was minted with."""
        self.assertEqual(derive_approval_scope(["s"], {"goal": "g"}, existing="carry-me"),
                         "carry-me")

    def test_scope_is_the_short_digest_of_the_plan(self) -> None:
        scope = derive_approval_scope(["step"], {"goal": "g"})
        self.assertEqual(len(scope), 16)
        self.assertTrue(all(c in "0123456789abcdef" for c in scope))

    def test_missing_plan_and_spec_still_yield_a_scope(self) -> None:
        self.assertEqual(derive_approval_scope(None, None),
                         derive_approval_scope([], {}))


class BuildApprovalPreviewTests(unittest.TestCase):
    """The preview is the contract of the user's consent."""

    def test_surfaces_every_field_the_decision_rests_on(self) -> None:
        preview = build_approval_preview(
            scope_id="scope-1",
            spec={"goal": "fix login", "affected_areas": ["auth"],
                  "acceptance_criteria": ["no 500"]},
            tool_name="write_file",
            parameters={"path": "auth.py"},
            remaining_steps=["run tests"],
        )
        self.assertEqual(preview, {
            "scope_id": "scope-1",
            "goal": "fix login",
            "affected_areas": ["auth"],
            "acceptance_criteria": ["no 500"],
            "tool": "write_file",
            "params": {"path": "auth.py"},
            "remaining_steps": ["run tests"],
        })

    def test_missing_spec_defaults_to_empty_fields(self) -> None:
        preview = build_approval_preview(
            scope_id="s", spec=None, tool_name="t", parameters=None,
        )
        self.assertEqual(preview["goal"], "")
        self.assertEqual(preview["affected_areas"], [])
        self.assertEqual(preview["acceptance_criteria"], [])
        self.assertEqual(preview["params"], {})
        self.assertEqual(preview["remaining_steps"], [])

    def test_preview_snapshots_the_parameters(self) -> None:
        """The consent snapshot must not track later mutation of the live params."""
        live = {"path": "a.txt"}
        preview = build_approval_preview(
            scope_id="s", spec={}, tool_name="write_file", parameters=live,
        )
        live["path"] = "secrets.txt"
        self.assertEqual(preview["params"], {"path": "a.txt"})

    def test_scope_is_carried_through(self) -> None:
        preview = build_approval_preview(
            scope_id="abc123", spec={}, tool_name="t", parameters={},
        )
        self.assertEqual(preview["scope_id"], "abc123")


if __name__ == "__main__":
    unittest.main()
