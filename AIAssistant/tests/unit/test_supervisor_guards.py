"""Guard-path coverage for the supervisor's policy helpers.

``orchestration/supervisor.py`` is the least-privilege chokepoint: it picks one
worker role, refuses tools the role may not run, verifies the result
independently and writes an audit line without ever failing a request. These
tests drive the branches the happy-path runner tests never reach — the explicit
A2A handoff targets, the ``tool is None`` authorisation, the malformed-result
verification arms, the audit-log eviction, the three audit-path fallbacks and
the OSError swallow.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_assistant.orchestration import supervisor
from ai_assistant.orchestration.supervisor import (
    Delegation,
    Specialist,
    _get_audit_path,
    audit,
    authorise,
    delegate,
    reflect_result,
    verify_result,
)
from ai_assistant.tools.builtin.transfer_tools import TRANSFER_TOOL_HANDLERS


class AuditPathResolutionTests(unittest.TestCase):
    def test_app_data_dir_wins_when_configured(self):
        with patch("ai_assistant.settings.data_dir",
                   return_value=Path("/data/3d")):
            path = _get_audit_path()
        self.assertEqual(Path(path), Path("/data/3d/AIAssistant/agent_audit.jsonl"))

    def test_the_cache_directory_is_used_when_app_data_is_unset(self):
        class _Paths:
            cache_dir = Path("/cache/3d")
        with patch("ai_assistant.settings.data_dir", return_value=None), \
                patch("ai_assistant.config.paths.get_paths", return_value=_Paths()):
            path = _get_audit_path()
        self.assertEqual(Path(path), Path("/cache/3d/agent_audit.jsonl"))

    def test_a_broken_path_resolver_falls_back_to_the_repo_cache(self):
        """Any resolver failure still yields a usable path, never an exception."""
        with patch("ai_assistant.settings.data_dir", return_value=None), \
                patch("ai_assistant.config.paths.get_paths", side_effect=RuntimeError("nope")):
            path = _get_audit_path()
        self.assertTrue(path.endswith(str(Path("Cache") / "agent_audit.jsonl")))
        self.assertTrue(Path(path).is_absolute())


class ExplicitHandoffTests(unittest.TestCase):
    def test_a_transfer_tool_routes_to_its_named_specialist(self):
        delegation = delegate("open the editor", "s1", tool="transfer_to_code_agent")
        self.assertIs(delegation.specialist, Specialist.CODE)
        self.assertEqual(delegation.reason, "explicit supervisor handoff to code")
        self.assertEqual(delegation.tool, "transfer_to_code_agent")

    def test_each_transfer_target_has_its_own_specialist(self):
        for tool, specialist in supervisor.TRANSFER_TARGETS.items():
            with self.subTest(tool=tool):
                self.assertIs(
                    delegate("hand off", "s1", tool=tool).specialist, specialist,
                )


class AuthorisationTests(unittest.TestCase):
    def test_a_direct_response_needs_no_specialist_policy(self):
        """``tool is None`` means the supervisor answers itself."""
        delegation = delegate("hello", "s1")
        self.assertIsNone(delegation.tool)
        self.assertEqual(authorise(delegation, needs_approval=False), (True, None))

    def test_a_code_tool_without_approval_is_refused(self):
        delegation = delegate("patch it", "s1", tool="write_file")
        self.assertIs(delegation.specialist, Specialist.CODE)
        allowed, reason = authorise(delegation, needs_approval=False)
        self.assertFalse(allowed)
        self.assertIn("approval workflow", reason)

    def test_a_code_tool_with_approval_is_allowed(self):
        delegation = delegate("patch it", "s1", tool="write_file")
        self.assertEqual(authorise(delegation, needs_approval=True), (True, None))

    def test_a_supervisor_role_is_refused_every_tool(self):
        delegation = Delegation(Specialist.SUPERVISOR, "some_tool", "reason", "key")
        allowed, reason = authorise(delegation, needs_approval=True)
        self.assertFalse(allowed)
        self.assertIn("No specialist policy exists", reason)

    def test_a_tool_outside_the_specialist_set_is_refused(self):
        """Least privilege is table-enforced, not an accident of classification.

        ``delegate`` never routes ``write_file`` to RESEARCH, but a hand-crafted
        delegation pairing them must be refused fail-closed (review 13).
        """
        delegation = Delegation(Specialist.RESEARCH, "write_file", "reason", "key")
        allowed, reason = authorise(delegation, needs_approval=True)
        self.assertFalse(allowed)
        self.assertIn("not permitted for specialist research", reason)


class VerificationArmTests(unittest.TestCase):
    def test_an_unstructured_result_fails_verification(self):
        outcome = verify_result(
            Delegation(Specialist.CODE, "read_file", "r", "k"), "just a string",
        )
        self.assertFalse(outcome["passed"])
        self.assertEqual(outcome["reason"], "Tool result is not an object.")

    def test_a_non_zero_exit_code_fails_verification(self):
        outcome = verify_result(
            Delegation(Specialist.CODE, "run_command", "r", "k"),
            {"return_code": 2, "output": "ok"},
        )
        self.assertFalse(outcome["passed"])
        self.assertEqual(outcome["reason"], "Command exited with code 2.")

    def test_an_explicit_success_flag_is_honoured(self):
        outcome = verify_result(
            Delegation(Specialist.CODE, "run_command", "r", "k"),
            {"success": False, "output": "nope"},
        )
        self.assertFalse(outcome["passed"])

    def test_an_error_payload_fails_verification(self):
        outcome = verify_result(
            Delegation(Specialist.CODE, "read_file", "r", "k"), {"error": "boom"},
        )
        self.assertFalse(outcome["passed"])
        self.assertEqual(outcome["reason"], "boom")

    def test_a_clean_structured_result_passes(self):
        outcome = verify_result(
            Delegation(Specialist.CODE, "read_file", "r", "k"),
            {"content": "file body"},
        )
        self.assertTrue(outcome["passed"])

    def test_a_read_tool_that_returns_no_evidence_is_refused(self):
        outcome = verify_result(
            Delegation(Specialist.CODE, "read_file", "r", "k"), {"content": ""},
        )
        self.assertFalse(outcome["passed"])
        self.assertIn("no usable evidence", outcome["reason"])

    def test_a_workflow_result_needs_the_qt_acknowledgement(self):
        pending = verify_result(
            Delegation(Specialist.WORKFLOW, "application_action", "r", "k"),
            {"pending_ui_ack": True},
        )
        self.assertTrue(pending["passed"])
        acknowledged = verify_result(
            Delegation(Specialist.WORKFLOW, "application_action", "r", "k"),
            {"success": True},
        )
        self.assertTrue(acknowledged["passed"])


class ReflectionArmTests(unittest.TestCase):
    def test_an_unstructured_result_is_revised(self):
        outcome = reflect_result(
            Delegation(Specialist.CODE, "read_file", "r", "k"), "not a dict",
            {"passed": True, "reason": "ok"},
        )
        self.assertEqual(outcome["decision"], "revise")
        self.assertEqual(outcome["reason"], "Tool returned an unstructured result.")

    def test_a_failed_result_is_revised(self):
        outcome = reflect_result(
            Delegation(Specialist.CODE, "read_file", "r", "k"), {"error": "boom"},
            {"passed": True, "reason": "ok"},
        )
        self.assertFalse(outcome["passed"])
        self.assertIn("boom", outcome["reason"])

    def test_a_failed_verification_is_revised_with_its_own_reason(self):
        outcome = reflect_result(
            Delegation(Specialist.CODE, "read_file", "r", "k"), {"content": "x"},
            {"passed": False, "reason": "no evidence"},
        )
        self.assertEqual(outcome["decision"], "revise")
        self.assertEqual(outcome["reason"], "no evidence")

    def test_a_passing_verification_continues(self):
        outcome = reflect_result(
            Delegation(Specialist.CODE, "read_file", "r", "k"), {"content": "x"},
            {"passed": True, "reason": "ok"},
        )
        self.assertTrue(outcome["passed"])
        self.assertEqual(outcome["decision"], "continue")

    def test_a_missing_verification_reason_still_names_the_failure(self):
        outcome = reflect_result(
            Delegation(Specialist.CODE, "read_file", "r", "k"), {"content": "x"},
            {"passed": False},
        )
        self.assertEqual(outcome["reason"], "Verification failed.")


class DelegationLogThrottleTests(unittest.TestCase):
    def setUp(self) -> None:
        supervisor._RECENT_DELEGATION_LOGS.clear()

    def tearDown(self) -> None:
        supervisor._RECENT_DELEGATION_LOGS.clear()

    def test_a_repeat_within_the_window_keeps_its_idempotency_key(self):
        """Same session + task → same key, so the throttle can dedupe it."""
        first = delegate("same task", "same-session", tool="read_file")
        second = delegate("same task", "same-session", tool="read_file")
        self.assertEqual(first.idempotency_key, second.idempotency_key)
        self.assertEqual(len(supervisor._RECENT_DELEGATION_LOGS), 1)

    def test_an_oversized_log_is_evicted_on_the_next_entry(self):
        """The map is bounded: exceeding 512 keys clears it rather than grow."""
        import time as _time
        now = _time.monotonic()
        supervisor._RECENT_DELEGATION_LOGS.update(
            {(f"stale-{i}", f"key-{i}"): now for i in range(513)}
        )
        # A brand-new delegation key forces an insert, which trips the cap
        # and wipes the whole map — the bound is enforced, not aspirational.
        delegate("fresh task", "brand-new-session", tool="read_file")
        self.assertEqual(len(supervisor._RECENT_DELEGATION_LOGS), 0)


class AuditWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.delegation = delegate("task", "s1", tool="read_file")

    def test_an_unwritable_audit_log_never_escapes(self):
        audit_path = Path(__file__).resolve().parent / "no-such-dir" / "agent_audit.jsonl"
        with (
            patch.object(supervisor, "_get_audit_path", return_value=str(audit_path)),
            patch.object(supervisor.os, "makedirs", side_effect=OSError("read-only")),
            self.assertLogs("ai_assistant.orchestration.supervisor", level="WARNING") as logged,
        ):
            audit("tool.completed", self.delegation, note="x")  # must not raise
        # The failure was reported as a warning, not raised …
        self.assertTrue(any("Unable to write agent audit event" in line for line in logged.output))
        # … and no partial audit file appeared.
        self.assertFalse(audit_path.exists())

    def test_the_record_carries_the_event_and_the_delegation(self):
        written: list[str] = []

        class _Handle:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def write(self, payload):
                written.append(payload)

        with patch.object(supervisor.os, "makedirs"), \
                patch.object(supervisor, "open", create=True, return_value=_Handle()):
            audit("tool.completed", self.delegation, note="x")

        self.assertEqual(len(written), 1)
        self.assertIn('"tool.completed"', written[0])
        self.assertIn('"delegation"', written[0])


class RemoteLookupRuntimeTests(unittest.TestCase):
    """The injected remote lookup must fail closed, never open."""

    def test_no_lookup_means_no_remote_endpoint(self):
        delegation = delegate("read it", "s1", tool="read_file")
        self.assertIsNone(delegation.remote_endpoint)
        self.assertNotIn("(A2A remote:", delegation.reason)

    def test_a_resolved_endpoint_is_appended_to_the_reason(self):
        delegation = delegate(
            "read it", "s1", tool="read_file",
            remote_lookup=lambda _name: "http://remote:8080/a2a",
        )
        self.assertEqual(delegation.remote_endpoint, "http://remote:8080/a2a")
        self.assertIn("(A2A remote: http://remote:8080/a2a)", delegation.reason)

    def test_a_lookup_that_raises_fails_closed(self):
        def _boom(_name):
            msg = "registry unavailable"
            raise RuntimeError(msg)

        delegation = delegate("read it", "s1", tool="read_file", remote_lookup=_boom)
        self.assertIsNone(delegation.remote_endpoint)
        self.assertNotIn("(A2A remote:", delegation.reason)

    def test_the_resolver_directly_handles_no_lookup(self):
        self.assertIsNone(supervisor._resolve_remote_endpoint(Specialist.CODE, None))


class SpecialisationBranchTests(unittest.TestCase):
    def test_prefer_code_steers_a_research_tool_away_from_research(self):
        delegation = delegate("read it", "s1", tool="read_file", prefer_code=True)
        self.assertIs(delegation.specialist, Specialist.CODE)
        self.assertEqual(delegation.reason, "coding task requires isolated repository context")

    def test_a_research_tool_defaults_to_the_research_role(self):
        delegation = delegate("read it", "s1", tool="read_file")
        self.assertIs(delegation.specialist, Specialist.RESEARCH)
        self.assertEqual(delegation.reason, "tool is read/RAG-only")

    def test_a_validation_tool_selects_the_verification_role(self):
        delegation = delegate("check it", "s1", tool="validate_file")
        self.assertIs(delegation.specialist, Specialist.VERIFICATION)
        self.assertEqual(delegation.reason, "tool validates an observable result")

    def test_a_desktop_action_selects_the_workflow_role(self):
        delegation = delegate("open browser", "s1", tool="application_action")
        self.assertIs(delegation.specialist, Specialist.WORKFLOW)
        self.assertEqual(delegation.reason, "tool dispatches a Qt desktop workflow")

    def test_an_unknown_tool_falls_back_to_the_supervisor(self):
        delegation = delegate("just answer", "s1", tool="not_a_tool")
        self.assertIs(delegation.specialist, Specialist.SUPERVISOR)
        self.assertEqual(delegation.reason, "direct response or unresolved intent")


class RoutingRegistryTests(unittest.TestCase):
    """The routing registry is data, and its invariants are enforced.

    Review finding 12: routing used to be an order-sensitive ``if``/``elif``
    chain and the handoff vocabulary lived in five parallel places. The rules
    are now a table — these tests are what keeps the table honest, so drift
    (a rule for a specialist with no policy row, a handoff target without a
    marker handler) fails the suite instead of misrouting at runtime.
    """

    def test_every_rule_names_a_specialist_that_has_a_policy_row(self):
        """A rule routing to a specialist missing from ALLOWED_TOOLS would
        fail-closed every call it produced — a wiring bug, not a policy."""
        for rule in supervisor._TOOL_ROUTE_RULES:
            with self.subTest(basis=rule.basis):
                self.assertIn(rule.specialist, supervisor.ALLOWED_TOOLS)
                self.assertTrue(rule.tools)

    def test_each_rule_reaches_its_specialist_through_delegate(self):
        """The table is what routing obeys, not the lookup path around it."""
        for rule in supervisor._TOOL_ROUTE_RULES:
            tool = sorted(rule.tools)[0]
            with self.subTest(basis=rule.basis, tool=tool):
                delegation = delegate("probe", "s1", tool=tool, prefer_code=rule.requires_prefer_code)
                self.assertIs(delegation.specialist, rule.specialist)
                self.assertEqual(delegation.reason, rule.reason)

    def test_every_transfer_specialist_may_run_its_own_marker(self):
        """A handoff whose own specialist refuses its marker would dead-end."""
        for name, specialist in supervisor.TRANSFER_TARGETS.items():
            with self.subTest(specialist=specialist.value):
                self.assertIn(name, supervisor.ALLOWED_TOOLS[specialist])

    def test_a_specialist_that_only_transfers_may_run_nothing_else(self):
        """Marker-only roles stay marker-only: fail-closed, least privilege."""
        marker_only = set(supervisor.TRANSFER_TARGETS.values()) - supervisor._EXPLICIT_TOOL_ROWS
        self.assertTrue(marker_only)
        for specialist in marker_only:
            with self.subTest(specialist=specialist.value):
                self.assertEqual(
                    supervisor.ALLOWED_TOOLS[specialist],
                    {name for name, s in supervisor.TRANSFER_TARGETS.items() if s is specialist},
                )

    def test_every_handoff_target_has_one_marker_handler(self):
        """The tools layer derives its markers from the same domain table."""
        self.assertEqual(set(TRANSFER_TOOL_HANDLERS), set(supervisor.TRANSFER_TARGETS))
        for name, specialist in supervisor.TRANSFER_TARGETS.items():
            with self.subTest(name=name):
                marker = TRANSFER_TOOL_HANDLERS[name]({"intent": "x"})
                self.assertEqual(marker["status"], f"transferred_to_{specialist.value}")


if __name__ == "__main__":
    unittest.main()
