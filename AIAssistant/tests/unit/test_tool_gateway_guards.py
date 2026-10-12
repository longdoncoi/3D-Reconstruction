"""Guard-path coverage for the tool execution gateway.

``ToolExecutionService`` is the only application path allowed to run a
registered tool, so every rejection it can produce is a security boundary
worth pinning: unknown tool, schema violation, missing scope, denied data
classification, missing approval grant, a crashing executor and a lying
executor (non-object payload). The single-use grant lifecycle and the
observability sink complete the contract.
"""
from __future__ import annotations

import unittest

from ai_assistant.application.tools import ToolExecutionService
from ai_assistant.domain.security import DataClassification, Principal
from ai_assistant.domain.tools import ToolSpec
from ai_assistant.plugins.registry import PluginRegistry

_SCHEMA = {
    "type": "object",
    "required": ["path"],
    "properties": {"path": {"type": "string"}},
}


def _spec(**overrides) -> ToolSpec:
    fields = {
        "name": "read_file",
        "description": "read a file",
        "input_schema": _SCHEMA,
        "timeout_seconds": 30,
    }
    fields.update(overrides)
    return ToolSpec(**fields)


def _service(executor=None, spec: ToolSpec | None = None, sink=None) -> ToolExecutionService:
    registry = PluginRegistry(allowed_plugins=frozenset({"builtin"}))
    registry.register_tool(spec or _spec(), executor or (lambda params: {"content": "x"}))
    return ToolExecutionService(registry, event_sink=sink)


def _principal(**overrides) -> Principal:
    args = {
        "subject": "tester",
        "scopes": frozenset({"read"}),
        "classification": DataClassification.INTERNAL,
    }
    args.update(overrides)
    return Principal(*args.values())


class UnknownToolTests(unittest.TestCase):
    def test_an_unregistered_tool_is_rejected_before_anything_else(self):
        result = _service().execute("no_such_tool", {}, _principal(), correlation_id="c1")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "tool_not_found")
        self.assertIn("no_such_tool", result.message)


class InputValidationTests(unittest.TestCase):
    def test_a_missing_required_parameter_is_rejected(self):
        result = _service().execute("read_file", {}, _principal(), correlation_id="c1")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "invalid_input")
        self.assertIn("path", result.message)


class ScopeEnforcementTests(unittest.TestCase):
    def test_a_principal_without_the_scope_is_forbidden(self):
        service = _service(spec=_spec(required_scope="admin:write"))
        result = service.execute("read_file", {"path": "a"}, _principal(), correlation_id="c1")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "forbidden")

    def test_a_principal_holding_the_scope_passes(self):
        service = _service(spec=_spec(required_scope="read"))
        result = service.execute("read_file", {"path": "a"}, _principal(), correlation_id="c1")
        self.assertTrue(result.success)


class DataClassificationTests(unittest.TestCase):
    def test_a_regulated_principal_cannot_use_a_restricted_tool(self):
        service = _service()  # default maximum is RESTRICTED
        principal = _principal(classification=DataClassification.REGULATED)
        result = service.execute("read_file", {"path": "a"}, principal, correlation_id="c1")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "data_policy_denied")


class ApprovalGrantTests(unittest.TestCase):
    def test_a_gated_tool_without_a_grant_asks_for_approval(self):
        service = _service(spec=_spec(requires_approval=True))
        result = service.execute("read_file", {"path": "a"}, _principal(), correlation_id="c1")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "approval_required")
        self.assertTrue(result.payload["approval_required"])
        self.assertEqual(result.payload["tool"], "read_file")

    def test_an_issued_grant_unlocks_exactly_one_invocation(self):
        service = _service(spec=_spec(requires_approval=True))
        params = {"path": "a"}
        service.issue_approval_grant("read_file", params, token="tok-1")
        first = service.execute("read_file", params, _principal(), approval_token="tok-1")
        self.assertTrue(first.success)
        # Single use: the same token is spent.
        second = service.execute("read_file", params, _principal(), approval_token="tok-1")
        self.assertEqual(second.error_code, "approval_required")

    def test_approval_covers_peeks_without_spending(self):
        """``approval_covers`` answers without consuming the grant."""
        service = _service(spec=_spec(requires_approval=True))
        service.issue_approval_grant("read_file", {"path": "a"}, token="tok-2")
        self.assertTrue(service.approval_covers("read_file", {"path": "a"}, "tok-2"))
        self.assertFalse(service.approval_covers("read_file", {"path": "b"}, "tok-2"))
        # Still unspent, so execution succeeds.
        result = service.execute("read_file", {"path": "a"}, _principal(), approval_token="tok-2")
        self.assertTrue(result.success)


class ExecutorContractTests(unittest.TestCase):
    def test_a_crashing_executor_becomes_a_structured_failure(self):
        def _boom(params):
            msg = "disk gone"
            raise RuntimeError(msg)

        result = _service(executor=_boom).execute(
            "read_file", {"path": "a"}, _principal(), correlation_id="c1",
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "tool_failure")
        self.assertIn("disk gone", result.message)

    def test_a_non_object_payload_is_rejected(self):
        result = _service(executor=lambda params: "oops").execute(
            "read_file", {"path": "a"}, _principal(), correlation_id="c1",
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "invalid_tool_result")

    def test_an_error_payload_is_a_tool_failure(self):
        result = _service(executor=lambda params: {"error": "boom"}).execute(
            "read_file", {"path": "a"}, _principal(), correlation_id="c1",
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "tool_failure")

    def test_a_clean_payload_succeeds(self):
        result = _service().execute("read_file", {"path": "a"}, _principal(), correlation_id="c1")
        self.assertTrue(result.success)
        self.assertIsNone(result.error_code)


class EventSinkTests(unittest.TestCase):
    def test_every_outcome_is_reported_to_the_sink(self):
        events: list[tuple] = []
        service = _service(sink=lambda request, result: events.append((request, result)))
        service.execute("read_file", {"path": "a"}, _principal(), correlation_id="c9")
        service.execute("no_such_tool", {}, _principal(), correlation_id="c10")
        self.assertEqual(len(events), 2)
        request, result = events[0]
        self.assertEqual(request.tool_name, "read_file")
        self.assertEqual(request.correlation_id, "c9")
        self.assertTrue(result.success)
        self.assertFalse(events[1][1].success)


if __name__ == "__main__":
    unittest.main()
