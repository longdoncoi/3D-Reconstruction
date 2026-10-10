"""Contract tests for the injected tool gateway (ADR 0003).

``PlatformToolGateway`` is the only object agent code uses to execute tools. It
must inject the principal bound to the current execution context (so A2A
capability scopes are enforced) and set the approval flag only for the explicit
``execute_approved`` path. These tests use a fake service and never touch the
registry.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_assistant.bootstrap.runtime import PlatformToolGateway
from ai_assistant.domain.security import Principal, bind_principal


class _FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.grants: list[tuple] = []

    def execute(self, tool_name, parameters, principal, correlation_id=None, *, approval_token=""):
        self.calls.append((tool_name, parameters, principal, approval_token))
        return _FakeResult({"success": True, "tool": tool_name})

    def issue_approval_grant(self, tool_name, parameters, token):
        self.grants.append((tool_name, parameters, token))

    def approval_covers(self, tool_name, parameters, token):
        return token == "live"


class _FakeResult:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def to_dict(self) -> dict:
        return dict(self._payload)


class PlatformToolGatewayTests(unittest.TestCase):
    def test_unconfigured_gateway_reports_structured_failure(self):
        result = PlatformToolGateway().execute("read_file", {})
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "runtime_unconfigured")

    def test_execute_uses_the_bound_principal_without_approval(self):
        service = _FakeService()
        principal = Principal("a2a:test", frozenset({"project.write"}))
        with bind_principal(principal):
            result = PlatformToolGateway(service).execute("write_file", {"path": "a.txt"})

        self.assertTrue(result["success"])
        self.assertEqual(service.calls[0][2], principal)
        self.assertEqual(service.calls[0][3], "")

    def test_execute_approved_forwards_the_single_use_grant(self):
        service = _FakeService()
        PlatformToolGateway(service).execute_approved("write_file", {"path": "a.txt"}, "grant-1")
        self.assertEqual(service.calls[0][3], "grant-1")

    def test_grant_issue_and_probe_delegate_to_the_service(self):
        service = _FakeService()
        gateway = PlatformToolGateway(service)
        gateway.issue_approval_grant("write_file", {"path": "a.txt"}, "grant-1")
        self.assertEqual(service.grants[0], ("write_file", {"path": "a.txt"}, "grant-1"))
        self.assertTrue(gateway.approval_covers("write_file", {"path": "a.txt"}, "live"))
        self.assertFalse(gateway.approval_covers("write_file", {"path": "a.txt"}, "spent"))

    def test_unbound_call_falls_back_to_the_local_principal(self):
        service = _FakeService()
        PlatformToolGateway(service).execute("get_project_status", {})
        self.assertEqual(service.calls[0][2].subject, "legacy-local")


if __name__ == "__main__":
    unittest.main()
