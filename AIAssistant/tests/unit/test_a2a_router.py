"""A2ARouter delegates all remote transport to the single TrustedA2AClient.

ADR 0002/0003: there is one allowlist-enforcing A2A client. The router only
selects a specialist and normalises outcomes; it must never perform its own
HTTP call or fall back to local execution.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_assistant.adapters import a2a_protocol


class _FakeClient:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def submit(self, endpoint, capability, message, *, metadata=None, context_id=None, classification=None):
        self.calls.append((endpoint, capability, message, metadata))
        return self.outcome


def _remote_registry():
    remote = a2a_protocol.RemoteAgent(
        url="http://remote.test",
        card={"name": "remote", "skills": [{"id": "supervisor"}]},
        skills={"supervisor": {"id": "supervisor"}},
        last_discovered=0.0,
    )
    return patch.dict(a2a_protocol._remote_registry, {"http://remote.test": remote}, clear=True)


class A2ARouterTests(unittest.TestCase):
    def test_route_delegates_to_the_trusted_client(self):
        client = _FakeClient({"routing": "remote_selected", "agent_url": "http://remote.test", "answer": "ok"})
        with patch.object(a2a_protocol, "a2a_available", return_value=True), _remote_registry():
            result = a2a_protocol.A2ARouter(client=client).route("supervisor", "hi", {"k": "v"})

        self.assertEqual(result["source"], "remote")
        self.assertEqual(result["answer"], "ok")
        self.assertEqual(client.calls[0][0], "http://remote.test")
        self.assertEqual(client.calls[0][1], "supervisor")
        self.assertEqual(client.calls[0][3], {"k": "v"})

    def test_rejected_routing_is_surfaced_as_failure(self):
        client = _FakeClient({"routing": "rejected_policy", "error": "not trusted"})
        with patch.object(a2a_protocol, "a2a_available", return_value=True), _remote_registry():
            result = a2a_protocol.A2ARouter(client=client).route("supervisor", "hi")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "not trusted")

    def test_no_remote_returns_unavailable(self):
        with patch.object(a2a_protocol, "a2a_available", return_value=True), \
                patch.dict(a2a_protocol._remote_registry, {}, clear=True):
            result = a2a_protocol.A2ARouter(client=_FakeClient({})).route("supervisor", "hi")

        self.assertEqual(result["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
