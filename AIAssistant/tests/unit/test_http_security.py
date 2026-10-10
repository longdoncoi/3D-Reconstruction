"""Tests for the transport trust policy (ADR 0008).

The desktop server binds to loopback, but the surfaces must fail closed on
their own: a non-loopback peer, a foreign browser origin, or a missing bearer
token must be refused independently of how the process was launched.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from ai_assistant.adapters.http.security import (
    TransportPolicy,
    a2a_guards,
    admin_guards,
    agent_guards,
)

_LOCAL = ("127.0.0.1", 50000)
_REMOTE = ("203.0.113.7", 50000)


def _client(policy: TransportPolicy, guards, peer=_LOCAL) -> TestClient:
    app = FastAPI()
    router = APIRouter(dependencies=guards(policy))
    router.add_api_route("/probe", lambda: {"ok": True}, methods=["GET", "POST"])
    app.include_router(router)
    return TestClient(app, client=peer)


class LocalSurfaceGuardTests(unittest.TestCase):
    def test_loopback_get_is_allowed(self) -> None:
        response = _client(TransportPolicy(), admin_guards).get("/probe")
        self.assertEqual(response.status_code, 200)

    def test_non_loopback_peer_is_refused(self) -> None:
        response = _client(TransportPolicy(), admin_guards, peer=_REMOTE).get("/probe")
        self.assertEqual(response.status_code, 403)

    def test_foreign_browser_origin_is_refused(self) -> None:
        client = _client(TransportPolicy(allowed_origins=("http://127.0.0.1",)), agent_guards)
        response = client.get("/probe", headers={"Origin": "http://evil.example"})
        self.assertEqual(response.status_code, 403)

    def test_allow_listed_origin_is_accepted(self) -> None:
        client = _client(TransportPolicy(allowed_origins=("http://127.0.0.1",)), agent_guards)
        response = client.get("/probe", headers={"Origin": "http://127.0.0.1"})
        self.assertEqual(response.status_code, 200)

    def test_missing_bearer_is_refused_when_a_token_is_configured(self) -> None:
        client = _client(TransportPolicy(admin_token="s3cret"), admin_guards)
        self.assertEqual(client.get("/probe").status_code, 401)
        self.assertEqual(client.get("/probe", headers={"Authorization": "Bearer wrong"}).status_code, 401)
        self.assertEqual(
            client.get("/probe", headers={"Authorization": "Bearer s3cret"}).status_code,
            200,
        )

    def test_token_is_optional_for_the_desktop_surface(self) -> None:
        self.assertEqual(_client(TransportPolicy(), agent_guards).get("/probe").status_code, 200)


class A2ASurfaceGuardTests(unittest.TestCase):
    def test_remote_a2a_is_refused_by_default(self) -> None:
        response = _client(TransportPolicy(), a2a_guards, peer=_REMOTE).get("/probe")
        self.assertEqual(response.status_code, 403)

    def test_remote_a2a_without_a_token_stays_closed(self) -> None:
        policy = TransportPolicy(allow_remote_a2a=True)
        response = _client(policy, a2a_guards, peer=_REMOTE).get("/probe")
        self.assertEqual(response.status_code, 503)

    def test_remote_a2a_requires_the_configured_token(self) -> None:
        policy = TransportPolicy(allow_remote_a2a=True, a2a_token="remote-secret")
        client = _client(policy, a2a_guards, peer=_REMOTE)
        self.assertEqual(client.get("/probe").status_code, 401)
        self.assertEqual(
            client.get("/probe", headers={"Authorization": "Bearer remote-secret"}).status_code,
            200,
        )


class PolicyLoadingTests(unittest.TestCase):
    def test_from_env_reads_tokens_and_settings(self) -> None:
        policy = TransportPolicy.from_env(
            allowed_origins=("http://127.0.0.1",),
            allow_remote_a2a=True,
            env={"AI_ADMIN_TOKEN": " admin ", "AI_AGENT_TOKEN": "agent", "AI_A2A_TOKEN": "a2a"},
        )
        self.assertEqual(policy.admin_token, "admin")
        self.assertEqual(policy.agent_token, "agent")
        self.assertEqual(policy.a2a_token, "a2a")
        self.assertTrue(policy.allow_remote_a2a)

    def test_from_env_defaults_to_no_tokens(self) -> None:
        policy = TransportPolicy.from_env(env={})
        self.assertEqual(policy.admin_token, "")
        self.assertEqual(policy.agent_token, "")
        self.assertEqual(policy.a2a_token, "")
        self.assertFalse(policy.allow_remote_a2a)


if __name__ == "__main__":
    unittest.main()
