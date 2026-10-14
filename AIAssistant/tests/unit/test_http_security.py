"""Tests for the transport trust policy (ADR 0008).

The desktop server binds to loopback, but the surfaces must fail closed on
their own: a non-loopback peer, a foreign browser origin, or a missing bearer
token must be refused independently of how the process was launched.
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from ai_assistant.adapters.http.security import (
    GuardedMountedApp,
    TransportPolicy,
    a2a_guards,
    admin_guards,
    agent_guards,
    mcp_exposed_without_token,
    mcp_guarded_app,
)

_LOCAL = ("127.0.0.1", 50000)
_REMOTE = ("203.0.113.7", 50000)


def _client(policy: TransportPolicy, guards, peer=_LOCAL) -> TestClient:
    app = FastAPI()
    router = APIRouter(dependencies=guards(policy))
    router.add_api_route("/probe", lambda: {"ok": True}, methods=["GET", "POST"])
    app.include_router(router)
    return TestClient(app, client=peer)


class _Recorder:
    """Minimal ASGI app that records the scopes it is asked to serve."""

    def __init__(self) -> None:
        self.scopes: list[str] = []

    async def __call__(self, scope, receive, send) -> None:
        self.scopes.append(scope.get("type"))
        if scope.get("type") != "http":
            return
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"ok": true}'})


async def _receive():
    return {"type": "http.request", "body": b""}


async def _send(_message) -> None:
    return None


def _mounted(policy: TransportPolicy, peer=_LOCAL) -> TestClient:
    app = FastAPI()
    app.mount("/mcp", mcp_guarded_app(_Recorder(), policy))
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
            env={"AI_ADMIN_TOKEN": " admin ", "AI_AGENT_TOKEN": "agent",
                 "AI_A2A_TOKEN": "a2a", "AI_MCP_TOKEN": " mcp "},
        )
        self.assertEqual(policy.admin_token, "admin")
        self.assertEqual(policy.agent_token, "agent")
        self.assertEqual(policy.a2a_token, "a2a")
        self.assertEqual(policy.mcp_token, "mcp")
        self.assertTrue(policy.allow_remote_a2a)

    def test_from_env_defaults_to_no_tokens(self) -> None:
        policy = TransportPolicy.from_env(env={})
        self.assertEqual(policy.admin_token, "")
        self.assertEqual(policy.agent_token, "")
        self.assertEqual(policy.a2a_token, "")
        self.assertEqual(policy.mcp_token, "")
        self.assertFalse(policy.allow_remote_a2a)


class MountedMCPGuardTests(unittest.TestCase):
    """``app.mount`` takes no route dependencies, so the MCP sub-app guards itself."""

    def test_loopback_caller_reaches_the_mounted_app(self) -> None:
        response = _mounted(TransportPolicy()).get("/mcp/list")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})

    def test_non_loopback_caller_is_refused(self) -> None:
        response = _mounted(TransportPolicy(), peer=_REMOTE).get("/mcp/list")
        self.assertEqual(response.status_code, 403)
        self.assertIn("loopback", response.json()["detail"])

    def test_foreign_browser_origin_is_refused(self) -> None:
        response = _mounted(TransportPolicy()).get("/mcp/list", headers={"Origin": "http://evil.example"})
        self.assertEqual(response.status_code, 403)
        self.assertIn("origin", response.json()["detail"])

    def test_a_configured_mcp_token_gates_remote_callers(self) -> None:
        client = _mounted(TransportPolicy(mcp_token="mcp-secret"), peer=_REMOTE)
        self.assertEqual(client.get("/mcp/list").status_code, 401)
        self.assertEqual(
            client.get("/mcp/list", headers={"Authorization": "Bearer wrong"}).status_code, 401,
        )
        self.assertEqual(
            client.get("/mcp/list", headers={"Authorization": "Bearer mcp-secret"}).status_code,
            200,
        )

    def test_non_http_scopes_pass_through_untouched(self) -> None:
        recorder = _Recorder()
        guarded = mcp_guarded_app(recorder, TransportPolicy())
        asyncio.run(guarded({"type": "lifespan"}, _receive, _send))
        self.assertEqual(recorder.scopes, ["lifespan"])

    def test_a_policy_argument_is_optional(self) -> None:
        self.assertIsInstance(mcp_guarded_app(_Recorder(), None), GuardedMountedApp)


class MCPExposurePolicyTests(unittest.TestCase):
    """A loopback bind needs no secret; an exposed one without a token stays closed."""

    def test_loopback_bind_needs_no_token(self) -> None:
        self.assertFalse(mcp_exposed_without_token("127.0.0.1", TransportPolicy()))

    def test_exposed_bind_without_a_token_is_blocked(self) -> None:
        self.assertTrue(mcp_exposed_without_token("0.0.0.0", TransportPolicy()))

    def test_exposed_bind_with_a_token_is_allowed(self) -> None:
        self.assertFalse(mcp_exposed_without_token("0.0.0.0", TransportPolicy(mcp_token="mcp-secret")))


if __name__ == "__main__":
    unittest.main()
