"""Transport trust policy for the HTTP, A2A and MCP surfaces (ADR 0008).

The desktop server binds to loopback, but binding is a deployment detail and
must not be the only control.  Every mutating surface therefore fails closed on
three independent checks:

1. **Peer check** — the socket peer must be loopback unless the deployment
   explicitly opts the surface into remote access.
2. **Origin check** — a browser-sent ``Origin``/``Referer`` must match the
   configured allow-list, which blocks cross-site request forgery even for
   endpoints that accept ``application/json``.
3. **Bearer token** — optional per surface, mandatory once a surface is exposed
   to the network (remote A2A can not be enabled without ``AI_A2A_TOKEN``).

Tokens are read from the environment so the Qt desktop client — which is a
loopback caller and cannot manage secrets — keeps working without a C++
change, while a deployment that exposes the port has an on-by-default gate.
Residual risk (another local process talking to ``127.0.0.1``) is documented in
``Docs/adr/0008-transport-trust-policy.md``.
"""
from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ...settings import DEFAULT_HOST, DEFAULT_PORT

_LOOPBACK_HOSTS = frozenset({"localhost", "::1", "127.0.0.1"})
# The deployment bind plus the dev-server port are implicitly allowed loopback
# origins; both derive from the canonical host/port instead of a literal list.
_ALLOWED_LOOPBACK_PORT = frozenset({DEFAULT_PORT, 8000})


@dataclass(frozen=True, slots=True)
class TransportPolicy:
    """Parsed once by the composition root and injected into the adapters."""

    allowed_origins: tuple[str, ...] = ()
    admin_token: str = ""
    agent_token: str = ""
    a2a_token: str = ""
    mcp_token: str = ""
    allow_remote_a2a: bool = False

    @classmethod
    def from_env(cls, *, allowed_origins: tuple[str, ...] = (), allow_remote_a2a: bool = False,
                 env: dict[str, str] | None = None) -> "TransportPolicy":
        values = os.environ if env is None else env
        return cls(
            allowed_origins=tuple(allowed_origins),
            admin_token=values.get("AI_ADMIN_TOKEN", "").strip(),
            agent_token=values.get("AI_AGENT_TOKEN", "").strip(),
            a2a_token=values.get("AI_A2A_TOKEN", "").strip(),
            mcp_token=values.get("AI_MCP_TOKEN", "").strip(),
            allow_remote_a2a=allow_remote_a2a,
        )


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host.casefold() in _LOOPBACK_HOSTS:
        return True
    try:
        return bool(ip_address(host).is_loopback)
    except ValueError:
        # Unknown/unparseable peer (including test client labels): fail closed.
        return host.casefold().endswith(".localhost")


def _reject(status_code: int, detail: str) -> None:
    raise HTTPException(status_code=status_code, detail=detail)


def require_loopback(request: Request, surface: str) -> None:
    """Reject any peer that is not loopback (403)."""
    peer = request.client.host if request.client else None
    if not _is_loopback(peer):
        _reject(403, f"{surface} is restricted to loopback callers")


def require_local_origin(request: Request, policy: TransportPolicy, surface: str) -> None:
    """Reject browser requests whose Origin is outside the deployment allow-list."""
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        return  # non-browser caller (Qt, curl, tests) — the peer check applies
    allowed = {item.rstrip("/") for item in policy.allowed_origins}
    allowed.update(f"http://{DEFAULT_HOST}:{port}" for port in _ALLOWED_LOOPBACK_PORT)
    allowed.update(f"http://localhost:{port}" for port in _ALLOWED_LOOPBACK_PORT)
    allowed.add(f"http://{DEFAULT_HOST}")
    allowed.add("http://localhost")
    if origin.rstrip("/") not in allowed:
        _reject(403, f"{surface} does not accept this origin")


def require_token(request: Request, token: str, surface: str) -> None:
    """Validate ``Authorization: Bearer <token>`` in constant time."""
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.casefold() != "bearer" or not hmac.compare_digest(value.strip(), token):
        _reject(401, f"Missing or invalid {surface} credentials")


def _evaluate(request: Request, *, policy: TransportPolicy, surface: str, token: str,
              allow_remote: bool) -> None:
    """One surface's trust checks, shared by the route guards and mounted apps."""
    if not allow_remote:
        require_loopback(request, surface)
    elif not token:
        # A remotely exposed surface without a shared secret stays closed.
        _reject(503, f"{surface} remote access requires a configured access token")
    if token:
        require_token(request, token, surface)
    require_local_origin(request, policy, surface)


def _guards(policy: TransportPolicy, *, surface: str, token: str, allow_remote: bool) -> list:
    def guard(request: Request) -> None:
        _evaluate(request, policy=policy, surface=surface, token=token, allow_remote=allow_remote)

    return [Depends(guard)]


def admin_guards(policy: TransportPolicy | None) -> list:
    """Destructive management surface: loopback + origin + optional bearer."""
    resolved = policy or TransportPolicy()
    return _guards(resolved, surface="admin", token=resolved.admin_token, allow_remote=False)


def agent_guards(policy: TransportPolicy | None) -> list:
    """Qt agent surface: loopback + origin + optional bearer."""
    resolved = policy or TransportPolicy()
    return _guards(resolved, surface="agent", token=resolved.agent_token, allow_remote=False)


def chat_guards(policy: TransportPolicy | None) -> list:
    """Qt completion surface: loopback + origin + optional bearer."""
    resolved = policy or TransportPolicy()
    return _guards(resolved, surface="chat", token=resolved.agent_token, allow_remote=False)


def a2a_guards(policy: TransportPolicy | None) -> list:
    """A2A surface: loopback until remote access is opted into *with* a token."""
    resolved = policy or TransportPolicy()
    return _guards(resolved, surface="a2a", token=resolved.a2a_token,
                   allow_remote=resolved.allow_remote_a2a)


def mcp_exposed_without_token(host: str | None, policy: TransportPolicy) -> bool:
    """True when the bind address reaches the network but MCP has no secret.

    Binding is a deployment detail (ADR 0008): a loopback bind needs no token,
    but publishing the port without ``AI_MCP_TOKEN`` would leave the tool
    surface reachable by every host behind the proxy. The composition root
    keeps the mount closed and says why, instead of letting callers discover
    403s.
    """
    return not _is_loopback(host) and not policy.mcp_token


class GuardedMountedApp:
    """Apply one surface's transport checks to a mounted ASGI sub-application.

    ``FastAPI.mount`` takes no route ``dependencies``, so a mounted sub-app —
    the MCP surface — would otherwise skip exactly the peer/origin/token checks
    every other surface applies. This wrapper runs the shared policy on each
    HTTP scope and answers with the same status codes; non-HTTP scopes
    (lifespan, websocket) pass through untouched.
    """

    def __init__(self, app: Any, *, policy: TransportPolicy, surface: str, token: str,
                 allow_remote: bool) -> None:
        self._app = app
        self._policy = policy
        self._surface = surface
        self._token = token
        self._allow_remote = allow_remote

    def _refusal(self, scope: Any) -> JSONResponse | None:
        try:
            _evaluate(Request(scope), policy=self._policy, surface=self._surface,
                      token=self._token, allow_remote=self._allow_remote)
        except HTTPException as error:
            return JSONResponse({"detail": error.detail}, status_code=error.status_code)
        return None

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        refusal = self._refusal(scope)
        if refusal is not None:
            await refusal(scope, receive, send)
            return
        await self._app(scope, receive, send)


def mcp_guarded_app(app: Any, policy: TransportPolicy | None) -> GuardedMountedApp:
    """Wrap the MCP sub-app so the mount enforces the MCP surface policy."""
    resolved = policy or TransportPolicy()
    return GuardedMountedApp(app, policy=resolved, surface="mcp", token=resolved.mcp_token,
                             allow_remote=bool(resolved.mcp_token))


__all__ = [
    "GuardedMountedApp",
    "TransportPolicy",
    "a2a_guards",
    "admin_guards",
    "agent_guards",
    "chat_guards",
    "mcp_exposed_without_token",
    "mcp_guarded_app",
    "require_local_origin",
    "require_loopback",
    "require_token",
]
