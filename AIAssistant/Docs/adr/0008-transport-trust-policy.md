# ADR 0008 — Loopback, origin and token transport trust policy

- Status: Accepted
- Date: 2026-10-10
- Supersedes: none

## Context

The desktop server binds to `127.0.0.1:8080` for the Qt client, but binding is
a deployment detail, not a security boundary. The complementary controls were
inconsistent: the admin surface had bearer-token support, the agent/chat
surfaces had none, and a browser page running on the same machine could send
cross-site requests that looked identical to the desktop client (Sprint 1.4
review).

## Decision

`adapters/http/security.py` implements one `TransportPolicy`, parsed from the
environment by the composition root and injected into every HTTP/A2A surface.
Every mutating surface fails closed on three independent checks
(`_guards` in `security.py`):

1. **Peer check** — the socket peer must be loopback unless the surface is
   explicitly opted into remote access (`require_loopback`).
2. **Origin check** — a browser-sent `Origin`/`Referer` must match the
   deployment allow-list (always including the local Qt origins); non-browser
   callers are checked by the peer rule instead (`require_local_origin`).
3. **Bearer token** — optional per surface via `AI_ADMIN_TOKEN` /
   `AI_AGENT_TOKEN`; mandatory once a surface is exposed to the network.

Remote A2A is the special case: `allow_remote_a2a=true` without a configured
`AI_A2A_TOKEN` returns **503** (fail closed) — it can never be enabled without
a shared secret.

Tokens read from the environment keep the Qt desktop client (a loopback caller
that cannot manage secrets) working with no C++ change, while a deployment that
exposes the port has an on-by-default gate.

## Consequences

- CSRF from another local browser tab and cross-site requests are blocked.
- Exposing a surface to the network is impossible without an explicit token.
- Residual risk is limited to another *local* process calling `127.0.0.1`,
  which requires local code execution and is out of scope for a transport
  policy; it is mitigated by the requirement gates in `application/tools.py`.
- Tests can exercise the policy with `TestClient(..., client=("127.0.0.1", ..))`
  because loopback peer detection only accepts real loopback addresses.