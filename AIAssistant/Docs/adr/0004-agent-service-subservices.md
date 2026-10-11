# ADR 0004 — AgentService split into injected sub-services

- Status: Accepted
- Date: 2026-10-09
- Supersedes: none

## Context

`AgentService` originally owned every cross-cutting concern of the agent
lifecycle: LLM orchestration, approval handling, desktop UI-action
acknowledgements, pending-action persistence and the legacy compatibility
surface. The class grew two independent state machines (approval gating and
UI-action acks) behind a single lock, and every new concern made the class
harder to test in isolation (`agents/service.py`).

## Decision

`AgentService` is decomposed into focused sub-services that own one lifecycle
each, and is itself assembled from them:

- `agents/approval_service.py` — the HITL approval lifecycle (preview, approve,
  reject, grant issuance, resume-after-approval).
- `agents/ui_action_service.py` — the desktop `application_action`
  acknowledgement lifecycle.
- `agents/service.py` — the composition point that owns shared task state,
  holds the sub-services and keeps the legacy public method surface.

Each sub-service is injectable and independently testable; the parent keeps a
compatibility facade for callers that predate the split (ADR 0001).

## Consequences

- A change to approval gating no longer risks UI-ack logic and vice versa.
- Sub-services can be unit-tested without constructing the full LLM stack.
- The public entry points of `AgentService` remain stable, so the Qt desktop
  transport and the legacy `chatbot_agent` shim are unaffected.