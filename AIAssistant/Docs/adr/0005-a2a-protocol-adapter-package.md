# ADR 0005 — A2A protocol lives in a dedicated adapter package

- Status: Accepted
- Date: 2026-10-09
- Supersedes: none

## Context

The A2A transport (cards, tasks, events) crept into several places: models
were imported from the protocol SDK, payloads were assembled inside the router,
and the router duplicated shared FastAPI plumbing. Clean Architecture boundary
checks flagged that protocol *types* leaked into orchestration and that the
wire format had no single owner to evolve (redaction, versioning, validation).

## Decision

The A2A protocol is isolated in `ai_assistant.adapters.a2a_*`:

- `a2a_router.py` owns the HTTP surface and its dependencies (task persistence,
  the engine port, transport trust policy).
- `a2a_payloads.py` owns payload construction/parsing and delegates sensitive
  value masking to the single redaction owner (`domain.governance.redact_value`,
  ADR 0006).
- `a2a_models.py` owns the protocol request/response models.
- `a2a.py` keeps backward-compatible re-exports for legacy callers.

The application layer (`application/agent_runs.py`) exposes only the
`AgentOrchestrator` port; the adapter never reaches into the engine internals
(ADR 0003). Remote reachability is governed by the transport trust policy
(ADR 0008).

## Consequences

- One module owns the wire format; redaction and validation cannot drift.
- The router is thin and testable without a live agent engine.
- Existing imports of the adapter names remain valid through `a2a.py`.