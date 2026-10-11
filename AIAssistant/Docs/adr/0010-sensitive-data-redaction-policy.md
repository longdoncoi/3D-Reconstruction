# ADR 0010 — Single-owner redaction policy for transports, events and logs

- Status: Accepted
- Date: 2026-10-10
- Supersedes: ADR 0005 (redaction paragraph)

## Context

Secrets, HITL continuation context and approval scopes were serialized into
durable task events, A2A payloads and log lines in several independently
maintained places. `continuation` in particular carries the internal execution
context (action id plus tool params); leaking it in a transport payload would
let a caller forge an approval for a pending action (Sprint 1.1 review).

## Decision

`domain/governance.py` is the single owner of the redaction policy:

- `SENSITIVE_KEYS` names every key whose value must never leave the process
  (`password`, `token`, `authorization`, `cookie`, `continuation`, `action_id`,
  `approval_scope`, ...).
- `redact_value(value, key)` recursively replaces those values with
  `[redacted]` and masks classic PII patterns (`scrub_pii`/`scrub_messages`).

A2A payload builders, durable task events (`application/tasks.py`) and log
scrubbing all delegate here, so adding or tightening a sensitive key is a
one-line change in one file.

## Consequences

- A new sensitive key is added exactly once and applies everywhere.
- Transport/log policies cannot silently diverge from the domain policy.
- Redaction is pure and unit-testable without importing any adapter.