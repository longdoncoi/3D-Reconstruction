# ADR 0009 — Single-use, invocation-bound approval grants

- Status: Accepted
- Date: 2026-10-10
- Supersedes: none

## Context

The approval gate used a boolean `approval_granted` flag on the run state.
Reviewers found two holes:

1. Once approved, a resumed run could call a *different* tool or different
   parameters with the same flag.
2. There was no replay protection: the same state could re-enter the reason
   node and re-dispatch the approved tool again without a new user decision.

Also, `approval_scope` is a 16-hex display scope that can collide between runs,
so it cannot be used as an authorization capability (Sprint 1.5 review).

## Decision

Approval is a **capability**, not a flag:

- `domain/approvals.py` computes `approval_fingerprint(tool, params)` — a
  SHA-256 over the exact invocation the user previewed (JSON canonical).
- `application/approvals.py` holds `InMemoryApprovalGrantStore`: grants are
  single-use (`covers()` consumes on first match), bound to one fingerprint,
  expire after a TTL (600 s, fail closed) and can be revoked.
- One approval issues **two** grants: `direct_token` for the immediate
  execution by `ApprovalService` and `run_token` for at most one re-dispatch
  when the graph resumes.
- The reason-node gate (`orchestration/nodes/reason.py::approval_covers_run`)
  now requires `approval_granted` **and** a matching `granted_fingerprint`
  **and** a non-consuming liveness probe into the grant store (`peek`). An
  exhausted or mismatched grant sends the node back to the user instead of
  retrying the refused tool.

Empty fingerprints (graphs constructed outside the runner, e.g. contract tests)
fall back to the run-level boolean; production paths always set
`granted_fingerprint`.

## Consequences

- One approval cannot be spent on a different tool, different parameters, or
  more than twice across direct execution and graph resume.
- Failing closed by expiry/revocation means stale approvals re-request the user
  rather than silently passing.
- The gateway protocol (`ports.ToolGateway`) carries `execute_approved(token)`,
  `issue_approval_grant` and `approval_covers` instead of an `approval_granted`
  boolean.