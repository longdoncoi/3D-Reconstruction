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

### Where the policy lives (2026-10-11)

The policy above originally sat inline in the reason node, which is the one
place that consults it. That tied three pure rules to a 480-line orchestration
decision tree, so they were moved to `domain/approvals.py` alongside
`approval_fingerprint`:

- `approval_covers_invocation(granted_fingerprint, tool, params)` — the
  constant-time comparison, including the empty-fingerprint fallback. The
  node's `approval_covers_run` now only reads run state and consults the
  optional liveness probe, delegating the rule itself to the domain.
- `derive_approval_scope(plan, plan_spec, existing)` — the 16-hex scope
  digest, with an existing state scope winning so a resumed graph keeps the id
  its approval was minted with.
- `build_approval_preview(...)` — the consent contract: goal, affected areas,
  acceptance criteria, the concrete tool and parameters, and remaining work,
  surfaced together so the preview cannot silently drop a field the user
  needs. It snapshots the parameters rather than aliasing them, so later
  mutation of the live dict cannot rewrite what was approved.

`domain` stays pure stdlib (enforced by `test_domain_is_pure`), and the reason
node still depends inward. The gate's *behaviour* is unchanged — the existing
`ApprovalCoversRunTests` pass unmodified.

## Consequences

- One approval cannot be spent on a different tool, different parameters, or
  more than twice across direct execution and graph resume.
- Failing closed by expiry/revocation means stale approvals re-request the user
  rather than silently passing.
- The gateway protocol (`ports.ToolGateway`) carries `execute_approved(token)`,
  `issue_approval_grant` and `approval_covers` instead of an `approval_granted`
  boolean.