# ADR 0018 — Hybrid Policy Enforcement

Status: ACCEPTED (Sprint 2)

Context: The agent claims hybrid / local-only policies but does not enforce them consistently at inference time.

Decision:
- `cloud_allowed()` is enforced in `openai_compatible_completion` (inference.py).
- `agent_hybrid_policy()` defaults to `local_only`; remote backends must have `data_classification` = `public`/`non_sensitive`.
- No claim of cloud capability should be made unless `AGENT_HYBRID_POLICY=cloud_allowed` and model is loaded locally.

Consequences:
- If policy is violated, `RuntimeError` is raised before any data leaves the host.
- References to ADR 0018 in other docs are now valid.
