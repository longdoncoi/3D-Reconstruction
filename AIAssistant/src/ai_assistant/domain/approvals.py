"""HITL approval capability: a grant is bound to one exact tool invocation.

The gateway must be able to answer "was *this* tool call, with *these*
parameters, approved by a user?" without consulting free-form booleans that any
caller could set.  The answer is a capability pair:

* ``approval_fingerprint(tool, params)`` — canonical digest of the invocation
  the preview showed to the user;
* an opaque grant token issued by an approval path and validated by
  :class:`~ai_assistant.application.tools.ToolExecutionService`.

The surrounding policy also lives here rather than in the orchestration node
that consults it: whether a grant still covers an invocation
(:func:`approval_covers_invocation`), what scope an approval is minted for
(:func:`derive_approval_scope`) and exactly what the user is shown before
approving (:func:`build_approval_preview`).

Pure stdlib: no transport, no framework.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "approval_covers_invocation",
    "approval_fingerprint",
    "build_approval_preview",
    "derive_approval_scope",
    "issue_grant_pair",
]


def issue_grant_pair(
    tool_name: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    run_id: str = "",
) -> tuple[str, str]:
    """The two single-use grants one HITL approval authorises (ADR 0010).

    One user approval binds exactly two single-use grants to the same
    invocation: a *run* token for at most one re-dispatch of the identical
    call inside the resumed graph, and a *direct* token for the immediate
    execution the approving service performs. Both decision loops — the
    LangGraph resume (:class:`~ai_assistant.agents.approval_service`) and the
    durable A2A deterministic loop
    (:class:`~ai_assistant.adapters.orchestration.deterministic`) — mint them
    here so the token shape, the ``#direct`` suffix, and the approval-id
    fallback cannot drift between engines.
    """
    fingerprint = approval_fingerprint(tool_name, parameters)
    run_token = run_id or f"approval-{fingerprint[:16]}"
    return run_token, f"{run_token}#direct"


def approval_fingerprint(tool_name: str, parameters: Mapping[str, Any] | None = None) -> str:
    """Stable digest of ``(tool, params)`` for grant binding.

    Keys are sorted so an equivalent parameter dictionary produced by the
    engine always maps to the same fingerprint as the one shown in the
    approval preview.
    """
    canonical = json.dumps(
        {"tool": tool_name, "params": dict(parameters or {})},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def approval_covers_invocation(
    granted_fingerprint: str | None,
    tool_name: str,
    parameters: Mapping[str, Any] | None = None,
) -> bool:
    """True when a grant still authorises exactly this tool invocation.

    The comparison is constant-time (``hmac.compare_digest``) so a caller
    cannot recover the granted fingerprint by timing the check.

    An *empty* grant means "run-level approval with no pinned invocation" —
    such a run falls back to the run-level grant, which the gateway still
    refuses without a live capability. Only a non-empty grant must match
    bit-for-bit, so a resumed run cannot spend its approval on a different
    tool or different parameters.
    """
    if not granted_fingerprint:
        return True
    return hmac.compare_digest(granted_fingerprint, approval_fingerprint(tool_name, parameters))


def derive_approval_scope(
    plan: list[Any] | None,
    plan_spec: Mapping[str, Any] | None,
    existing: str | None = None,
) -> str:
    """Stable id for the plan/spec an approval is scoped to.

    An approval is only valid for the plan it was granted against; this digest
    lets a later step detect that the plan changed underneath a pending
    approval. A scope already carried in state wins, so a resumed graph keeps
    the id it was minted with.
    """
    if existing:
        return existing
    payload = json.dumps(
        {"plan": list(plan or []), "spec": dict(plan_spec or {})},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_approval_preview(
    *,
    scope_id: str,
    spec: Mapping[str, Any] | None,
    tool_name: str,
    parameters: Mapping[str, Any] | None,
    remaining_steps: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """What the user is shown before approving — the contract of the consent.

    Everything the decision rests on is surfaced together: the goal and
    acceptance criteria the plan promised, the concrete tool and parameters
    about to run, and what work is still outstanding. Keeping this in the
    domain means the preview cannot silently lose a field the user needs.
    """
    plan_spec = spec or {}
    return {
        "scope_id": scope_id,
        "goal": plan_spec.get("goal", ""),
        "affected_areas": plan_spec.get("affected_areas", []),
        "acceptance_criteria": plan_spec.get("acceptance_criteria", []),
        "tool": tool_name,
        "params": dict(parameters or {}),
        "remaining_steps": list(remaining_steps or []),
    }
