"""Deterministic policy eval for supervisor routing and safety boundaries.

Test cases are derived from the canonical tool sets in ``multi_agent`` so the
eval stays in sync when tools are added or removed — no hard-coded lists.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_assistant.orchestration.supervisor import (  # noqa: E402
    _CODE_TOOLS,
    _RESEARCH_TOOLS,
    _TRANSFER_TARGETS,
    _VERIFICATION_TOOLS,
    _WORKFLOW_TOOLS,
    Specialist,
    authorise,
    delegate,
    verify_result,
)

# A2A and LangSmith imports — guarded so the eval runs even without them.
try:
    from ai_assistant.adapters.a2a_protocol import A2ARouter, AgentCard, build_agent_card
    _a2a_eval_ok = True
except ImportError:
    _a2a_eval_ok = False

try:
    from ai_assistant.observability import langsmith_available, langsmith_trace, span
    _obs_eval_ok = True
except ImportError:
    _obs_eval_ok = False

# ── Evidence factory ─────────────────────────────────────────────────────────
# ``verify_result`` checks specific fields for discovery tools; we build the
# minimal evidence payload that satisfies those checks so the eval exercises
# the *routing + authorisation + verification* pipeline end-to-end without
# depending on real tool output.

# Mirrors the ``evidence_fields`` dict inside ``verify_result``.
_EVIDENCE_FIELDS: dict[str, tuple[str, ...]] = {
    "read_file":       ("content",),
    "search_text":     ("results",),
    "find_files":      ("matches",),
    "list_directory":  ("entries",),
    "analyze_code":    ("functions", "classes", "includes"),
    "git_diff":        ("content",),
}

_PLACEHOLDER = "eval-stub"


def _build_evidence(tool: str, specialist: Specialist) -> dict[str, Any]:
    """Return a minimal successful result dict that passes ``verify_result``."""
    evidence: dict[str, Any] = {"success": True}
    if specialist is Specialist.WORKFLOW:
        # Workflow tools need Qt acknowledgement or explicit success.
        evidence["pending_ui_ack"] = True
    fields = _EVIDENCE_FIELDS.get(tool)
    if fields:
        # Populate the first required field with a non-empty sentinel.
        evidence[fields[0]] = [_PLACEHOLDER] if fields[0] != "content" else _PLACEHOLDER
    return evidence


# ── Expected routing table ───────────────────────────────────────────────────
# Built from the source-of-truth sets so any new tool automatically gets a
# test case.

_TOOL_TO_SPECIALIST: dict[str, Specialist] = {}
for _tool in _RESEARCH_TOOLS:
    _TOOL_TO_SPECIALIST[_tool] = Specialist.RESEARCH
for _tool in _VERIFICATION_TOOLS:
    _TOOL_TO_SPECIALIST[_tool] = Specialist.VERIFICATION
for _tool in _CODE_TOOLS:
    _TOOL_TO_SPECIALIST[_tool] = Specialist.CODE
for _tool in _WORKFLOW_TOOLS:
    _TOOL_TO_SPECIALIST[_tool] = Specialist.WORKFLOW
for _tool, _specialist in _TRANSFER_TARGETS.items():
    _TOOL_TO_SPECIALIST[_tool] = _specialist


@dataclass
class EvalResult:
    tool: str
    scenario: str
    passed: bool
    specialist: str
    detail: str = ""


# ── Eval scenarios ───────────────────────────────────────────────────────────

def _eval_routing() -> list[EvalResult]:
    """Every known tool routes to the correct specialist."""
    results: list[EvalResult] = []
    for tool, expected in _TOOL_TO_SPECIALIST.items():
        delegation = delegate("eval-routing", "eval", tool, {})
        ok = delegation.specialist is expected
        results.append(EvalResult(
            tool=tool, scenario="routing",
            passed=ok, specialist=str(delegation.specialist),
            detail="" if ok else f"expected {expected.value}",
        ))
    return results


def _eval_unknown_tool_fallback() -> list[EvalResult]:
    """Unknown tools fall back to SUPERVISOR and are denied."""
    delegation = delegate("eval-fallback", "eval", "__nonexistent_tool__", {})
    allowed, reason = authorise(delegation, False)
    ok = delegation.specialist is Specialist.SUPERVISOR and not allowed
    return [EvalResult(
        tool="__nonexistent_tool__", scenario="fallback",
        passed=ok, specialist=str(delegation.specialist),
        detail="" if ok else f"allowed={allowed} reason={reason}",
    )]


def _eval_authorisation() -> list[EvalResult]:
    """Code-write tools require approval; read/workflow tools are auto-allowed."""
    results: list[EvalResult] = []
    for tool, expected_specialist in _TOOL_TO_SPECIALIST.items():
        delegation = delegate("eval-auth", "eval", tool, {})
        is_code_write = tool in _CODE_TOOLS
        # Code write tools: needs_approval=False should be DENIED.
        # All other tools: auto-allowed.
        allowed, _ = authorise(delegation, needs_approval=False)
        if is_code_write:
            ok = not allowed  # must be denied without approval
            scenario = "auth-code-denied"
        else:
            ok = allowed  # should be auto-allowed
            scenario = "auth-allowed"
        results.append(EvalResult(
            tool=tool, scenario=scenario,
            passed=ok, specialist=str(delegation.specialist),
            detail="" if ok else f"allowed={allowed}",
        ))
    # Code write tools WITH approval should be allowed.
    for tool in _CODE_TOOLS:
        delegation = delegate("eval-auth-approved", "eval", tool, {})
        allowed, _ = authorise(delegation, needs_approval=True)
        ok = allowed
        results.append(EvalResult(
            tool=tool, scenario="auth-code-approved",
            passed=ok, specialist=str(delegation.specialist),
            detail="" if ok else "denied even with approval",
        ))
    return results


def _eval_verification() -> list[EvalResult]:
    """Successful results with proper evidence pass verification."""
    results: list[EvalResult] = []
    for tool, expected_specialist in _TOOL_TO_SPECIALIST.items():
        delegation = delegate("eval-verify", "eval", tool, {})
        evidence = _build_evidence(tool, expected_specialist)
        verified = verify_result(delegation, evidence)
        ok = verified["passed"]
        results.append(EvalResult(
            tool=tool, scenario="verification",
            passed=ok, specialist=str(delegation.specialist),
            detail="" if ok else verified.get("reason", ""),
        ))
    return results


def _eval_verification_failure() -> list[EvalResult]:
    """Error results and empty evidence are correctly rejected."""
    results: list[EvalResult] = []
    # Error result should fail.
    delegation = delegate("eval-verify-err", "eval", "read_file", {})
    verified = verify_result(delegation, {"error": "simulated failure"})
    ok = not verified["passed"]
    results.append(EvalResult(
        tool="read_file", scenario="verify-error",
        passed=ok, specialist=str(delegation.specialist),
        detail="" if ok else "error result was not rejected",
    ))
    # Discovery tool with empty evidence should fail.
    for tool in _EVIDENCE_FIELDS:
        if tool not in _TOOL_TO_SPECIALIST:
            continue
        delegation = delegate("eval-verify-empty", "eval", tool, {})
        verified = verify_result(delegation, {"success": True})
        ok = not verified["passed"]
        results.append(EvalResult(
            tool=tool, scenario="verify-empty-evidence",
            passed=ok, specialist=str(delegation.specialist),
            detail="" if ok else "empty evidence was not rejected",
        ))
    return results


def _eval_prefer_code_override() -> list[EvalResult]:
    """When prefer_code=True, research/verification tools route to CODE."""
    results: list[EvalResult] = []
    eligible = _RESEARCH_TOOLS | _VERIFICATION_TOOLS | _CODE_TOOLS
    for tool in eligible:
        delegation = delegate("eval-prefer-code", "eval", tool, {}, prefer_code=True)
        ok = delegation.specialist is Specialist.CODE
        results.append(EvalResult(
            tool=tool, scenario="prefer-code",
            passed=ok, specialist=str(delegation.specialist),
            detail="" if ok else f"expected code, got {delegation.specialist.value}",
        ))
    # Workflow tools should NOT be affected by prefer_code.
    for tool in _WORKFLOW_TOOLS:
        delegation = delegate("eval-prefer-code-wf", "eval", tool, {}, prefer_code=True)
        ok = delegation.specialist is Specialist.WORKFLOW
        results.append(EvalResult(
            tool=tool, scenario="prefer-code-unaffected",
            passed=ok, specialist=str(delegation.specialist),
            detail="" if ok else f"workflow tool was overridden to {delegation.specialist.value}",
        ))
    return results


def _eval_none_tool() -> list[EvalResult]:
    """No tool (None) routes to SUPERVISOR and is auto-allowed."""
    delegation = delegate("eval-none", "eval", None, {})
    allowed, _ = authorise(delegation, False)
    ok = delegation.specialist is Specialist.SUPERVISOR and allowed
    return [EvalResult(
        tool="(none)", scenario="none-tool",
        passed=ok, specialist=str(delegation.specialist),
        detail="" if ok else f"allowed={allowed}",
    )]


# ── Runner ───────────────────────────────────────────────────────────────────

def _eval_a2a_agent_cards() -> list[EvalResult]:
    """Every Specialist has a valid Agent Card skill entry."""
    results: list[EvalResult] = []
    if not _a2a_eval_ok:
        results.append(EvalResult(
            tool="(a2a)", scenario="a2a-cards-skipped",
            passed=True, specialist="n/a",
            detail="a2a_protocol not importable — test skipped (expected in minimal env)",
        ))
        return results

    card = build_agent_card()
    ok_type = isinstance(card, AgentCard)
    results.append(EvalResult(
        tool="(a2a)", scenario="a2a-card-type",
        passed=ok_type, specialist="n/a",
        detail="" if ok_type else f"expected AgentCard, got {type(card).__name__}",
    ))
    # Every specialist should appear as a skill in the card.
    skill_ids = {skill.id for skill in card.skills}
    for specialist in Specialist:
        ok = specialist.value in skill_ids
        results.append(EvalResult(
            tool=specialist.value, scenario="a2a-card-skill",
            passed=ok, specialist=specialist.value,
            detail="" if ok else "missing from Agent Card skills",
        ))
    return results


def _eval_a2a_unavailable() -> list[EvalResult]:
    """A2A router reports unavailable instead of silently executing locally."""
    results: list[EvalResult] = []
    if not _a2a_eval_ok:
        results.append(EvalResult(
            tool="(a2a)", scenario="a2a-unavailable-skipped",
            passed=True, specialist="n/a",
            detail="a2a_protocol not importable — test skipped",
        ))
        return results

    router = A2ARouter()
    # With no remote agent registered, the caller receives an explicit remote
    # outcome and decides whether a separately-authorised local execution is
    # appropriate.
    for specialist in Specialist:
        result = router.route(specialist.value, "eval-task")
        ok = result.get("source") == "remote" and result.get("status") == "unavailable"
        results.append(EvalResult(
            tool=specialist.value, scenario="a2a-unavailable",
            passed=ok, specialist=specialist.value,
            detail="" if ok else f"source={result.get('source')} status={result.get('status')}",
        ))
    return results


def _eval_langsmith_fallback() -> list[EvalResult]:
    """Observability functions work when LangSmith is not configured."""
    results: list[EvalResult] = []
    if not _obs_eval_ok:
        results.append(EvalResult(
            tool="(observability)", scenario="langsmith-fallback-skipped",
            passed=True, specialist="n/a",
            detail="observability not importable — test skipped",
        ))
        return results

    # span() should not raise even without any provider.
    try:
        with span("eval-test-span", test="true"):
            pass
        span_ok = True
    except Exception as error:  # noqa: BLE001
        span_ok = False
        results.append(EvalResult(
            tool="(observability)", scenario="langsmith-span-fallback",
            passed=False, specialist="n/a",
            detail=f"span() raised: {error}",
        ))
    if span_ok:
        results.append(EvalResult(
            tool="(observability)", scenario="langsmith-span-fallback",
            passed=True, specialist="n/a",
        ))

    # langsmith_trace() should be a no-op context when LS is disabled.
    try:
        with langsmith_trace("eval-test", run_type="chain") as ctx:
            ctx["outputs"] = {"test": True}
        trace_ok = True
    except Exception as error:  # noqa: BLE001
        trace_ok = False
        results.append(EvalResult(
            tool="(observability)", scenario="langsmith-trace-fallback",
            passed=False, specialist="n/a",
            detail=f"langsmith_trace() raised: {error}",
        ))
    if trace_ok:
        results.append(EvalResult(
            tool="(observability)", scenario="langsmith-trace-fallback",
            passed=True, specialist="n/a",
        ))

    # langsmith_available() should return False when not configured.
    ls_avail = langsmith_available()
    # We don't know if the user has LS configured, but the function must not crash.
    results.append(EvalResult(
        tool="(observability)", scenario="langsmith-available-callable",
        passed=isinstance(ls_avail, bool), specialist="n/a",
        detail="" if isinstance(ls_avail, bool) else f"returned {type(ls_avail).__name__}",
    ))

    return results


_ALL_SCENARIOS = [
    _eval_routing,
    _eval_unknown_tool_fallback,
    _eval_authorisation,
    _eval_verification,
    _eval_verification_failure,
    _eval_prefer_code_override,
    _eval_none_tool,
    _eval_a2a_agent_cards,
    _eval_a2a_unavailable,
    _eval_langsmith_fallback,
]


def main() -> int:
    all_results: list[EvalResult] = []
    for scenario_fn in _ALL_SCENARIOS:
        all_results.extend(scenario_fn())

    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    failed = [r for r in all_results if not r.passed]

    report = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": passed / total if total else 0.0,
        "scenarios": {},
    }
    # Group by scenario for readability.
    for result in all_results:
        bucket = report["scenarios"].setdefault(result.scenario, {"passed": 0, "failed": 0, "cases": []})
        if result.passed:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
        bucket["cases"].append({
            "tool": result.tool,
            "passed": result.passed,
            "specialist": result.specialist,
            **({"detail": result.detail} if result.detail else {}),
        })

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if failed:
        print(f"\n✗ {len(failed)} case(s) failed:", file=sys.stderr)
        for result in failed:
            print(f"  [{result.scenario}] {result.tool}: {result.detail}", file=sys.stderr)

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
