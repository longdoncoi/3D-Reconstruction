"""Deterministic policy eval for supervisor routing and safety boundaries.

Test cases are derived from the canonical tool sets in ``multi_agent`` so the
eval stays in sync when tools are added or removed — no hard-coded lists.

The ``gateway-*`` scenarios drive the real ``ToolExecutionService`` instead of
the supervisor: scope, classification, single-use approvals and cancellation
revocation are checked against the code that enforces them.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
# The package lives under ``src/`` (ADR 0010 layout); pointing at the project
# root alone leaves ``ai_assistant`` unimportable.
sys.path.insert(0, str(ROOT / "src"))

from ai_assistant.application.approvals import InMemoryApprovalGrantStore
from ai_assistant.application.tools import ToolExecutionService
from ai_assistant.domain.memory import format_memory_block, recall_rank
from ai_assistant.domain.parallel import clamp_parallel_max
from ai_assistant.domain.security import DataClassification, Principal, principal_for
from ai_assistant.domain.tools import EVIDENCE_FIELDS, SideEffect, ToolSpec
from ai_assistant.orchestration.helpers import normalise_plan_payload, valid_parallel_group
from ai_assistant.orchestration.supervisor import (
    CODE_TOOLS,
    RESEARCH_TOOLS,
    TRANSFER_TARGETS,
    VERIFICATION_TOOLS,
    WORKFLOW_TOOLS,
    Specialist,
    authorise,
    delegate,
    verify_result,
)

# A2A and LangSmith imports — guarded so the eval runs even without them.
# A guard must not mean *silence*: the three cases below record why they could
# not run, and ``main`` exits non-zero when any of them is skipped, so a green
# run can never come from an unchecked case (the failure mode this repo keeps
# deleting under ADR 0015).
try:
    from ai_assistant.adapters.a2a_protocol import A2ARouter, AgentCard, build_agent_card
    _a2a_eval_ok = True
    _a2a_import_error = ""
except ImportError as exc:
    _a2a_eval_ok = False
    _a2a_import_error = str(exc)

try:
    from ai_assistant.observability import langsmith_available, langsmith_trace, span
    _obs_eval_ok = True
    _obs_import_error = ""
except ImportError as exc:
    _obs_eval_ok = False
    _obs_import_error = str(exc)

# ── Evidence factory ─────────────────────────────────────────────────────────
# ``verify_result`` checks specific fields for discovery tools; we build the
# minimal evidence payload that satisfies those checks so the eval exercises
# the *routing + authorisation + verification* pipeline end-to-end without
# depending on real tool output.

# Source of truth lives in ``domain.tools``; imported (not copied) so the eval
# can never drift from the table ``verify_result`` actually uses.
_EVIDENCE_FIELDS = EVIDENCE_FIELDS

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
for _tool in RESEARCH_TOOLS:
    _TOOL_TO_SPECIALIST[_tool] = Specialist.RESEARCH
for _tool in VERIFICATION_TOOLS:
    _TOOL_TO_SPECIALIST[_tool] = Specialist.VERIFICATION
for _tool in CODE_TOOLS:
    _TOOL_TO_SPECIALIST[_tool] = Specialist.CODE
for _tool in WORKFLOW_TOOLS:
    _TOOL_TO_SPECIALIST[_tool] = Specialist.WORKFLOW
for _tool, _specialist in TRANSFER_TARGETS.items():
    _TOOL_TO_SPECIALIST[_tool] = _specialist


@dataclass
class EvalResult:
    tool: str
    scenario: str
    passed: bool
    specialist: str
    detail: str = ""
    #: True when the case never ran because a dependency was missing. It is
    #: still reported as not-passed: ``main`` exits non-zero on it.
    skipped: bool = False


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
        is_code_write = tool in CODE_TOOLS
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
    for tool in CODE_TOOLS:
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
    eligible = RESEARCH_TOOLS | VERIFICATION_TOOLS | CODE_TOOLS
    for tool in eligible:
        delegation = delegate("eval-prefer-code", "eval", tool, {}, prefer_code=True)
        ok = delegation.specialist is Specialist.CODE
        results.append(EvalResult(
            tool=tool, scenario="prefer-code",
            passed=ok, specialist=str(delegation.specialist),
            detail="" if ok else f"expected code, got {delegation.specialist.value}",
        ))
    # Workflow tools should NOT be affected by prefer_code.
    for tool in WORKFLOW_TOOLS:
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


# ── Tool gateway: the injection-proof execution boundary (ADR 0009) ─────────


def _gateway_registry(executed: list[dict[str, Any]]) -> Any:
    """A scoped read tool plus an approval-gated write tool, via the registry."""
    from ai_assistant.plugins.registry import PluginRegistry

    def read_executor(params: dict[str, Any]) -> dict[str, Any]:
        executed.append({"tool": "eval_read", **params})
        return {"success": True, "content": "evidence"}

    def write_executor(params: dict[str, Any]) -> dict[str, Any]:
        executed.append({"tool": "eval_write", **params})
        return {"success": True, "content": "written"}

    registry = PluginRegistry(allowed_plugins=frozenset({"eval"}))
    registry.register_tool(
        ToolSpec(
            name="eval_read", description="eval read fixture", plugin_id="eval",
            input_schema={
                "type": "object", "additionalProperties": False,
                "properties": {"path": {"type": "string"}}, "required": ["path"],
            },
            timeout_seconds=5, side_effect=SideEffect.READ, required_scope="project.read",
            maximum_classification=DataClassification.INTERNAL,
        ),
        read_executor,
    )
    registry.register_tool(
        ToolSpec(
            name="eval_write", description="eval write fixture", plugin_id="eval",
            input_schema={
                "type": "object", "additionalProperties": False,
                "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
                "required": ["path", "text"],
            },
            timeout_seconds=5, side_effect=SideEffect.WRITE,
            requires_approval=True, required_scope="project.write",
        ),
        write_executor,
    )
    return registry


def _eval_tool_gateway_policy() -> list[EvalResult]:
    """The real gateway refuses every unsanctioned call over the dashboard.

    An LLM — or text injected into its context — can name any tool and any
    parameters. These cases drive the *actual* ``ToolExecutionService`` and
    assert the policy outcomes a reviewer can check in the report:

    * an unknown (injected) tool name never reaches an executor;
    * a principal without the scope, or whose clearance does not cover the
      tool's data ceiling, is refused before execution;
    * an approval-gated write without a grant is refused;
    * a grant unlocks *exactly* the approved invocation, once — replayed or
      re-pointed at different parameters it is refused;
    * ``revoke_all`` (what a cancellation does) invalidates outstanding grants.
    """
    results: list[EvalResult] = []
    executed: list[dict[str, Any]] = []
    grants = InMemoryApprovalGrantStore()
    service = ToolExecutionService(_gateway_registry(executed), grants=grants)
    # The clearance follows the granted scopes, exactly as the platform
    # derives it: the write scope is an elevated one.
    scoped = principal_for("eval-user", {"project.read", "project.write"})

    def case(tool: str, scenario: str, passed: bool, detail: str = "") -> None:
        results.append(EvalResult(
            tool=tool, scenario=scenario, passed=passed, specialist="gateway",
            detail=detail,
        ))

    # Scope: no scopes → refused; the required scope → executes.
    denied = service.execute("eval_read", {"path": "a.md"}, Principal(subject="anon", scopes=frozenset()))
    ok = not denied.success and denied.error_code == "forbidden"
    case("eval_read", "gateway-scope-denied", ok,
         "" if ok else f"error_code={denied.error_code}")

    allowed = service.execute("eval_read", {"path": "a.md"}, scoped)
    ok = allowed.success and allowed.payload.get("content") == "evidence"
    case("eval_read", "gateway-scope-allowed", ok,
         "" if ok else f"payload={allowed.payload}")

    # Clearance: a principal below the tool's data ceiling is refused. The
    # write tool touches restricted data, so a public-clearance caller is
    # refused even though it holds the write scope.
    public = Principal(
        subject="eval-user", scopes=frozenset({"project.read", "project.write"}),
        classification=DataClassification.PUBLIC,
    )
    classified = service.execute("eval_write", {"path": "out.txt", "text": "x"}, public)
    ok = not classified.success and classified.error_code == "data_policy_denied"
    case("eval_write", "gateway-classification", ok,
         "" if ok else f"error_code={classified.error_code}")

    # An injected tool name is not in the registry: nothing to execute.
    executed_before = len(executed)
    unknown = service.execute("eval_shell", {"cmd": "rm -rf /"}, scoped)
    ok = unknown.error_code == "tool_not_found" and len(executed) == executed_before
    case("eval_shell", "gateway-unknown-tool", ok,
         "" if ok else f"error_code={unknown.error_code}")

    # Approval gate: no grant → refused, executor untouched.
    approved = {"path": "out.txt", "text": "hello"}
    different = {"path": "out.txt", "text": "injected"}
    executed_before = len(executed)
    gated = service.execute("eval_write", approved, scoped)
    ok = (not gated.success and gated.error_code == "approval_required"
          and gated.payload.get("approval_required") is True
          and len(executed) == executed_before)
    case("eval_write", "gateway-approval-required", ok,
         "" if ok else f"error_code={gated.error_code}")

    # A grant is bound to the approved invocation: other parameters stay closed.
    token_one = uuid4().hex
    service.issue_approval_grant("eval_write", approved, token_one)
    executed_before = len(executed)
    mismatched = service.execute("eval_write", different, scoped, approval_token=token_one)
    ok = (not mismatched.success and mismatched.error_code == "approval_required"
          and len(executed) == executed_before)
    case("eval_write", "gateway-approval-binding", ok,
         "" if ok else f"error_code={mismatched.error_code}")

    # …and it unlocks exactly one execution: the replay is refused.
    unlocked = service.execute("eval_write", approved, scoped, approval_token=token_one)
    replayed = service.execute("eval_write", approved, scoped, approval_token=token_one)
    ok = (unlocked.success and not replayed.success
          and replayed.error_code == "approval_required")
    case("eval_write", "gateway-approval-single-use", ok,
         "" if ok else f"unlocked={unlocked.success} replay={replayed.error_code}")

    # Cancellation revokes outstanding grants (``revoke_all`` on the store).
    token_two = uuid4().hex
    service.issue_approval_grant("eval_write", approved, token_two)
    grants.revoke_all()
    cancelled = service.execute("eval_write", approved, scoped, approval_token=token_two)
    ok = not cancelled.success and cancelled.error_code == "approval_required"
    case("eval_write", "gateway-cancel-revokes", ok,
         "" if ok else f"error_code={cancelled.error_code}")

    return results


# ── Runner ───────────────────────────────────────────────────────────────────

def _eval_a2a_agent_cards() -> list[EvalResult]:
    """Every remote Specialist has an Agent Card skill entry — and only those."""
    results: list[EvalResult] = []
    if not _a2a_eval_ok:
        results.append(EvalResult(
            tool="(a2a)", scenario="a2a-cards-skipped",
            passed=False, specialist="n/a", skipped=True,
            detail=(
                "a2a_protocol not importable — case NOT RUN "
                f"({_a2a_import_error}). Install the missing dependency; "
                "an unchecked case may not report pass."
            ),
        ))
        return results

    card = build_agent_card()
    ok_type = isinstance(card, AgentCard)
    results.append(EvalResult(
        tool="(a2a)", scenario="a2a-card-type",
        passed=ok_type, specialist="n/a",
        detail="" if ok_type else f"expected AgentCard, got {type(card).__name__}",
    ))
    # Every remote specialist should appear as a skill in the card. PLUGIN is
    # a local composition role: no remote peer can implement tools this
    # process loaded, so advertising it would be a dead skill (ADR 0039).
    skill_ids = {skill.id for skill in card.skills}
    for specialist in Specialist:
        expected = specialist is not Specialist.PLUGIN
        ok = (specialist.value in skill_ids) is expected
        if expected:
            detail = "" if ok else "missing from Agent Card skills"
        else:
            detail = "" if ok else "the local plugin role must not be advertised over A2A"
        results.append(EvalResult(
            tool=specialist.value, scenario="a2a-card-skill",
            passed=ok, specialist=specialist.value, detail=detail,
        ))
    return results


def _eval_a2a_unavailable() -> list[EvalResult]:
    """A2A router reports unavailable instead of silently executing locally."""
    results: list[EvalResult] = []
    if not _a2a_eval_ok:
        results.append(EvalResult(
            tool="(a2a)", scenario="a2a-unavailable-skipped",
            passed=False, specialist="n/a", skipped=True,
            detail=(
                f"a2a_protocol not importable — case NOT RUN ({_a2a_import_error}). "
                "Install the missing dependency; an unchecked case may not report pass."
            ),
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
            passed=False, specialist="n/a", skipped=True,
            detail=(
                f"observability not importable — case NOT RUN ({_obs_import_error}). "
                "Install the missing dependency; an unchecked case may not report pass."
            ),
        ))
        return results

    # span() should not raise even without any provider.
    try:
        with span("eval-test-span", test="true"):
            pass
        span_ok = True
    except Exception as error:
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
    except Exception as error:
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


def _eval_memory_and_parallel_policy() -> list[EvalResult]:
    """ADR 0043/0044: the memory block and the parallel-group contract."""
    results: list[EvalResult] = []
    now = 1_800_000_000.0
    records = [
        {"id": "old", "kind": "semantic", "text": "vai tro project manager",
         "created_at": now - 90 * 86400, "salience": 0.1},
        {"id": "new", "kind": "episodic", "text": "vai tro cua ky su A",
         "created_at": now - 1 * 86400, "salience": 0.1},
        {"id": "noise", "kind": "semantic", "text": "mau sac giao dien",
         "created_at": now, "salience": 5.0},
    ]
    ranked = [record["id"] for record in recall_rank(records, "vai tro", 5, now=now)]
    results.append(EvalResult(
        tool="(memory)", scenario="memory-policy",
        passed=ranked == ["new", "old"], specialist="n/a",
        detail="" if ranked == ["new", "old"]
        else f"recall ranked {ranked} (zero-overlap noise must be dropped, newer wins)",
    ))
    results.append(EvalResult(
        tool="(memory)", scenario="memory-policy",
        passed=format_memory_block([]) == "", specialist="n/a",
        detail="an empty recall must not emit a [Long-term memory] header",
    ))
    rendered = format_memory_block([{"kind": "semantic", "text": "du an X"}])
    results.append(EvalResult(
        tool="(memory)", scenario="memory-policy",
        passed="[Long-term memory]" in rendered and "- [semantic] du an X" in rendered,
        specialist="n/a",
        detail="" if rendered else "a non-empty recall rendered nothing",
    ))
    results.append(EvalResult(
        tool="(memory)", scenario="memory-policy",
        passed=recall_rank(records, "vai tro", 0, now=now) == [], specialist="n/a",
        detail="top_k <= 0 must return no records",
    ))

    results.append(EvalResult(
        tool="(parallel)", scenario="parallel-policy",
        passed=valid_parallel_group([0, 2], 3) == [0, 2], specialist="n/a",
        detail="a strictly increasing in-plan group must be accepted",
    ))
    results.append(EvalResult(
        tool="(parallel)", scenario="parallel-policy",
        passed=valid_parallel_group([0, True], 3) is None, specialist="n/a",
        detail="a JSON true is not a step index",
    ))
    _plan, spec = normalise_plan_payload({
        "requires_plan": True, "steps": ["a", "b"],
        "parallel_groups": [[0, 2]],
    })
    results.append(EvalResult(
        tool="(parallel)", scenario="parallel-policy",
        passed=spec["parallel_groups"] == [], specialist="n/a",
        detail="a group pointing past the plan must be dropped entirely",
    ))
    results.append(EvalResult(
        tool="(parallel)", scenario="parallel-policy",
        passed=clamp_parallel_max(0) == 1 and clamp_parallel_max(99) == 8,
        specialist="n/a",
        detail="the fan-out width must clamp to [1, 8]",
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
    _eval_tool_gateway_policy,
    _eval_a2a_agent_cards,
    _eval_a2a_unavailable,
    _eval_langsmith_fallback,
    _eval_memory_and_parallel_policy,
]


def main() -> int:
    all_results: list[EvalResult] = []
    for scenario_fn in _ALL_SCENARIOS:
        all_results.extend(scenario_fn())

    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    failed = [r for r in all_results if not r.passed]
    skipped = [r for r in all_results if r.skipped]

    report = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "skipped": len(skipped),
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
            if result.skipped:
                print(f"  [NOT RUN] [{result.scenario}] {result.tool}: {result.detail}", file=sys.stderr)
            else:
                print(f"  [{result.scenario}] {result.tool}: {result.detail}", file=sys.stderr)
        if skipped:
            print(
                "A case that never ran is not a passing case: install the "
                "missing dependency named above and re-run the eval.",
                file=sys.stderr,
            )

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
