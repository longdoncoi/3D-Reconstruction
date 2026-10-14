"""Mutation check for the load-bearing guards (ADR 0015 / 0011 / 0018).

100% statement/branch coverage says the guards are *executed*, not that they
are *correct*: a test that asserts the wrong direction still covers the line.
This script inverts one decision at a time in each guard and runs the tests
that own it. A mutant is KILLED when the suite fails (the guard's behaviour is
pinned) and SURVIVES when it stays green (the test would not notice the guard
being disabled).

Usage (from AIAssistant/):  PYTHONPATH=src python scripts/mutation_check.py
Exit code 0 = every mutant killed; 1 = at least one survived.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Mutant:
    """One deliberate logic inversion in one guard, with the tests that own it."""

    file: str
    original: str
    mutated: str
    tests: tuple[str, ...]
    label: str


MUTANTS: tuple[Mutant, ...] = (
    # ── Sandbox allow-list (ADR 0011) ────────────────────────────────────────
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "if not argv or Path(argv[0]).name.casefold() not in _ALLOWED:",
        "if not argv:",
        ("tests.unit.test_sandbox",),
        "executable allow-list removed (any binary runs)",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        'return "Sandbox: dangerous cmake -E subcommand blocked."',
        "return None",
        ("tests.unit.test_sandbox",),
        "cmake -E <arbitrary cmd> allowed",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        'return "Sandbox: git -c config injection blocked."',
        "return None",
        ("tests.unit.test_sandbox",),
        "git -c core.sshCommand injection allowed",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        "return f\"Sandbox: subcommand '{subcmd}' is not in the {executable} allow-list.\"",
        "return None",
        ("tests.unit.test_sandbox",),
        "unknown subcommand allowed",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        'if target_name in _DENIED_WRITE_PATHS or any(p in str(Path(path).resolve()).lower() for p in ["sandbox.py", ".env", "security.py"]):',
        "if False:",
        ("tests.unit.test_sandbox",),
        "write-deny on .env/policy files removed",
    ),
    # ── HITL approval gate ───────────────────────────────────────────────────
    Mutant(
        "src/ai_assistant/application/tools.py",
        "if spec.requires_approval and not self._grants.covers(",
        "if False and not self._grants.covers(",
        ("tests.unit.test_tool_gateway_guards", "tests.unit.test_deterministic_orchestrator"),
        "approval-required tools run without approval",
    ),
    # ── Contract: Qt history bound ───────────────────────────────────────────
    Mutant(
        "src/ai_assistant/agents/models.py",
        "history:             list[dict] = Field(default_factory=list, max_length=200)",
        "history:             list[dict] = Field(default_factory=list, max_length=20)",
        ("tests.contract.test_agent_contract_qt",),
        "history bound regressed 200 -> 20 (long sessions 422 again)",
    ),
    # ── Remote-leg opt-in (ADR 0018) ─────────────────────────────────────────
    Mutant(
        "src/ai_assistant/settings.py",
        'return values.get("AGENT_HYBRID_POLICY", "local_only").casefold()',
        'return values.get("AGENT_HYBRID_POLICY", "cloud_allowed").casefold()',
        ("tests.unit.test_inference",),
        "default hybrid policy flipped to cloud_allowed",
    ),
    # ── /health honesty ──────────────────────────────────────────────────────
    Mutant(
        "src/ai_assistant/adapters/http/health_routes.py",
        'status = "not_ready" if unready else ("degraded" if reasons else "ready")',
        'status = "ready"',
        ("tests.unit.test_http_health_routes",),
        "/health always claims ready",
    ),
    # ── RAG cache seal fail-closed ───────────────────────────────────────────
    Mutant(
        "src/ai_assistant/rag/cache_integrity.py",
        "if key is None or not recorded:\n        return False",
        "if key is None or not recorded:\n        return True",
        ("tests.unit.test_rag_cache_integrity", "tests.unit.test_index"),
        "unverifiable cache trusted (fail-open)",
    ),
    # ── Cancel that races the worker ─────────────────────────────────────────
    Mutant(
        "src/ai_assistant/application/coordination.py",
        'if previous is not None and previous.cancelled and previous.metadata.get("pending_cancel"):',
        "if False:",
        ("tests.unit.test_task_coordinator_lifecycle",),
        "pending cancel lost when the worker registers its run",
    ),
    Mutant(
        "src/ai_assistant/application/coordination.py",
        "if record.status in self._TERMINAL and not record.cancelled:",
        "if False:",
        ("tests.unit.test_task_coordinator_lifecycle",),
        "cancelling a finished run resurrects it",
    ),
)


def run_mutant(mutant: Mutant) -> bool:
    """Apply one mutation, run its tests, restore the file. True = killed."""
    target = BASE / mutant.file
    original = target.read_text(encoding="utf-8")
    if original.count(mutant.original) != 1:
        print(f"  SKIP  {mutant.label}: anchor not unique in {mutant.file}")
        return True
    try:
        target.write_text(original.replace(mutant.original, mutant.mutated), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": "src"}
        outcome = subprocess.run(
            [sys.executable, "-m", "unittest", *mutant.tests, "-q"],
            cwd=BASE, capture_output=True, text=True, timeout=600, env=env,
        )
    finally:
        target.write_text(original, encoding="utf-8")
    return outcome.returncode != 0


def main() -> int:
    survivors: list[str] = []
    for index, mutant in enumerate(MUTANTS, start=1):
        killed = run_mutant(mutant)
        verdict = "KILLED  " if killed else "SURVIVED"
        print(f"[{index:2d}/{len(MUTANTS)}] {verdict}  {mutant.file.split('/')[-1]}: {mutant.label}")
        if not killed:
            survivors.append(f"{mutant.file}: {mutant.label} (tests: {', '.join(mutant.tests)})")
    print(f"\nMutants: {len(MUTANTS)}  killed: {len(MUTANTS) - len(survivors)}  survived: {len(survivors)}")
    for entry in survivors:
        print(f"  SURVIVED -> {entry}")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
