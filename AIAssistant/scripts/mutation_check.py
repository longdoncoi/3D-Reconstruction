"""Mutation check for the load-bearing guards (ADR 0015 / 0011 / 0018).

100% statement/branch coverage says the guards are *executed*, not that they
are *correct*: a test that asserts the wrong direction still covers the line.
This script inverts one decision at a time in each guard and runs the tests
that own it. A mutant is KILLED when the suite fails (the guard's behaviour is
pinned) and SURVIVES when it stays green (the test would not notice the guard
being disabled).

Each mutant is written into the real source file and restored in a ``finally``
that re-reads the file and verifies it came back byte-identical. A lock file
keeps two runs from overlapping, and its presence after a crash refuses to start
again until the source is restored by hand. Run it *alone* — never in the same
step as the suite, which reads exactly these files; CI runs it after the tests
in ``Build-Python-ci.yml``.

An anchor that no longer matches the source counts as STALE and fails the run:
the check may not go green by mutating code that is not there.

Usage (from AIAssistant/):  PYTHONPATH=src python scripts/mutation_check.py
Exit code 0 = every mutant killed and every anchor present; 1 otherwise.
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
        "if executable.casefold() not in _ALLOWED:",
        "if False:",
        ("tests.unit.test_sandbox",),
        "executable allow-list removed (any binary runs)",
    ),
    Mutant(
        "src/ai_assistant/tools/sandbox.py",
        'if "/" in executable or "\\\\" in executable:',
        "if False:",
        ("tests.unit.test_sandbox",),
        "path-shaped executable accepted (/opt/attacker/git runs as git)",
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
    # ── Transport policy: the MCP mount fails closed (ADR 0008) ──────────────
    Mutant(
        "src/ai_assistant/adapters/http/security.py",
        "return not _is_loopback(host) and not policy.mcp_token",
        "return False",
        ("tests.unit.test_http_security",),
        "public bind without AI_MCP_TOKEN left exposed",
    ),
    # ── Classification survives persistence (ADR 0010) ───────────────────────
    Mutant(
        "src/ai_assistant/adapters/persistence/sqlite_store.py",
        "classification=DataClassification(classification),",
        "classification=DataClassification.INTERNAL,",
        ("tests.unit.test_final_edges_a",),
        "reload promotes a RESTRICTED task back to INTERNAL",
    ),
    # ── Remote egress refuses sensitive params (ADR 0002/0010) ───────────────
    Mutant(
        "src/ai_assistant/agents/runner.py",
        "if has_sensitive_keys(params):",
        "if False:",
        ("tests.unit.test_runner",),
        "credential-bearing params dispatched to a remote specialist",
    ),
)


#: Marks an in-flight run. Its absence means "the tree is safe to mutate": a
#: lock left behind by a killed process has to be cleared by hand *after*
#: restoring the source, which is exactly the conversation we want to force.
LOCK_PATH = BASE / "scripts" / ".mutation_check.lock"


def _acquire_lock() -> None:
    """Refuse to overlap another run (or a crashed one still holding it)."""
    try:
        handle = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        message = (
            f"mutation_check: {LOCK_PATH} exists — another run is active, or a "
            "previous one was killed mid-mutation. Verify the source is intact "
            "(git status / git diff), restore it if not, then delete the lock. "
            "Never run this alongside the test suite."
        )
        raise SystemExit(message) from None
    os.write(handle, str(os.getpid()).encode("ascii"))
    os.close(handle)


def _release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def run_mutant(mutant: Mutant) -> str:
    """Mutate one guard, run its tests, restore the file verbatim.

    Returns ``"KILLED"``, ``"SURVIVED"`` or ``"STALE"`` — the last when the
    anchor no longer matches the source, which fails the run rather than
    passing it silently.
    """
    target = BASE / mutant.file
    original = target.read_text(encoding="utf-8")
    if original.count(mutant.original) != 1:
        print(f"  STALE {mutant.label}: anchor not unique/absent in {mutant.file}")
        return "STALE"
    try:
        target.write_text(original.replace(mutant.original, mutant.mutated), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": "src"}
        outcome = subprocess.run(
            [sys.executable, "-m", "unittest", *mutant.tests, "-q"],
            cwd=BASE, capture_output=True, text=True, timeout=600, env=env,
        )
    finally:
        target.write_text(original, encoding="utf-8")
        if target.read_text(encoding="utf-8") != original:
            message = (
                f"{mutant.file} was not restored byte-for-byte; restore it "
                "before running anything against this tree."
            )
            raise RuntimeError(message)
    return "KILLED" if outcome.returncode != 0 else "SURVIVED"


def main() -> int:
    survivors: list[str] = []
    stale: list[str] = []
    _acquire_lock()
    try:
        for index, mutant in enumerate(MUTANTS, start=1):
            verdict = run_mutant(mutant)
            print(
                f"[{index:2d}/{len(MUTANTS)}] {verdict:9s} "
                f"{mutant.file.split('/')[-1]}: {mutant.label}"
            )
            entry = f"{mutant.file}: {mutant.label} (tests: {', '.join(mutant.tests)})"
            if verdict == "SURVIVED":
                survivors.append(entry)
            elif verdict == "STALE":
                stale.append(entry)
    finally:
        _release_lock()
    killed = len(MUTANTS) - len(survivors) - len(stale)
    print(
        f"\nMutants: {len(MUTANTS)}  killed: {killed}  "
        f"survived: {len(survivors)}  stale: {len(stale)}"
    )
    for entry in survivors:
        print(f"  SURVIVED -> {entry}")
    for entry in stale:
        print(f"  STALE    -> {entry} (anchor no longer matches the source)")
    return 1 if survivors or stale else 0


if __name__ == "__main__":
    sys.exit(main())
