# ADR 0016: branch coverage gate raised to 100

Date: 2026-10-12

Status: Accepted

## Context

ADR 0014 installed the coverage gate at `fail_under = 88` against a 91%
measured baseline, with the explicit intent to ratchet it upward rather than
let it rot as a ceiling. The climb finished at 100% statements and 100%
branches (6893 statements, 1844 branches, 1434 tests), so the gate now reads
`fail_under = 100` and `ARCHITECTURE.md` claims 100 instead of 88/~91.

## Decisions

1. **Unreachable defensive duplicates are deleted, not covered.** Three
   guards could never fire and therefore could never be covered honestly:
   - `llama_cpp_backend._load` re-checked `mmproj_path is None` twice after
     the entry guard had already raised for it (GPU path and CPU-retry path).
     Both duplicates are gone; the entry guard is the single enforcement
     point, and `cast(str, …)` carries the narrowing to mypy without adding a
     branch coverage.py would flag.
   - `reason._handle_step_answer` re-checked
     `turn.current_plan_step is not None` after the plan-less early return.
     The check is gone; the index is recorded unconditionally with a comment
     naming the guard that guarantees it.
2. **Race-shaped arcs get race-shaped tests.** The tracer's inner
   `if _tracer is None` False arm is only reachable when another thread wins
   the lock race, so the test wins it deliberately with a lock whose
   `__enter__` pre-sets the sentinel. The shrink loop's exhaustion arm needs
   an unshrinkable anchor at index 0, so the test builds a system-less,
   user-less history whose middle summary the loop is forbidden to touch.
3. **Reload-based tests restore the whole module dictionary.** Re-executing
   the `a2a-sdk` import probe rebinds the module's classes; restoring only
   selected globals left stale class objects that broke `isinstance` checks
   elsewhere. The teardown now puts back the entire captured `__dict__`.
4. **Windows file-lock discipline in tests.** SQLite connections (WAL mode)
   and RotatingFileHandlers are closed before their temporary directories are
   removed; the vacuum rollback test runs against `:memory:` instead of the
   filesystem.

## Consequences

- `python -m coverage report` fails the build on any single uncovered line
  or partial branch. New code must arrive with its tests, not after them.
- `ARCHITECTURE.md` (gate description, verification commands) matches
  `pyproject.toml`.
- Both evals stay green alongside the gate: tool eval 7/7, agentic eval
  106/106, accuracy 1.0.
