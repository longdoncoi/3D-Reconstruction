# ADR 0014: Branch-coverage gate, one environment owner, complexity gate

Date: 2026-10-12

Status: Accepted

## Context

The 3.0 architecture review (Oct 2026) left four structural gaps after ADR 0013:

1. **The coverage gate measured statements only.** Every conditional in the
   ReAct loop, the approval policy and the transport adapters could lose a
   branch without turning CI red. The gate also lived in two places (the CI
   workflow passed `--fail-under=90`, `pyproject.toml` said nothing), so a
   local run and the CI job did not measure the same thing.

2. **`legacy/config.py` was still a second environment owner.** It parsed
   `.env` itself, issued five `os.environ.setdefault` calls, applied three
   `warnings.filterwarnings` rules, and mutated `sys.path` — all as
   import-time side effects, on every production startup. ADR 0001 requires
   every environment read to flow through `settings`, and the AST boundary
   test that enforces this rule explicitly exempts `legacy` because of it.

3. **`server_manager.py` passed Windows-only `creationflags` unconditionally.**
   `subprocess.Popen(creationflags=0x00000200)` raises `ValueError` on POSIX,
   so the Qt process manager could not start the server on Linux or macOS.

4. **No cyclomatic-complexity gate.** The review found a 353-line decision
   function; nothing prevented the next one.

Smaller defects found during the same pass: two `TestCase` class names were
duplicated across files (a failing test's traceback was ambiguous);
`orchestration/supervisor.py` exported five tool-group sets through `__all__`
while keeping their leading underscore, so the only external consumer — the
agentic eval — had to import private names; and `evals/latest_tool_call_report.json`,
a file the eval rewrites on every run, was tracked in Git.

## Decision

### 1. One coverage configuration (`pyproject.toml`)

`[tool.coverage.run]` sets `branch = true` and `source = ["ai_assistant"]`;
`[tool.coverage.report]` sets `fail_under = 88` and omits `legacy`. The CI
workflow now runs `coverage run` / `coverage report` with no flags, so the
config has exactly one home. The measured baseline is **91% of statements and
branches**; the 88 floor leaves ~3pt of headroom, enough that genuine
regressions go red without test noise doing the same.

### 2. `settings` owns the process environment

`settings.py` gained the accessors and the one-time bootstrap the legacy shim
used to inline:

- `force_langgraph_agent()` and `vision_llm_enabled()` — the two toggles that
  were parsed by `legacy/config.py` with its own string comparisons.
- `load_local_dotenv(base_dir)` — the single `KEY=value` parser. Merges with
  `setdefault`, ignores comments/malformed lines, and treats a missing file as
  a no-op.
- `apply_startup_environment_defaults()` — the five library-quieting env keys
  plus the warning filters, idempotent so more than one bootstrap path may
  call it.

`legacy/config.py` now calls these instead of reimplementing them, and its
`sys.path` mutation is gone: importing `ai_assistant.legacy.config` already
requires the `ai_assistant` package to be findable, so the insert could never
help and only mutated global interpreter state. Its public names
(`USE_LANGGRAPH_AGENT`, `FORCE_LANGGRAPH_AGENT`, `ENABLE_VISION_LLM`) are
unchanged, so no legacy importer broke.

### 3. Platform-guarded detach in `server_manager.py`

`_detached_popen_kwargs()` returns `creationflags` on Windows and
`start_new_session=True` elsewhere. The `print(..., flush=True)` calls stay:
the module docstring now records that stdout is the *transport* the Qt client
reads (it keys on the `[SUCCESS]` sentinel), so routing them through
`logging` would break the integration.

### 4. Complexity gate

`select` gains `C90` with `max-complexity = 20`. Enabling it flagged three
functions above the bar, and each was decomposed rather than suppressed —
no `# noqa: C901` remains:

| Function | Before | After | Split |
|---|---|---|---|
| `build_a2a_router` | 47 | 1 | one `_add_*_routes` group per surface (card, send, standard 0.3, legacy `/a2a/tasks`, JSON-RPC), with `_submit_message` / `_text_message` / `_dispatch_json_rpc` owning what more than one endpoint shares |
| `run_langgraph_agent` | 30 | 16 | `_execute_tool` (authorisation → canonicalise → one gateway call) and the state-to-response mapping (`_pending_status`, `_finish_task`, `_close_trace`, `_store_pending_action`, `_pending_response`, `_completed_response`) |
| `prepare_reasoning` | 24 | 5 | one stage function per assembly step (compact, prompt prepend, coding banner, plan progress, selection, UI hint, early-final aggregation, directive, pre-call compact, forced envelope) |

The seven remaining functions sit between 16 and 20 and pass; the gate exists
to stop the next 47-branch function, not to force every loop down to 15.

### 5. Naming and hygiene

- The duplicated `TestCase` class names were disambiguated
  (`ModelLoaderBackendStateTests` / `ModelLoaderLifecycleTests`,
  `ConstrainedCompletionEnvelopeTests`, `A2ARouterFacadeTests`); the
  behaviour-focused files kept the original names.
- The supervisor's tool groups dropped their underscore
  (`CODE_TOOLS`, `RESEARCH_TOOLS`, `TRANSFER_TARGETS`, `VERIFICATION_TOOLS`,
  `WORKFLOW_TOOLS`) — they were in `__all__` already, so the prefix was
  contradicting the module's own declared API. The agentic eval now imports
  public names.
- `evals/latest_tool_call_report.json` is gitignored and untracked.

## Consequences

- A branch regression anywhere in `ai_assistant` fails CI, not just a
  statement regression.
- `legacy` is no longer an environment owner, and the AST boundary test now
  scans it like every other module — its `legacy` exemption is gone, so a
  reintroduced import-time read there fails CI.
- The process manager starts on POSIX, not only Windows.
- The three former complexity hotspots are decomposed rather than waived, so
  the `C90` gate is debt-free: nothing carries a `# noqa: C901`.
- The supervisor's tool groups are public API, so the agentic eval no longer
  imports private names that a refactor could silently rename.
- Verified: `ruff check .` clean, `mypy src/ai_assistant` clean (124 files),
  `lint-imports` 4/4 contracts kept, 902 tests pass, 91% branch coverage.
