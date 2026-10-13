# ADR 0015: unenforced claims are pinned or removed

Date: 2026-10-12

Status: Accepted

## Context

The follow-up pass after ADR 0013/0014 found four things the codebase *claimed*
but nothing *enforced*. Each was individually small; together they meant a
reader had to verify every assertion by hand.

1. **The agent-eval workflow had never been green.** `agent-evals.yml` runs
   `python AIAssistant/evals/run_tool_eval.py` from the repository root with
   only `pydantic` installed. Both eval scripts put `parents[1]` — the
   `AIAssistant/` project root — on `sys.path`, but the package lives at
   `AIAssistant/src/ai_assistant` (the ADR 0010 layout). Every run died at
   `import ai_assistant` with `ModuleNotFoundError`, so the job's real output
   was a red X rather than an eval result. Nothing observed this because the
   workflow is not part of the branch-protection set the review exercised
   locally.

2. **`evals/run_agentic_eval.py` was not wired into CI at all.** It is the
   more meaningful of the two: 106 cases across 17 scenarios covering
   specialist routing, authorisation boundaries, verification and reflection,
   plus the A2A and LangSmith integration points. It is deterministic and
   offline — it needs no key, no network and no model — yet only a human
   running it by hand would ever notice a regression.

3. **The supervisor's remote-specialist lookup was installed by a file no test
   reads.** `orchestration/supervisor.py` exposes
   `set_remote_specialist_lookup` as the dependency-inversion seam that keeps
   orchestration free of adapter imports (ADR 0002), and
   `bootstrap/server.py:_register_optional_protocols` installs
   `resolve_remote_specialist` at startup
   and clears it on shutdown. The existing tests pin the *mechanism* — the
   port shape, that delegation fails closed, that nothing in `orchestration`
   imports an adapter — but none of them opens the composition root. Deleting
   the install line would have left every test green while A2A remote
   delegation stopped being reachable in production.

4. **Three `legacy/` globals had no reader.** `llm_module._chat_handler`,
   `llm_module._RAG_SYSTEM_PROMPT` (which built a full system prompt at import
   time, for nobody) and `rag_module.is_vision_model`. They sat beside four
   globals that *are* live — `llm`, `is_vision_model`, `active_model_desc` and
   `llm_lock`, all declared in `LLMModuleLike` and read by the chat, admin and
   health routers — so "legacy global state" described both the real runtime
   state and dead weight without distinguishing them.

5. **`README.md` documented five paths that do not exist.** The "Hệ sinh
   thái AI Agent Workflow" section described `Instructions.md`, `.prompts/`,
   `.review/`, `.tasks/` and `scripts/`, and named a CI pipeline `build.yml`.
   None of the six is in the repository; an agent following the README would
   have looked for a system prompt that was never written. The section also
   sat under an empty `## Kiến trúc Tổng quan` heading.

## Decision

**An architectural claim is either enforced by something, or it is deleted.**

1. **Both evals run in `agent-evals.yml`**, on every push and PR touching
   `AIAssistant/**`. Each script puts `ROOT / "src"` — not the project root —
   on `sys.path`, so the import works with the workflow's existing minimal
   dependency set. Both reports are uploaded as artifacts
   (`agent-tool-call-trend`, `agentic-policy-trend`) and summarised in the job
   output; both exit non-zero on the first failing case, so they are
   regression tests rather than scored benchmarks.

2. **The composition-root wiring is pinned.**
   `tests/unit/test_a2a_import_order.py` parses `bootstrap/server.py` with
   the AST and asserts two things: startup installs a lookup named
   `resolve_remote_specialist`, and some call clears it with `None`. Reading
   the source is deliberate — importing the script would run the server.

3. **Unread globals are deleted, not annotated.** The three dead names are
   gone. The four live ones are documented *as* live, naming the protocol and
   the routers that read them, so the next reader can tell which half of
   "legacy global state" is load-bearing.

4. **`README.md` documents only paths that exist.** The phantom section is
   replaced with what is actually present — `AGENTS.md`, `.codex/`, the five
   named workflows, and `AIAssistant/ARCHITECTURE.md` with its ADRs — and the
   `AIAssistant/` tree entry now points at the composition root, the package
   layout, the tests and the evals.

## Consequences

- `agent-evals.yml` can go green, and — more importantly — can go red for a
  real reason. Removing the eval step is now a visible regression rather than
  a no-op.
- Deleting the remote-lookup install fails a test instead of silently
  disabling A2A remote delegation.
- `legacy/llm_module.py` no longer computes a system prompt during import.
- An agent that reads `README.md` is pointed at files that are there.
- The bar generalises: the next assertion added to the docs needs either a
  test, a gate, or a deletion.
