# AI Agent Platform architecture

The server is a **policy-first Clean Architecture platform** with enforced
layer boundaries, a single execution gateway, and protocol isolation. The
The composition root is `src/ai_assistant/bootstrap/server.py`; the production
package is `src/ai_assistant`.

```text
Qt / HTTP / MCP / A2A
          │
       adapters
          │
 application use-cases ─── plugin registry ─── builtin plugins
          │
      domain contracts
```

## Runtime rules

- `ToolExecutionService` is the sole tool policy and execution gateway, and the
  composition root wires its event sink: every execution — agent loop, HITL
  approval, desktop action, A2A task worker — is recorded as one redacted JSON
  line on the `agent.tools` logger (`observability/tool_events.py`). Without
  that sink the executions that do not come from the agent loop left no trace.
- The RAG cache is unpickled only after an HMAC check: `save_cache` seals the
  FAISS index, chunk pickle and BM25 pickle with a key stored outside the
  cache dir (`<data_dir>/.secrets/cache_hmac_key`, 0600), and `load_cache`
  (`rag/cache_integrity.py`) treats a missing key, a missing record or a
  mismatched digest as a cache miss — never as code to run. Threat model
  stated in the module: guards against tampering without read access and
  silent corruption, not against an attacker already running as this user.
- Cancellation recovery is measured where the runner actually stops:
  `TaskCoordinator.cancel()` stamps `cancelled_at`, `finish()` reports the
  delta through an observer injected by the composition root into the
  `agent_recovery_seconds` Prometheus histogram (SCORECARD #4). The
  coordinator itself stays free of observability imports (import contract).
  Spans leave the process only when `AGENT_OBSERVABILITY=1` and
  `AGENT_OTLP_ENDPOINT` names a collector (`observability/tracing.py`); a
  missing exporter package degrades to in-process spans with one warning.
- Concurrency is serialized per session, never globally: `AgentService.execute`
  holds a per-session `RLock` for the whole run (`agents/service.py`), so two
  runs of one session can not share a checkpointer `thread_id` or clobber the
  coordinator's record while other sessions keep running in parallel — and
  `cancel`/`approve` take no session lock, so they keep working while a run
  holds one. The runner's abort branch finishes the record
  (`success=False`, error attached) before re-raising, so a crash can no longer
  leave a task parked in `running` forever.
- Per-node latency is measured in-process: `span()` records each observation
  into a bounded window (512/node), and `p95_latency_by_node()`
  (`observability/metrics.py`) returns `{node: p95}` (nearest-rank) — the
  SCORECARD #2 signal without a PromQL round-trip. Only timed graph nodes
  (`plan`/`reason`/`tool`/`reflect`) appear; `approval` is a counter, not a
  timed span.
- Outbound A2A calls carry the active trace: `TrustedA2AClient.submit` attaches
  the W3C `traceparent` header from `current_traceparent()` (absent when there
  is no valid span to propagate). Inbound extraction — continuing a caller's
  trace on our own A2A routes — is not wired yet.
- `settings.py` is the sole owner of the process environment (ADR 0001, 0014):
  it parses `.env` (`load_local_dotenv`), applies the library-quieting
  defaults and warning filters (`apply_startup_environment_defaults`), and
  exposes every toggle as an on-demand accessor. The composition root calls
  both functions before its own import-time settings reads
  (`bootstrap/server.py`), so an operator's `.env` reaches the running server
  and not only the tests; no module re-implements the `KEY=value` parser and no
  feature reads the environment directly at import time.
- Coverage and complexity are gated in `pyproject.toml` (ADR 0014), not in
  the CI command line: branch coverage is on with a 100 floor at a 100%
  measured baseline, and `C90` caps every function at complexity 15. The gate
  is debt-free — the nine functions that exceeded it across both tightenings
  (`build_a2a_router`, `run_langgraph_agent`, `prepare_reasoning`,
  `build_chat_router`, `constrained_completion`, `plan_node`, `tool_node`,
  `build_index_from_scratch`, `tool_list_directory`) were decomposed into
  named helpers, so nothing carries a `# noqa: C901`.
- Lint breadth is `E/W/F/I/BLE/S/RUF/C90/N/TRY` (ADR 0014, extended ADR 0016):
  `N` pins the few names that mirror external APIs (A2A wire fields, faiss
  doubles, the PascalCase entry point) via reasoned `# noqa`s, and `TRY`
  keeps `return` out of `try` bodies, logs tracebacks where they are caught
  and hoists long raise messages — except where the exception type itself is
  a tested contract (`discover()` catches `ValueError`, the decoders raise
  `RuntimeError`), which stays `noqa`d with the reason attached.
- Every transport and orchestration strategy reaches tools through an injected
  `PlatformToolGateway` (`bootstrap/runtime.py`). The gateway is built by the DI
  container and constructor-injected into the agent service, MCP adapter and A2A
  orchestrator; no module-level service locator remains. The MCP adapter binds
  the gateway once per `McpToolServer(gateway)` instance at construction time,
  `TaskCoordinator` is created in the container and injected, and the agent
  tool catalog is built by the composition root
  (`create_tool_registry(rag_context=rag_module)`), which closes the
  `RagContextPort` (`ports.py`) over the `rag_search` handler — the tool layer
  imports no bootstrap module, and a registry built without a provider reports
  the index as unavailable instead of reaching for a global.
  `ToolRegistry` is a catalog and cannot execute (ADR 0003).
- The LangGraph runner (`agents/runner.py`) is the single production decision
  loop. A2A reuses it through the `AgentOrchestrator` port
  (`adapters/orchestration/langgraph.py`); the application layer never imports
  LangGraph and holds no loop of its own. The framework-independent fallback is
  an adapter (`adapters/orchestration/deterministic.py`) injected by tests/embeds.
  The runner receives its wiring as one frozen `RunDeps` bundle
  (`agents/deps.py`, ADR 0013) plus the per-invocation arguments. Importing
  `ai_assistant.orchestration` pulls no graph — the package re-exports only
  `AgentState`, so the agent layer imports cleanly without langgraph installed
  and the `None` engine genuinely degrades to `ServiceUnavailableError`
  instead of the import itself failing (AST-guarded in
  `test_architecture_boundaries.py`, review finding 5).
- There is exactly one `ToolSpec` type — the policy record in `domain/tools.py`.
  The executable catalog entry is `tools/definition.py:ToolDefinition`; the old
  `ai_assistant.core` package is gone.
- Capability scopes are enforced across the shared engine: the A2A principal is
  bound with `bind_principal` for the call and read by `PlatformToolGateway`.
- The hybrid-policy gate sits *outside* the decode-error wrappers: both
  `constrained_completion` and `structured_completion` refuse through
  `_remote_leg_allowed` before the transport and before the `try`, so a policy
  refusal is never relabelled as a JSON decoding failure (ADR 0018, review
  finding 16). The llama.cpp grammar cache in `agents/completion.py` is keyed
  by the schema *text* under its own lock — a `hash()` key could serve one
  schema's grammar for another, and the compile is not safe to run twice
  (finding 26).
- Pending approvals are rehydrated from the file-backed store when
  `AgentService` starts, so A2A resume survives a process restart. A refused
  load or save is logged, never swallowed — the API still answers
  `pending_approval` afterwards (finding 25) — and a parked entry snapshots
  the live `steps`/`messages` lists, so a later save cannot rewrite what the
  client was already handed (finding 9).
- The dead-API surface is gone rather than documented (ADR 0015, findings
  17/21/22): `route_task`, `force_langgraph_agent()` and
  `AgentExecuteRequest.force_langgraph`, `ToolAppAgent`, the write-only
  `state.last_reflection`, the never-consumed `specialist_instruction`
  handoff, the `supervisor_route` parameter that four layers threaded and no
  caller ever passed (the trace label is now the constant it always carried),
  the tests-only `orchestration/compat` shim, the `idempotent` flag that the
  whole catalog chain propagated and nothing enforced, `file_tools.py`'s
  orphaned citation fast path (duplicating `summarize.format_code_citation`),
  and the `generate_action_id`/`plan_hints` guards no caller could reach. The
  supervisor pipeline that remains computes once per distinct call:
  `run_langgraph_agent` memoizes the delegation for each `(tool, params)` pair,
  so dispatch, execution, verification and reflection share one
  classification instead of four (finding 14, pinned by
  `test_one_delegation_is_shared_across_the_tool_callbacks`).
- The P2/P3 design-debt batch made four implicit conventions type- or
  table-enforced (findings 7/8/10/13/15/27): the `USE_LANGGRAPH_AGENT` toggle
  that degraded resume differently from `execute` is gone (the LangGraph loop
  is the only production engine, so resume runs it whenever the graph is
  wired); grant minting is one domain policy
  (`domain.approvals.issue_grant_pair`) shared by the LangGraph-resume and
  durable A2A loops; the runner is handed only a `ToolExecutionGateway` and can
  never mint a grant — that lives on the separate `ApprovalGrantIssuer` port
  the approval services hold (pinned by an AST boundary test); supervisor
  least-privilege is a fail-closed `ALLOWED_TOOLS` table consulted by
  `authorise`, not a side effect of how `delegate` classified the call; every
  supervisor/observability/LLM callable is an overridable `AgentRunContext`
  field rather than a closed-over module function; and a resumed run
  re-validates its tool calls with the real validator instead of a no-op
  (finding 27 — a resumed run may emit *new* calls).
- The following debt batch (findings 18, 24-pair 7, 19) tightened the
  ReAct loop's structure: `tool_node` is a thin coordinator over `_dispatch`
  (cancel gate, pending-call pairing, payload validation, execution, error
  counting) and `_record` (result/verification steps, circuit breaker, RAG and
  completion shortcuts, reflection fast paths) — finding 18; tool-result
  semantics live in one domain contract (`domain.tools.tool_result_succeeded`,
  `tool_result_has_evidence`, `EVIDENCE_FIELDS`) consumed by both
  `supervisor.verify_result` and the coding-workflow classifier, so the two
  evidence tables can no longer drift (blank-string content is evidence
  nowhere, `git_diff` needs its `content` field on both call sites) — finding
  24 pair 7; and every step-discriminator read goes through
  `state.has_step_type(step, <state.StepType>)`, a closed literal vocabulary
  that makes a typo'd routing string a mypy error instead of a condition that
  silently never matches — the read side of finding 19. The write side of the
  same finding is closed too: `AgentState.steps` is `list[state.Step]`, a
  TypedDict whose `type` key is required and closed and whose payload keys
  form a declared vocabulary, so an unknown discriminator or an undeclared
  payload key at any construction site is a mypy error (0 issues over 128
  files, measured 2026-10-14). The resume path is inside the net:
  `runner.run_langgraph_agent` and `ContextService` type `initial_steps` as
  `list[Step]`, and the services that build resume steps
  (`agents.approval_service`, `agents.ui_action_service`) are checked against
  the same vocabulary. Steps deserialized from persistence or HTTP still
  cross the trust boundary as runtime dicts (`Any`); validating their shape
  is a runtime concern the type layer deliberately does not own.
- The HITL approval *policy* is domain-owned (`domain/approvals.py`):
  `approval_covers_invocation` (constant-time fingerprint comparison),
  `derive_approval_scope` and `build_approval_preview`. The reason node
  consults it and reads run state, but holds no approval rule of its own
  (ADR 0009).
- MCP is an external adapter; agent execution does not call `/mcp` locally.
- A2A owns durable task lifecycle. Its only remote transport is
  `adapters/a2a_client.TrustedA2AClient` (allowlist-only, classification-gated);
  `A2ARouter` is a routing facade over it and reports `remote_selected`,
  `rejected_*`, or `degraded_dependency` outcomes. The allowlist is the single
  merged deployment set from `settings.a2a_trusted_endpoints()` (config
  `[a2a] trusted_endpoints` + `A2A_REMOTE_AGENTS`); with none configured every
  remote dispatch fails closed (ADR 0012).
- The supervisor reaches A2A only through dependency inversion: it exposes
  `set_remote_specialist_lookup`, and the composition root installs
  `a2a_protocol.resolve_remote_specialist`. No orchestration module imports an
  adapter (enforced by an AST boundary test), so there is no import cycle.
- LangGraph stays isolated behind its engine (`orchestration/graph.py`) and
  checkpointer adapter (`adapters/persistence/checkpointing.py`), and LangSmith
  behind `observability/langsmith_integration.py`; AST boundary tests fail CI if
  any other module imports those frameworks (ADR 0012).
- LangSmith remains optional and must not receive restricted/regulated data.
- Tool inputs are validated from the catalog schema at the single execution
  gateway, including requests received through MCP.
- Browser CORS origins are an explicit deployment setting in `config/base.toml`.

## A2A API

The primary A2A 0.3 HTTP+JSON surface is `POST /message:send`, `GET
/tasks/{id}`, `GET /tasks`, `POST /tasks/{id}:cancel`, and `POST
/tasks/{id}:subscribe`; discovery remains at `/.well-known/agent.json`.
The Agent Card advertises its REST interface and requires an explicit
`A2A-Version: 0.3` when the client sends a version header.

Approval and Qt acknowledgements pause the durable task in `input-required`.
They resume only through the declared
`urn:3d-reconstruction:a2a:approval-resume:v1` extension at `POST
/tasks/{id}:resume`; execution context is persisted internally and never
exposed in artifacts. The former `/a2a/tasks/*` and JSON-RPC `tasks/*`
endpoints remain compatibility adapters while callers migrate.

## Plugins

Builtin compatibility tools are registered during bootstrap. Third-party
plugins use the `ai_assistant.plugins` Python entry-point group and are loaded
only when their `plugin_id` is explicitly enabled in `config/plugins.toml`.

See `Docs/adr/` for the binding decisions and `config/*.toml` for deployment
profiles. Install the package during development with `pip install -e AIAssistant`.

## Agent service decomposition (ADR 0004)

The former monolithic `AgentService` (542 lines) has been decomposed into:

- `AgentService` — facade preserving the public interface; delegates to sub-services.
- `ApprovalService` — Human-in-the-Loop approval lifecycle (session binding,
  rejection, execution, LangGraph resume).
- `UIActionService` — Desktop UI action continuation (result processing,
  queued action chaining, LangGraph resume).

Each sub-service receives the same injected dependencies (`llm_runtime`,
`pending_actions`, `pending_lock`, `tool_gateway`, `tool_registry`) and can be
tested independently.

### Shared run wiring (`agents/context.py`, ADR 0012)

The three services previously each repeated the same ~26 keyword arguments
when calling `run_langgraph_agent` (the LLM runtime, tool registry/gateway,
pending store, and every supervisor/observability/LLM callable) — ~78
duplicated lines, and a new injected dependency meant editing three call
sites. `AgentRunContext` now owns that block once: it is a frozen record of
the wiring a run needs, and each service calls `AgentRunContext.run(...)` with
only the arguments that vary per invocation. The main execute path keeps the
real tool validator, action-id generator and A2A router; the approval and
desktop-ack resume paths use the named no-op `RESUME_*` helpers with
`a2a_router=None`.

### Runner dependency bundle (`agents/deps.py`, ADR 0013)

`run_langgraph_agent` itself used to accept 46 keyword-only parameters — the
LLM runtime, the tool catalog/gateway, the pending store and lock, the task
coordinator, ~20 supervisor/observability/LLM callables, the graph engine
class and the A2A transport. `AgentRunContext` deduplicated the *call sites*,
but the runner signature was still a DI container as an argument list.

`RunDeps` (a frozen dataclass) now owns that block once. The runner takes
`deps: RunDeps` plus only the per-invocation arguments (prompt, task, resume
state, approval grant, event sink) — 21 parameters instead of 46. Adding one
injected dependency means adding one field to `RunDeps`; no call site changes.
`validate_tool_call_fn` and `generate_action_id_fn` stay per-invocation so the
resume paths can swap them for the named no-op helpers.

### Reason node decomposition (ADR 0013)

`orchestration/nodes/reason.py:reason_node` was a single 353-line function
with 19 top-level branches. It is now a 78-line dispatch sequence; each
outcome the model can produce is handled by one focused private function
(`_handle_step_answer`, `_handle_final_answer`, `_handle_validation_error`,
`_handle_forbidden_transfer`, `_guard_step_action_mismatch`,
`_guard_duplicate_failed_call`, `_dispatch_tool_call`) that receives the
shared `_Turn` record and returns the partial state update or `None` to fall
through. The context-budget protocol is now typed: `orchestration/state.py`
owns `CONTEXT_EXHAUSTED_MESSAGE` and `is_context_exhausted()`; the runner's
`complete` callback returns the constant and the reason node calls the
predicate — the literal no longer exists in two places.

## A2A adapter decomposition (ADR 0005)

The former monolithic `adapters/a2a.py` (310 lines) has been decomposed into:

- `adapters/a2a_models.py` — Pydantic request/response schemas (A2A 0.3 compliant).
- `adapters/a2a_payloads.py` — payload builders, state mapping, sensitive data redaction.
- `adapters/a2a_router.py` — FastAPI router with all standard and legacy endpoints.
- `adapters/a2a.py` — backward-compatible re-exports.

## Legacy migration (ADR 0006, 0012, 0013, 0014, 0018)

The legacy `modules/` package has been **removed**, and so has the
`src/ai_assistant/legacy/` shim package that briefly replaced it — the
directory this section once described no longer exists. What the shims held
now sits with its owner:

- Environment parsing, the library-quieting defaults and the warning filters
  belong to `settings` alone (ADR 0014); the composition root calls them once
  at startup. The toggles the shim exposed as constants are accessors now
  (`vision_llm_enabled`; the `USE_LANGGRAPH_AGENT` loop toggle is gone — the
  LangGraph loop is the only production engine).
- The live model state — `llm`, `is_vision_model`, `active_model_desc`,
  `llm_lock`, `USE_VISION_LLM` — is `bootstrap/server.py`'s `_LLMModule`,
  which `load_model` owns; the HTTP routers read it through `LLMModuleLike`
  (`adapters/http/_types`). `_chat_handler` and the eagerly-computed
  `_RAG_SYSTEM_PROMPT`, which built a system prompt at import time for nobody,
  had no reader and are gone (ADR 0014).
- The live index state — `knowledge_index`, `knowledge_chunks`, `bm25_index`,
  `embed_model_ref`, `_reranker`, guarded by `rag_lock` — is
  `bootstrap/server.py`'s `_RAGModule`, which `initialize_rag` owns and the
  admin routes reset. Its `is_vision_model` had no reader and is gone.

The one-line re-export shims
`legacy/{a2a_protocol,action_manifest,agent_module,checkpointing,lsp_client,mcp_server}.py`
were deleted in ADR 0012, and `adapters/legacy_agent.py` in ADR 0013: each had
zero production importers, and the latter built a second `AgentService` and a
second `ToolRegistry` at import time.

All production code imports from `ai_assistant.*` directly. The former
`LangGraphAgent` exports now live in `orchestration.graph` — the tests-only
`compat` shim had zero production importers and was removed by the ADR 0015
dead-code sweep — and the Qt process
manager is `bootstrap.server_manager`. An import-linter contract and the AST
boundary tests keep `modules` from being resurrected.

`bootstrap/server_manager.py` is console-facing by contract: its `print(..., flush=True)`
lines *are* the transport the Qt client reads (it keys on the `[SUCCESS]`
sentinel), so they are not candidates for `logging`. Its child-process detach
flags are platform-guarded — `creationflags` on Windows, `start_new_session`
elsewhere — because a non-zero `creationflags` raises `ValueError` on POSIX
(ADR 0014).

The AST boundary test that forbids import-time environment reads used to skip
`legacy/` because of `config.py`. That exemption is **removed** (ADR 0014) and
the package it covered was deleted with it: every module under
`src/ai_assistant/` is scanned — `settings` as the approved reader, `bootstrap`
as the composition root — so a reintroduced inline read fails CI.

## Streaming endpoints (ADR 0005)

The legacy `/a2a/tasks/{id}/events` and the standard `/tasks/{id}:subscribe`
SSE endpoints share one polling loop (`adapters/a2a_router._task_pages`) so the
cursor/terminal/sleep logic exists in exactly one place; each endpoint keeps its
own wire format (`event:` frames vs plain `data:` status updates).

## Type safety (ADR 0007)

The `LLMRuntime` protocol now uses concrete types:

```python
class LLMRuntime(Protocol):
    llm: LLMBackend | None
    llm_lock: LockLike
```

This replaces the former `llm: Any` / `llm_lock: Any` declaration, enabling
static type checking across the agent layer. `LLMBackend` is owned by
`ai_assistant.ports` (contract 1 keeps `ports` away from `ai_assistant.llm`)
and re-exported by `ai_assistant.llm.contracts`; `LockLike` is a structural
lock protocol satisfied by both `threading.Lock` and `threading.RLock`.

## Transport hardening decisions

Sprint hardening is pinned by dedicated ADRs:

- ADR 0008 — loopback peer + `Origin` allow-list + optional bearer per surface;
  remote A2A cannot be enabled without `AI_A2A_TOKEN` (fail closed).
- ADR 0009 — approval is a single-use, invocation-bound capability
  (`approval_fingerprint` + `InMemoryApprovalGrantStore`), not a boolean flag.
- ADR 0010 — the domain owns one redaction policy (`SENSITIVE_KEYS` /
  `redact_value`) for A2A payloads, task events and logs.
- ADR 0011 — the command sandbox rejects inline interpreter code (`-c`, `-m`,
  `-`) and shell substitution (`$(`/backticks).
- ADR 0013 — `RunDeps` bundle, reason-node decomposition, typed
  context-exhausted sentinel, single `runtime_unconfigured` wire shape,
  dead-code removal (`a2a_protocol` payload copies, `legacy_agent` adapter),
  LSP tools activated.
- ADR 0014 — branch-coverage and cyclomatic-complexity gates live in
  `pyproject.toml` (one home, so local and CI runs measure the same thing);
  `settings` is the sole environment owner, and the last second owner
  (`legacy/config.py`) was deleted with the rest of `legacy/`; the Qt process
  manager's detach flags are platform-guarded; the supervisor's tool groups
  are public API.

## Verification

```bash
cd AIAssistant
ruff check .
python -m mypy src/ai_assistant
python -m compileall -q evals src
PYTHONPATH=src lint-imports --no-logo
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
python -m coverage run -m unittest discover -s tests -p "test_*.py"
python -m coverage report   # branch coverage, floor 100, baseline 100
```

`tests/integration/test_architecture_boundaries.py` enforces the layer rules
in ADR 0001 and the single execution gateway / single tool model / canonical
engine in ADR 0003 using pure AST checks.
`tests/integration/test_agent_orchestration.py` pins the A2A ↔ LangGraph
contract translation and least-privilege principal binding;
`tests/integration/test_pending_durability.py` pins restart-safe approval
rehydration; `tests/unit/test_a2a_router.py` pins the single A2A transport;
`tests/integration/test_a2a_streaming_e2e.py` drives the full A2A lifecycle
over the JSON-RPC transport — submit → working → completed, `:cancel`,
`:resume`, `message:stream`, task events, and capability rejection — through
the real FastAPI router with in-memory persistence.

Edge-case test coverage:
- `tests/unit/test_concurrent_approval.py` — thread-safe PendingActionStore ops.
- `tests/unit/test_session_serialization.py` — one run per session: two runs of
  the same session never overlap, a second session runs while the first still
  holds its lock, and the lock object is per session (pins the `RLock` in
  `AgentService.execute`; the runner's abort branch finishing the record is
  pinned in `tests/unit/test_runner.py`).
- `tests/unit/test_plugin_reentry.py` — plugin loader idempotency and duplicate handling.
- The remaining review findings are pinned where they live: a crashed run
  still finishes its record and a parked entry snapshots the live state lists
  (`tests/unit/test_runner.py`), the ADR 0018 gate refuses on *both* decoders
  without being relabelled as a decode failure
  (`tests/unit/test_completion_guard_paths.py`,
  `tests/unit/test_completion_envelopes.py`), a refused pending-store
  load/save is logged (`tests/unit/test_pending_store_edges.py`), and the
  orchestration package stays framework-free
  (`tests/integration/test_architecture_boundaries.py`).
- `tests/unit/test_a2a_streaming.py` — SSE streaming, payload safety, redaction.
- `tests/unit/test_checkpoint_recovery.py` — backend selection and fallback.
- `tests/unit/test_completion_envelopes.py`, `tests/unit/test_completion_guard_paths.py`,
  `tests/unit/test_a2a_client.py`, `tests/unit/test_a2a_protocol_guards.py`,
  `tests/unit/test_task_lifecycle_guards.py`, `tests/unit/test_llm_backend.py` —
  targeted coverage for the completion decoders, the A2A client routing
  policy, the task worker lifecycle, model loading, the llama.cpp backend and
  the FastAPI composition boundary. (The old `test_coverage_boost*.py` batches
  were split verbatim into these topic files; no batch file remains.)
- `tests/unit/test_index.py`, `tests/unit/test_loaders.py` — image-aware
  embedding, the IVF corpus path, scan edge cases, encoding/docx/pdf/email
  loader fallbacks, hierarchical markdown sections and the sliding-window
  overlap guard (lifts `rag/index.py` and `rag/loaders.py` to 100%).
- `tests/unit/test_server_manager.py`, `tests/integration/test_server_module.py`
  — the supervisor's `main()` decision matrix (same model / switch model /
  shutdown survivor / startup timeout / missing log, plus the tail loop) and
  the composition-root branches the module import itself cannot reach (A2A/MCP
  deployment-profile matrix, lifespan lifecycle, chatbot-agent holder, win32
  proactor shim), lifting both `bootstrap/server.py` and
  `bootstrap/server_manager.py` to 100%.
- `tests/unit/test_reason_node_guards.py` — the ReAct dispatch loop's
  self-correction guards (empty answer, hallucinated inline answer, malformed
  tool call, the `transfer_to_toolapp_agent` bounce, the desktop step/action
  mismatch on both its reject and exhausted-fallthrough paths, and the
  duplicate-failed-call guard) plus the post-reason router
  (lifts `orchestration/nodes/reason.py` to 100%).

Coverage of `src/ai_assistant/*` is **100.00% measured (2026-10-14)** with
branch coverage enabled: 7 840 statements and 2 068 branches, 0 missed and 0
partial, produced by 1 651 tests (1 skipped, 0 failures). Because coverage
alone cannot detect a test that asserts the wrong direction, the load-bearing
guards (sandbox allow-list, HITL approval, remote policy, `/health`, cache
seal, cancel races) are additionally mutation-checked:
`PYTHONPATH=src python scripts/mutation_check.py` inverts one decision at a
time and requires the owning tests to fail — 16/16 killed as measured
2026-10-14, a count CI re-runs on every push (the check fails if an anchor
stops matching the source, so it cannot go green on stale mutants). The number is a measurement, not a target: it reproduces with
`faiss-cpu`, `rank-bm25` and `mcp` installed (`requirements-test.txt`), and a
stripped environment measures less because the optional-dependency branches
flip with it. The declared exclusions are the two
`if __name__ == "__main__":` process entry points in `pyproject.toml`'s
`exclude_also`, which covers the composition roots `bootstrap/server.py` and
`bootstrap/server_manager.py`; their platform shims, deployment-profile
branches and entry points are covered through extracted helpers and targeted
tests rather than excluded. Five `# pragma: no cover` lines remain, each
naming a branch this environment cannot drive: the `asyncio` ImportError guard
(`agents/context.py`), the `sys.path` idempotence line and the Qt-launched
entry point (`bootstrap/server_manager.py`), and the two `tool_name is None`
arms that `_handle_final_answer` catches first
(`orchestration/nodes/reason.py`). Run it with:

```bash
PYTHONPATH=src python -m coverage run -m unittest discover -s tests -p "test_*.py"
PYTHONPATH=src python -m coverage report
```

There are deliberately no flags: `pyproject.toml` owns the whole coverage
configuration (`branch = true`, `source = ["ai_assistant"]`, `exclude_also`
and the `fail_under = 100` floor), so a local run and CI measure the same
thing.
`.github/workflows/Build-Python-ci.yml` enforces it next to the lint job and
the full test suite.

## Agent evals

Two deterministic evals run in `.github/workflows/agent-evals.yml` on every
push and PR touching `AIAssistant/**`:

- `evals/run_tool_eval.py` — constrained tool-call envelopes: schema
  validation, the `_validation_error` path and the resume shapes.
- `evals/run_agentic_eval.py` — supervisor policy: routing to the right
  specialist, authorisation boundaries, verification and reflection across
  17 scenarios (106 cases), plus the A2A and LangSmith integration points.

Both are offline and exit non-zero on the first failing case, so they are
regression tests rather than scored benchmarks. Neither imports a real LLM.

Two more scripts are manual, not part of that job:

- `evals/run_model_in_loop_eval.py` — grades the **real planner**
  (`PLANNER_PROMPT` through the loaded backend) against a golden set of
  literal prompts and reports kind accuracy + p95 latency. With no backend it
  prints `SKIP` and exits 0 *without* an accuracy figure, so a green run can
  never be a synthetic number. (The earlier version called a rule classifier
  whose vocabulary could never match the expectations, and
  `evals/run_golden_set_eval.py` — which only asserted that its own fixture
  existed — was deleted as fake-green.)
- `evals/run_tool_eval.py` and `evals/run_agentic_eval.py` are the two the CI
  job runs (above).

Each script puts `src/` — not the project root — on `sys.path`. Pointing at
the root left `ai_assistant` unimportable and made the whole eval job fail
with `ModuleNotFoundError`; the wiring is now pinned by running both scripts
in CI (ADR 0014).
