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
 application use-cases ─── plugin registry ─── legacy/builtin plugins
          │
      domain contracts
```

## Runtime rules

- `ToolExecutionService` is the sole tool policy and execution gateway.
- `settings.py` is the sole owner of the process environment (ADR 0001, 0014):
  it parses `.env` (`load_local_dotenv`), applies the library-quieting
  defaults and warning filters (`apply_startup_environment_defaults`), and
  exposes every toggle as an on-demand accessor. No other module reads the
  environment at import time, and no module re-implements the `KEY=value`
  parser. `legacy/config.py` — the last second owner — now delegates.
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
  tool catalog is built by the composition root (`create_tool_registry()`).
  `ToolRegistry` is a catalog and cannot execute (ADR 0003).
- The LangGraph runner (`agents/runner.py`) is the single production decision
  loop. A2A reuses it through the `AgentOrchestrator` port
  (`adapters/orchestration/langgraph.py`); the application layer never imports
  LangGraph and holds no loop of its own. The framework-independent fallback is
  an adapter (`adapters/orchestration/deterministic.py`) injected by tests/embeds.
  The runner receives its wiring as one frozen `RunDeps` bundle
  (`agents/deps.py`, ADR 0013) plus the per-invocation arguments.
- There is exactly one `ToolSpec` type — the policy record in `domain/tools.py`.
  The executable catalog entry is `tools/definition.py:ToolDefinition`; the old
  `ai_assistant.core` package is gone.
- Capability scopes are enforced across the shared engine: the A2A principal is
  bound with `bind_principal` for the call and read by `PlatformToolGateway`.
- Pending approvals are rehydrated from the file-backed store when
  `AgentService` starts, so A2A resume survives a process restart.
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

## Legacy migration (ADR 0006, 0012, 0013, 0014)

The legacy `modules/` package has been **removed**. Implementations live in
`src/ai_assistant/legacy/`:

- `legacy/config.py` — re-exports from `settings` + legacy constants. It is no
  longer an environment owner: the `.env` parser, the startup env defaults and
  the warning filters moved to `settings` (ADR 0014), and its `sys.path`
  mutation — which could never help, since importing this module already
  requires `ai_assistant` to be findable — is gone. Its public names
  (`USE_LANGGRAPH_AGENT`, `FORCE_LANGGRAPH_AGENT`, `ENABLE_VISION_LLM`) are
  unchanged.
- `legacy/llm_module.py` — re-exports from `llm` + the live model state.
  `llm`, `is_vision_model`, `active_model_desc` and `llm_lock` are what the
  HTTP routers actually read and swap (`LLMModuleLike` in
  `adapters/http/_types` declares them), so they stay where `load_model`
  owns them. The two globals with no reader anywhere — `_chat_handler` and
  the eagerly-computed `_RAG_SYSTEM_PROMPT`, which built a system prompt at
  import time for nobody — are gone (ADR 0014).
- `legacy/rag_module.py` — re-exports from `rag` + the live index state
  (`knowledge_index`, `knowledge_chunks`, `bm25_index`, `embed_model_ref`,
  `_reranker`, guarded by `rag_lock`) that `initialize_rag` owns and the
  admin routes reset. Its `is_vision_model` had no reader and is gone.

The shims `legacy/{a2a_protocol,action_manifest,agent_module,checkpointing,lsp_client,mcp_server}.py`
were deleted in ADR 0012: each was a one-line re-export with no live importer.
`adapters/legacy_agent.py` was deleted in ADR 0013: it was a compatibility
bridge for a removed package that built a second `AgentService` and a second
`ToolRegistry` at import time and had zero production importers.

All production code imports from `ai_assistant.*` directly. The former
`LangGraphAgent` exports now live in `orchestration.compat`, and the Qt process
manager is `bootstrap.server_manager`. An import-linter contract and the AST
boundary tests keep `modules` from being resurrected.

`bootstrap/server_manager.py` is console-facing by contract: its `print(..., flush=True)`
lines *are* the transport the Qt client reads (it keys on the `[SUCCESS]`
sentinel), so they are not candidates for `logging`. Its child-process detach
flags are platform-guarded — `creationflags` on Windows, `start_new_session`
elsewhere — because a non-zero `creationflags` raises `ValueError` on POSIX
(ADR 0014).

The AST boundary test that forbids import-time environment reads used to skip
`legacy/` because of `config.py`. That exemption is **removed** (ADR 0014): the
shim now delegates to `settings`, so `legacy/` is scanned like every other
module and a reintroduced inline read fails CI.

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
  `settings` is the sole environment owner and `legacy/config.py` delegates to
  it; the Qt process manager's detach flags are platform-guarded; the
  supervisor's tool groups are public API.

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
- `tests/unit/test_plugin_reentry.py` — plugin loader idempotency and duplicate handling.
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

Coverage of `src/ai_assistant/*` is **100%** with branch coverage enabled
(7 189 statements, 1 904 branches, measured Oct 2026). That includes the
composition roots `bootstrap/server.py` and `bootstrap/server_manager.py`,
which previously pulled the total below the floor; their platform shims,
deployment-profile branches and process entry points are now covered through
extracted helpers and targeted tests rather than excluded. Run it with:

```bash
PYTHONPATH=src python -m coverage run -m unittest discover -s tests -p "test_*.py"
PYTHONPATH=src python -m coverage report
```

There are deliberately no flags: `pyproject.toml` owns the whole coverage
configuration (`branch = true`, `source`, the `*/legacy/*` omit and the
`fail_under = 100` floor), so a local run and CI measure the same thing.
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

Each script puts `src/` — not the project root — on `sys.path`. Pointing at
the root left `ai_assistant` unimportable and made the whole eval job fail
with `ModuleNotFoundError`; the wiring is now pinned by running both scripts
in CI (ADR 0014).
