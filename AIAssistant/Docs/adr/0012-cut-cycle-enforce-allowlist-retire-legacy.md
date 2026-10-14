# ADR 0012 — Cut the supervisor/A2A cycle, enforce the A2A allowlist, retire dead legacy shims

- Status: Accepted
- Date: 2026-10-11
- Supersedes: none
- Related: 0001 (platform boundaries), 0002 (MCP/A2A trust), 0006 (legacy modules removed)

## Context

A review of `AIAssistant` scored the architecture ~8/10 and named the debts
blocking a clean 10/10:

1. **Import cycle** — `orchestration/supervisor.py` imported
   `adapters.a2a_protocol` inside a module-scope `try/except`, while
   `a2a_protocol._default_capabilities()` imported `Specialist` from the
   supervisor inside a function body. The cycle was import-order dependent:
   importing the supervisor first made the `try/except` swallow an
   `ImportError` and silently disable A2A (`delegate()` then never consulted the
   remote registry). A dedicated regression test (`test_a2a_import_order.py`)
   existed only to keep that fragile ordering alive.
2. **Vacuous A2A allowlist** — `A2ARouter._dispatch` built
   `TrustedA2AClient(trusted_endpoints=frozenset({agent.url}))`, so the client
   always trusted exactly the endpoint it was about to call and the
   `submit()` policy check never failed. The operator-facing
   `[a2a] trusted_endpoints` (parsed into `ArchitectureSettings`) and the
   `A2A_REMOTE_AGENTS` env var were two disconnected signals, neither of which
   reached the enforcement point — contrary to ADR 0002.
3. **Unisolated graph framework** — `ARCHITECTURE.md` claims LangGraph is
   "isolated behind its adapter", but no test enforced it.
4. **Dead legacy shims** — six `legacy/*` modules were pure re-exports with no
   live importer.
5. **Scattered configuration** — `CHARS_PER_TOKEN`/`LLM_N_CTX` were defined
   twice; `127.0.0.1:8080` was hard-coded in five places; `server_manager.py`
   duplicated the bind address and had no tests.

## Decision

### 1. Move `Specialist` into the domain; invert the A2A lookup

`Specialist` now lives in `ai_assistant.domain.specialists` (framework-free),
so every layer may depend on it inward. `orchestration/supervisor.py` imports
it from the domain and **never imports any adapter** — enforced by
`test_orchestration_does_not_import_any_adapter`.

The supervisor's remote lookup is dependency inversion: it exposes
`set_remote_specialist_lookup(resolver)` and a private
`_resolve_remote_endpoint` that fails closed (returns `None`, logging a
warning) when the resolver is missing or raises. The composition root
(`bootstrap/server.py`) installs
`adapters.a2a_protocol.resolve_remote_specialist` only when
`platform.settings.enable_a2a` is true, and clears it otherwise. The previous
`try/except ImportError` and `_A2A_IMPORT_OK` flag are gone.

`adapters/http/chat_routes.py` imports `Specialist` from the domain at module
level (no more lazy import).

### 2. Enforce one deployment A2A allowlist

`settings.a2a_trusted_endpoints()` merges the two operator signals into a
single normalised set: `[a2a] trusted_endpoints` from the parsed
`ArchitectureSettings` (via a new cached `settings.load_settings()`) plus the
`A2A_REMOTE_AGENTS` env override. It is empty until an operator configures one
— nothing is trusted by default.

`A2ARouter._dispatch` now builds
`TrustedA2AClient(trusted_endpoints=a2a_trusted_endpoints())`, so the
`submit()` policy check is real: any endpoint outside the deployment allowlist
is rejected with `routing=rejected_policy`.

### 3. Pin the framework-isolation boundary with AST tests

Two boundary tests in `tests/integration/test_architecture_boundaries.py`
encode the isolation promises structurally:

- `test_langgraph_framework_stays_isolated` — only
  `orchestration/graph.py` (engine) and
  `adapters/persistence/checkpointing.py` (checkpointer adapter) may import
  `langgraph`/`langchain`/`langchain_core`.
- `test_langsmith_tracing_stays_isolated` — only
  `observability/langsmith_integration.py` may import `langsmith`.

(import-linter cannot express "only these files may import X", so the AST
tests are the authoritative guard.)

### 4. Retire the six dead legacy shims

Removed `legacy/{a2a_protocol,action_manifest,agent_module,checkpointing,lsp_client,mcp_server}.py`
— each was a one-line re-export with no live importer. The three live shims
(`config`, `llm_module`, `rag_module`) remain for the desktop surface.

### 5. Single-source the runtime constants

- `legacy/config.py` re-exports `CHARS_PER_TOKEN`/`LLM_N_CTX` from
  `llm/defaults.py` instead of redefining them.
- `settings.py` owns `DEFAULT_HOST`, `DEFAULT_PORT` and `DEFAULT_BASE_URL`;
  the A2A card defaults, the transport policy's implicit loopback origins and
  `server_manager.py` all derive from them.

### 6. Collapse the three agent services' run wiring into `AgentRunContext`

`AgentService`, `ApprovalService` and `UIActionService` each spelled out the
same ~26 keyword arguments when calling `run_langgraph_agent` — the injected
LLM runtime, tool registry/gateway, pending store, and every
supervisor/observability/LLM callable. That was ~78 duplicated lines across
three call sites, and adding one dependency meant editing all three.

`agents/context.py` now owns that block once. `AgentRunContext` is a frozen,
immutable record of the wiring a run needs; each service builds one from its
own injected dependencies and calls `AgentRunContext.run(...)` with only the
arguments that actually vary per invocation (prompt, task, resume state,
event sink). The graph engine class and `Specialist` are supplied by the
context, so no service repeats them.

The first-pass path keeps the real `validate_tool_call` and
`generate_action_id`. *(Update, finding 27: the approval and desktop-ack resume
paths no longer swap in a `RESUME_VALIDATE_TOOL_CALL` no-op — a resumed run may
emit **new** tool calls, so every call is validated by the real validator. Only
the `RESUME_ACTION_ID` no-op remains, since a resumed continuation re-uses the
original action id.)* The A2A router stays on the
main execute path only — the resume paths run with `a2a_router=None`, matching
the pre-refactor behaviour.

`LANGGRAPH_AVAILABLE` (a module-level import flag) is replaced by
`self._ctx.local_agent_graph is not None`, which is equivalent but computed
from the injected graph class, so tests can substitute it.

The public constructors and attributes of all three services are unchanged;
the sub-services expose their shared state as read-only properties over the
context, and the facade still passes one shared pending store to both
sub-services.

### 7. Raise the CI coverage floor to 90

`.github/workflows/Build-Python-ci.yml` gates the offline suite with
`coverage report --fail-under`. The floor was 88, set when the measured
baseline was ~89.5%.

The floor was raised to 90 only after measuring what CI actually sees: this
job installs `requirements-test.txt`, which omits the optional `mcp`
dependency, so `adapters/mcp.py` cannot be imported there and is counted at
0%. With `mcp` blocked at the import hook, the suite still passes (the two
mcp-dependent tests skip cleanly) and measures ~89.97% — *below* 90. Setting
the floor to 90 on that baseline would have turned CI red immediately, so the
gap was closed with tests first:

- `tests/unit/test_agent_resume_services.py` — the HITL approve/reject
  lifecycle, single-use grant issuance, multi-step desktop action chaining,
  and the LangGraph resume hand-off for both sub-services.
- `tests/unit/test_http_types_protocols.py` — the HTTP adapter's
  injected-dependency protocols (ADR 0001), previously never exercised.
- `tests/unit/test_agent_run_context.py` — pins the shared wiring (see §6).

Measured CI baseline is now ~91.6%, leaving ~1.5pt of headroom above the
floor. The workflow's `workflow_dispatch`, path filters, job timeouts and
`cancel-in-progress: false` are unchanged.

## Consequences

- The supervisor↔adapter import cycle is gone; A2A availability no longer
  depends on import order, and the fragility-regression test was replaced by a
  stronger "orchestration imports no adapter" guard.
- Remote A2A is fail-closed: with no configured allowlist no remote dispatch
  can occur, satisfying ADR 0002's "endpoint allowlist and explicit policy".
- Swapping LangGraph (or LangSmith) is now verified by CI, not just claimed in
  documentation.
- ~90 lines of dead compatibility code removed; the remaining `legacy/`
  package is three modules with a documented removal path.
- `server_manager.py` is covered by unit tests and shares the canonical bind
  address.
- The agent-run dependency block exists in exactly one place, so a new
  injected dependency is a one-line change in `context.py` instead of three.
  `tests/unit/test_agent_run_context.py` pins the wiring each service hands
  the runner, so the consolidation cannot silently drop a dependency.
