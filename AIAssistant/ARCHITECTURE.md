# AI Agent Platform architecture

The server is a **policy-first Clean Architecture platform** with enforced
layer boundaries, a single execution gateway, and protocol isolation. The
compatibility entrypoint is `StartChatbotServer.py`; the production package is
`src/ai_assistant`.

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
- There is exactly one `ToolSpec` type — the policy record in `domain/tools.py`.
  The executable catalog entry is `tools/definition.py:ToolDefinition`; the old
  `ai_assistant.core` package is gone.
- Capability scopes are enforced across the shared engine: the A2A principal is
  bound with `bind_principal` for the call and read by `PlatformToolGateway`.
- Pending approvals are rehydrated from the file-backed store when
  `AgentService` starts, so A2A resume survives a process restart.
- MCP is an external adapter; agent execution does not call `/mcp` locally.
- A2A owns durable task lifecycle. Its only remote transport is
  `adapters/a2a_client.TrustedA2AClient` (allowlist-only, classification-gated);
  `A2ARouter` is a routing facade over it and reports `remote_selected`,
  `rejected_*`, or `degraded_dependency` outcomes.
- LangGraph remains isolated behind its adapter so it stays replaceable.
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

## A2A adapter decomposition (ADR 0005)

The former monolithic `adapters/a2a.py` (310 lines) has been decomposed into:

- `adapters/a2a_models.py` — Pydantic request/response schemas (A2A 0.3 compliant).
- `adapters/a2a_payloads.py` — payload builders, state mapping, sensitive data redaction.
- `adapters/a2a_router.py` — FastAPI router with all standard and legacy endpoints.
- `adapters/a2a.py` — backward-compatible re-exports.

## Legacy migration (ADR 0006)

The legacy `modules/` package has been **removed**. Implementations live in
`src/ai_assistant/legacy/`:

- `legacy/config.py` — re-exports from `settings` + legacy constants.
- `legacy/llm_module.py` — re-exports from `llm` + legacy global state.
- `legacy/rag_module.py` — re-exports from `rag` + legacy global state.
- `legacy/mcp_server.py` — re-exports `McpToolServer` from `adapters/mcp`
  (import-compatible shim; the composition root builds its own instance).
- `legacy/a2a_protocol.py` — re-exports from `adapters/a2a_protocol`.
- `legacy/agent_module.py` — composition-root compatibility bridge.

All production code imports from `ai_assistant.*` directly. Two root-level
compatibility shims remain only for external consumers: `LangGraphAgent.py`
re-exports `orchestration.graph` and `server_manager.py` is the process manager
the Qt desktop app launches. An import-linter contract and the AST boundary
tests keep `modules` from being resurrected.

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

## Verification

```bash
cd AIAssistant
ruff check .
python -m compileall -q StartChatbotServer.py evals src
python -m unittest discover -s tests -p "test_*.py"
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
- `tests/unit/test_coverage_boost*.py`, `tests/unit/test_llm_backend.py` —
  targeted coverage for the completion decoders, the A2A client routing
  policy, the task worker lifecycle, model loading, the llama.cpp backend and
  the FastAPI composition boundary.
- `tests/unit/test_index.py`, `tests/unit/test_loaders.py` — image-aware
  embedding, the IVF corpus path, scan edge cases, encoding/docx/pdf/email
  loader fallbacks, hierarchical markdown sections and the sliding-window
  overlap guard (lifts `rag/index.py` to 100% and `rag/loaders.py` to 99%).

Statement coverage of `src/ai_assistant/*` is **90%** locally (measured Oct
2026) and **~89.5%** in the offline CI environment, where the optional runtime
packages (`mcp`, PIL, …) are absent. The RAG indexing and loader modules are
covered to 99–100%. Run it with:

```bash
PYTHONPATH=src python -m coverage run -m unittest discover -s tests -p "test_*.py"
PYTHONPATH=src python -m coverage report --include="src/ai_assistant/*" --omit="*/legacy/*"
```

CI gates statement coverage at 88% (the offline environment lacks the optional
`mcp` package, which costs roughly half a point) and runs the lint job and the
full test suite in `.github/workflows/Build-Python-ci.yml`.
