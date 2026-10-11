# ADR 0003 — One execution core, one orchestration engine

- Status: Accepted
- Date: 2026-10-09
- Supersedes: none

## Context

The platform historically grew **two agent loops**:

1. `ai_assistant.agents.runner.run_langgraph_agent` (backed by
   `LangGraphAgent.LocalAgentGraph`) — the graph planner/reasoner/reflector used
   by the Qt desktop `/v1/agent/*` transport.
2. `ai_assistant.application.agent_runs.AgentRunService` — a small,
   framework-independent loop used by the A2A transport.

Reviewers reasonably flagged this as "two engines". Policy, validation,
approvals, auditing and side effects were duplicated between them and could
drift, and the two loops made different planning decisions for the same task.

## Decision

There is exactly **one execution core**: `ToolExecutionService` (application
layer), reached only through the injected gateway
`ai_assistant.bootstrap.runtime.PlatformToolGateway`.

- `ToolExecutionService` is the *only* component allowed to invoke a tool
  executor. It owns schema validation, scope checks, data-classification policy,
  approval gating, event emission and result shaping.
- Every transport and every orchestration strategy must execute through the
  injected `ToolGateway` port:
  - Qt desktop agent: `agents/runner.py` receives the gateway from
    `AgentService`.
  - A2A: delegates to the canonical engine through `AgentOrchestrator`, which
    reaches tools through the same gateway.
  - MCP: `adapters/mcp.py` is configured with the gateway at mount time.
- The gateway is an object built by the DI container, not a module-level service
  locator: `agents` never imports `bootstrap`, and `bootstrap.runtime` holds no
  mutable state.
- `ToolRegistry` is a **catalog only**. It no longer exposes an `execute` method,
  and no code may call `ToolSpec.handler` directly.

### One tool model

There is exactly one type named `ToolSpec`: the framework-free policy record in
`domain/tools.py` consumed by `ToolExecutionService`. The agent-facing catalog
entry (JSON-schema parameters, handler, prompt/grammar serialization) is
`tools/definition.py:ToolDefinition`. The former `ai_assistant.core` package
that held a second, drifting `ToolSpec` was removed.

### One canonical decision loop

The LangGraph runner (`agents/runner.py` + `orchestration/`) is the single
production decision loop. The A2A lifecycle reuses it through the
`AgentOrchestrator` port (`ports.py`); the adapter
(`adapters/orchestration/langgraph.py`) translates between the durable A2A task
contract and the Qt engine contract. The application layer therefore depends on
a protocol, never on LangGraph.

The former bounded deterministic loop is now itself an **adapter**
(`adapters/orchestration/deterministic.py:DeterministicAgentOrchestrator`).
`AgentRunService` holds no decision loop; it resolves scopes and delegates to the
injected orchestrator, or fails explicitly when none is configured.

| Path | Decision loop | Module |
| --- | --- | --- |
| Qt desktop `/v1/agent/*` | LangGraph graph (plan/reason/reflect) | `agents/runner.py` + `orchestration/` |
| A2A `/tasks/*` | same LangGraph graph via `AgentOrchestrator` | `adapters/orchestration/langgraph.py` |
| A2A without injected engine | deterministic adapter (tests/embeds) | `adapters/orchestration/deterministic.py` |

### One remote A2A transport

`adapters/a2a_protocol.A2ARouter` is a routing facade only: it selects an
eligible remote agent and delegates the HTTP call to
`adapters/a2a_client.TrustedA2AClient`, which enforces the deployment allowlist
and refuses to export sensitive classifications. No second HTTP path exists.

Capability least-privilege is preserved across the shared engine: the A2A
principal resolved by `AgentRunService` is bound with `bind_principal`
(`domain/security.py`) for the whole call, and `PlatformToolGateway` executes
tools as the bound principal instead of the legacy local identity.

## Consequences

- Policy can no longer drift between transports; a change in
  `ToolExecutionService` applies everywhere.
- Both transports now make the same planning decisions; A2A gains the
  plan/reason/reflect, verification and specialist behaviour of the desktop
  agent.
- `tests/integration/test_architecture_boundaries.py` enforces the single
  gateway (`ToolRegistry.execute` gone, no `spec.handler(...)`), a single
  `ToolSpec`, the removal of `ai_assistant.core`, that `agents` never imports
  `bootstrap`, that no global tool-service locator remains, and that A2A reuses
  the canonical engine through the port without embedding a loop in
  `application`.
- A2A requires the LangGraph engine at runtime on the desktop path. When it is
  unavailable the task fails explicitly (`ServiceUnavailableError`) instead of
  silently running a second, weaker loop.
- Cross-restart resume of an A2A approval works because the file-backed pending
  store is rehydrated when `AgentService` is constructed at startup; an approval
  whose action has expired still fails explicitly.

