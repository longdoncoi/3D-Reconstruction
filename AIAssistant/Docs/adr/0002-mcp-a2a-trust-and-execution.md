# ADR 0002: MCP/A2A trust and execution

## Decision

`ToolExecutionService` is the only tool execution path. MCP adapts protocol
requests into that service; internal agents never issue an HTTP request to the
local MCP endpoint. Tool specifications declare schema, scope, side effect,
approval, classification, timeout, and plugin ownership.

A2A is task-oriented. Tasks persist in SQLite for desktop deployments and
publish card, submit, get, cancel, and event-stream endpoints. A remote call
requires an endpoint allowlist and explicit policy. There is no automatic
remote-to-local fallback; callers receive an explicit routing outcome.

## Consequences

- Approval-gated tools only execute after approval through the same service.
- Remote A2A must be enabled and configured by an operator.
- Restricted and regulated payloads cannot be sent through the remote A2A
  adapter.
