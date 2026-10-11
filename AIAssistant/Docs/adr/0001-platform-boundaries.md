# ADR 0001: Modular AI Agent Platform boundaries

## Decision

`ai_assistant.domain` and `ai_assistant.application` contain no FastAPI,
LangGraph, MCP, A2A, model-provider, vector-store, filesystem, or environment
imports. Technologies implement ports in `ai_assistant.adapters`; bootstrap is
the only composition root. Existing modules remain compatibility adapters until
each concern is migrated and its compatibility endpoint is retired.

## Consequences

- LangGraph is replaceable orchestration, not the policy or task-state owner.
- The legacy `/v1/chat/completions` and `/v1/agent/*` contracts remain stable.
- No new feature may read environment variables or register itself at import
  time outside `settings` and `bootstrap`.
