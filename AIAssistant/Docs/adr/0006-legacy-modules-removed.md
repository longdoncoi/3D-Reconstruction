# ADR 0006 — Legacy `modules/` package removed

- Status: Accepted
- Date: 2026-10-09
- Supersedes: none

## Context

The platform originally shipped a flat `modules/` package that mixed
infrastructure, state, and UI-adjacent logic and was imported from the agent
runs, the HTTP adapters and the composition root. The import-linter contracts
and the AST boundary guards made `modules/` the one package that everything was
forbidden to touch — but the package itself still existed and could be
resurrected by a single careless import.

## Decision

The `modules/` package is **removed**. Backward compatibility lives in
`src/ai_assistant/legacy/`, a thin shim package whose modules only re-export
from the new structure:

- `legacy/config.py` — re-exports from `ai_assistant.settings` + `ai_assistant.config`.
- `legacy/llm_module.py` — re-exports from `ai_assistant.llm` + legacy global state.
- `legacy/rag_module.py` — re-exports from `ai_assistant.rag` + legacy global state.
- `legacy/mcp_server.py` — re-exports from `ai_assistant.adapters.mcp`.
- `legacy/agent_module.py` — composition-root compatibility bridge.
- `legacy/a2a_protocol.py`, `legacy/checkpointing.py`, `legacy/lsp_client.py` —
  shims kept for the compatibility adapter and legacy-format tests.

All production code imports from `ai_assistant.*` directly. An import-linter
contract (`The legacy modules package stays removed`) and the AST boundary
tests keep `modules` from being resurrected. Dead shims whose names were not
referenced anywhere are deleted outright (Sprint 3 cleanup), not preserved.

## Consequences

- The forbidden import direction is structurally impossible to reintroduce.
- Shims still sell an escape hatch: new code is guided toward `ai_assistant.*`
  by code review and the boundary tests.
- The shim package shrinks every time a legacy caller migrates.