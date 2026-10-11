# ADR 0007 — Static type safety for the LLM runtime port

- Status: Accepted
- Date: 2026-10-10
- Supersedes: none

## Context

`ports.LLMRuntime` previously declared `llm: Any` and `llm_lock: Any`, which
meant mypy could not check the agent layer against the module
(`ai_assistant.legacy.llm_module`) that actually backs it. Every time someone
renamed a backend property, the mismatch surfaced only at runtime on the Qt
desktop, not in CI (Sprint 2 review).

## Decision

`ports.LLMRuntime` uses concrete types:

```python
class LLMRuntime(Protocol):
    llm: LLMBackend | None
    llm_lock: LockLike
```

- `LLMBackend` — the inference port (`is_vision_supported`,
  `model_description`, `generate`) — is owned by `ai_assistant.ports` (it is a
  port, and contract 1 forbids `ports` from importing `ai_assistant.llm`).
  `ai_assistant.llm.contracts` re-exports it for the `llm` adapters.
- `LockLike` is a structural protocol satisfied by both `threading.Lock` and
  `threading.RLock`; the legacy runtime owns an `RLock`, and the concrete
  `threading.Lock` type (or `contextlib.AbstractContextManager`) refuses it.
- `legacy.llm_module` annotates its module-level `llm_lock: LockLike` so a
  module (not an instance) satisfies the port protocol, which mypy otherwise
  rejects on module-vs-protocol attribute comparison.

## Consequences

- `mypy src/ai_assistant` now checks all 121+ modules cleanly, including the
  legacy shims that are reachable through the compatibility adapter.
- Renaming a backend property or lock field breaks CI instead of the desktop.
- The `Any` declarations that let layer boundaries rot are gone.