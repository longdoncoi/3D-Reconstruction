# ADR 0013: RunDeps bundle, reason-node decomposition, dead-code removal

Date: 2026-10-12

Status: Accepted

## Context

The 3.0 architecture review (Oct 2026) scored the platform 8/10. Three
concrete structural gaps blocked the top mark:

1. **`run_langgraph_agent` accepted 46 keyword-only parameters.** Twenty of
   them were callables that never vary between invocations (the supervisor
   policy hooks, the observability functions, the LLM completion adapters).
   `AgentRunContext` (ADR 0012) had deduplicated the *call sites*, but the
   runner signature itself was still a dependency-injection container spelled
   out as an argument list.

2. **`orchestration/nodes/reason.py:reason_node` was a single 353-line
   function** with 19 top-level branches, one nested helper, and a magic-string
   protocol: the runner's `complete` callback returned the literal
   `"Context quá dài, dừng Agent."` and the reason node matched it with a
   three-prefix `casefold().startswith(...)` tuple defined independently.

3. **Dead and duplicated code.** `adapters/a2a_protocol.py` carried an unused
   `_text_message` / `_safe_value` / `task_payload` copy left behind by the
   ADR 0005 split — it hardcoded `"state": "submitted"`, used a placeholder
   timestamp, and applied a second redaction key set that bypassed the
   domain-owned policy (ADR 0010). `adapters/legacy_agent.py` built a second
   `AgentService` and a second `ToolRegistry` at import time and had zero
   production importers. `tools/lsp_client.py` (317 lines) had zero production
   importers — it was registered nowhere, so the two LSP tools were dead in
   the live catalog.

## Decision

### 1. `RunDeps` (agents/deps.py)

`run_langgraph_agent` now takes one frozen `RunDeps` bundle plus the
per-invocation arguments (prompt, task, resume state, approval grant, event
sink). The runner signature drops from 46 parameters to 21, and the 25
shared dependencies live in one dataclass with an explicit field order
(required wiring first, optional transport/framework/token-budget last).
`AgentRunContext.run` builds the bundle once; the three agent sub-services
see no change. `RunDeps` is exported from `ai_assistant.agents`.

`validate_tool_call_fn` and `generate_action_id_fn` deliberately stay
per-invocation: the resume paths swap them for the named no-op helpers.

### 2. Reason-node decomposition

`reason_node` is now a 78-line dispatch sequence. Each outcome the model can
produce is handled by one focused private function that receives the shared
`_Turn` record and returns either the partial state update or `None` to fall
through:

| Handler | Outcome |
|---|---|
| `_compact_context` | context budget exhausted → summarise and retry |
| `_handle_step_answer` | inline answer for a general-knowledge plan step (or a hallucinated one) |
| `_handle_final_answer` | stop request, blocked when plan/coding evidence is incomplete |
| `_handle_validation_error` | parsed envelope failed schema validation |
| `_handle_forbidden_transfer` | direct `transfer_to_toolapp_agent` call |
| `_guard_step_action_mismatch` | right tool, wrong desktop action for the plan step |
| `_guard_duplicate_failed_call` | tool call identical to one that already failed review |
| `_record_thinking` | surface pre-fence text as a `thinking` step |
| `_dispatch_tool_call` | delegation, `tool_call` step, approval gate |

`stable_fingerprint` and `_failed_call_fingerprints` are module-level
helpers (the fingerprint logic was a nested closure).

### 3. Typed context-exhausted sentinel

`orchestration/state.py` now owns `CONTEXT_EXHAUSTED_MESSAGE`,
`CONTEXT_EXHAUSTED_PREFIXES` and `is_context_exhausted()`. The runner's
`complete` callback returns the constant; the reason node calls the predicate.
The literal no longer exists in two places.

### 4. Single `runtime_unconfigured` wire shape

`tools/tool_contract.py` owns `RUNTIME_UNCONFIGURED_RESULT`. The gateway
adapter (`bootstrap/runtime.py`), the MCP adapter (`adapters/mcp.py`), the
runner and the approval service all import it instead of re-declaring the
same three-key dict inline.

### 5. Dead-code removal

- `adapters/a2a_protocol.py`: deleted the unused `_text_message`,
  `_safe_value` and `task_payload` copies. `adapters/a2a_payloads.py` is the
  single payload/redaction owner (ADR 0010).
- `adapters/legacy_agent.py`: deleted. It was a compatibility bridge for a
  package that no longer existed, built a second `AgentService` and a second
  `ToolRegistry` at import time, and had zero production importers. The two
  tests that drove it now exercise the canonical APIs
  (`agents.completion.parse_tool_call`, `agents.service.AgentService`).
  A boundary test pins the deletion the same way ADR 0012 pins the removed
  `modules` package.

### 6. LSP tools activated

`go_to_definition` and `find_references` are now registered in
`tools/factory.py:builtin_handlers()` and advertised in
`Config/agent_tools.json` (24 tools). Their policy contract is `read_only`
with a 30-second timeout (a cold `clangd`/`pylsp` start needs headroom).

> **Đính chính 2026-10-21 (ADR 0035).** Mục này viết *"activated"*, và đó là
> **không đúng** với trạng thái thực tế. Hai công cụ này **được đăng ký và
> được quảng bá cho mô hình**, nhưng chúng **không có mặt trong bảng
> `ALLOWED_TOOLS`** của supervisor (`orchestration/supervisor.py`), nên
> `authorise()` **từ chối mọi lời gọi** — với mọi specialist. Đo được:
>
> ```
> go_to_definition  -> refused (not in RESEARCH/VERIFICATION/CODE/WORKFLOW rows)
> find_references   -> refused (not in any row)
> ```
>
> Hệ quả: hai công cụ được mô hình nhìn thấy nhưng không bao giờ chạy, và
> **sự từ chối đó chính là thứ che lỗi `shutil.which` của `lsp_client`** — lỗi
> cho phép một binary do agent cài trong thư mục làm việc được *thực thi*.
> ADR 0035 đã sửa lỗi đó (`lsp_client` nay dùng resolver chung,
> `ai_assistant.exec_policy`), nên đường đã an toàn — nhưng **việc bật hai
> công cụ này là một quyết định riêng**, chưa được đưa ra ở đây.
>
> Trạng thái đó nay được **khai báo** trong mã
> (`supervisor.REGISTERED_NOT_ENABLED_TOOLS`) và có một guard chống trôi: mỗi
> công cụ mà mô hình nhìn thấy phải **hoặc** có một hàng trong
> `ALLOWED_TOOLS`, **hoặc** nằm trong danh sách khai báo ấy. Bật một công cụ
> mà không cập nhật khai báo sẽ làm CI đỏ — nên lần đọc ADR này sau này không
> thể tin nhầm nữa.
>
> Bản sửa đúng của mục này là: chúng đã được **đăng ký**, chưa được **kích
> hoạt**. Khi bật, bảng `ALLOWED_TOOLS` phải có hàng cho chúng (và chúng đi
> qua cùng một policy gateway như mọi tool khác).

## Consequences

- Adding one injected dependency to an agent run now means adding one field
  to `RunDeps`; no call site changes.
- The reason node's decision tree is reviewable and testable per outcome; a
  new guard is one small function added to the dispatch list.
- The context-budget protocol and the `runtime_unconfigured` wire shape each
  have exactly one definition.
- `adapters/a2a_protocol.py` no longer carries a second, broken payload
  builder; the A2A 0.3 surface serialises through `a2a_payloads` only.
- The code agent **can** jump to definitions and find references through the
  same policy gateway as every other tool — **khi** chúng được thêm vào bảng
  `ALLOWED_TOOLS` (xem đính chính ở trên: hiện chưa, và mọi lời gọi bị từ
  chối).
- Verified: `ruff check .` clean, `mypy src/ai_assistant` clean (124 files),
  889 unit/integration/contract tests pass.
