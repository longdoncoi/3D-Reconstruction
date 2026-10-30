# ADR 0046 — Tách composition root: lifespan, entry point, proactor shim

- Status: Accepted
- Date: 2026-10-28
- Supersedes: none

## Bối cảnh

`bootstrap/server.py` là composition root cấp process (ADR 0020/0022/0042):
`compose()` dựng logging, model registry, container, services, app và router
đúng một lần. Nhưng cùng một file còn chứa ba trách nhiệm không liên quan:

1. **Lifespan FastAPI** — discovery A2A, prune task, mở/đóng MCP và hai store.
2. **Entry point** — `main()` dựng `uvicorn.Config/Server`, arm watcher
   graceful stop, `sys.exit(1)` khi warm-up lỗi.
3. **Windows proactor shim** — bọc `_call_connection_lost` để
   `ConnectionResetError` không thành traceback.

Hệ quả: file 743 dòng, ba lý do để sửa khác nhau, và review 2026-10-27 ghi
đây là design debt còn mở. Ràng buộc khi tách:

- Import `bootstrap.server` vẫn chỉ được kéo stdlib + `ai_assistant.settings`
  (ADR 0022) — không kéo FastAPI/uvicorn/asyncio internals.
- State composed là **một instance process-wide**; lifespan phải đọc đúng
  instance đó, không phải bản sao.
- Test hiện patch `server.logger`, `server.platform`, `server.mcp_server`,
  `server.memory_store`, `server.task_retention_days`,
  `server._discover_a2a_peers`, `server.sys.platform`,
  `server._silence_connection_reset`, `server.bootstrap_runtime`,
  `server._stop_server_when_requested`, `server.app`,
  `server.runtime_modules`; các điểm patch này phải tiếp tục có hiệu lực.
- Hai entry point được tài liệu hoá phải còn chạy:
  `python src/ai_assistant/bootstrap/server.py` (script, không có package
  context) và `python -m ai_assistant.bootstrap.server`.

## Quyết định

### 1. `bootstrap/lifecycle.py` sở hữu lifespan

`lifespan(app)` đọc state qua `bootstrap.server` **tại thời điểm gọi**:
`server.logger`, `server.platform`, `server.mcp_server`, `server.memory_store`,
`server.task_retention_days`, `server._discover_a2a_peers`,
`server._SERVER_START_TIME`, `server.agent_bind_host/port`. Nhờ vậy state vẫn
là instance duy nhất của composition root, và một test patch `server.platform`
vẫn điều khiển đúng lifespan thật.

`_discover_a2a_peers` ở lại `server.py`: nó là discovery của registry do
composition root sở hữu, và là điểm patch của test.

### 2. `bootstrap/cli.py` sở hữu entry point

`stop_server_when_requested()` (watcher) và `main()` chuyển sang đây.
`main()` đọc `server.compose`, `server.bootstrap_runtime`, `server.app`,
`server.runtime_modules`, `server._stop_server_when_requested` tại thời điểm
gọi. Nó import `ai_assistant.bootstrap.server` bằng **absolute import**, không
phải relative: khi chạy như script, `__package__` rỗng nên relative import
không dùng được; absolute import cũng bảo đảm graph composed nằm trong đúng
module mà uvicorn phục vụ, không phải bản `__main__` thứ hai.

`server.py` giữ tên lịch sử bằng assignment
`_stop_server_when_requested = stop_server_when_requested`, nên test patch
`server._stop_server_when_requested` vẫn hoạt động.

### 3. `bootstrap/platform_patch.py` sở hữu shim

`silence_connection_reset(transport_cls)` và
`patch_proactor_transport(*, platform, silence)` là code thuần stdlib.
`server._patch_proactor_transport()` còn đúng một dòng, truyền
`sys.platform` và `_silence_connection_reset` — hai điểm patch cũ vẫn nguyên
giá trị.

### 4. `server.py` chỉ còn composition + re-export

`server.py` giữ `compose()`/`_compose_once()`, state, `__getattr__` PEP 562,
`bootstrap_runtime()`, `_refresh_agent_routes()`,
`_register_optional_protocols()`, `_build_supervisor_delegate()` và
`_discover_a2a_peers()`. Nó re-export `main`, `_stop_server_when_requested`,
`lifespan`, `_silence_connection_reset` để entry point, `create_app` và test
giữ nguyên tên.

## Hệ quả

- `server.py` 743 → 640 dòng; mỗi module mới có một lý do sửa.
- Ba module mới đều nằm trong phép đo coverage (không có exclusion mới) và
  đạt 100 % statement + branch.
- Hai mutant của quyết định shutdown đi theo code: watcher sang `cli.py`,
  contained close sang `lifecycle.py`. `MIN_MUTANTS` giữ 226;
  `PINNED_DECISION_FILES` thêm `cli.py` và `lifecycle.py` để không file nào
  âm thầm mất mutant cuối.
- Import-time contract không đổi: `bootstrap.server` top-level chỉ còn
  stdlib + `ai_assistant.settings`; `lifecycle`/`cli`/`platform_patch` cũng
  chỉ stdlib ở top-level (uvicorn và proactor internals vẫn import trong hàm).
- Đo lại 2026-10-29 (sau follow-up consolidation-on-resume của ADR 0043):
  **2 443 test, 0 fail, 6 skip**; coverage **100.00 %** (10 337 statement,
  2 808 branch); mutation **242/242 killed**; mypy **151 file**, 0 issue;
  lint-imports **9/9**.

## Không thay đổi

- ADR 0020/0022 (compose lười, không side effect lúc import) và ADR 0042
  (một composition root cấp process) — chỉ đổi chỗ đặt code.
- Hành vi lifespan: thứ tự startup, containment của từng close, và thứ tự
  đóng memory store trước task store.
- Hành vi entry point: warm-up trước khi serve, exit 1 khi warm-up lỗi,
  re-raise `BaseException` của uvicorn sau khi log.
