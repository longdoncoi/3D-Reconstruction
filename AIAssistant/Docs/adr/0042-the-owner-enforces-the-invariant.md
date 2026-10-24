# ADR 0042 — Chủ sở hữu thực thi invariant, mặc định fail-closed và dừng êm

- Status: Accepted
- Date: 2026-10-24
- Supersedes: none

## Bối cảnh

Vòng review thứ mười đọc source thay vì tin bảng gate, và hỏi ba câu mà một
bảng gate xanh không trả lời được:

1. **Ai thực thi invariant?** Drain contract của `model_reload()` có đúng,
   nhưng `load_model()` / `release()` vẫn public. Một caller mới (route, test,
   bootstrap) gọi thẳng primitive là bypass được cửa sổ drain — chính xác lỗi
   mà ADR 0041 vừa đóng ở tầng route.
2. **Mặc định khi thiếu cấu hình là gì?** Ba chỗ vẫn "thoáng" (fail-open):
   `scopes_for_capability(None)` trả scope write/execute; tool thiếu `policy`
   thành read-only; thiếu `Config/agent_tools.json` trả registry rỗng.
3. **Tắt server có chạy cleanup không?** `/admin/shutdown` gọi
   `os._exit(0)`: bỏ qua toàn bộ lifespan `finally` (đóng task store,
   checkpoint, flush), đúng thứ mà graceful shutdown tồn tại để làm.

Cùng lúc, adapter HTTP còn chạm private member của RAG (`_reranker`,
`_release_ml_memory`), hai protocol mô tả cùng một lease theo hai kiểu, và
docstring composition root nói "không còn module-level mutable state" trong
khi `_COMPOSED` là singleton cấp process.

## Quyết định

### 1. Chủ sở hữu thực thi invariant, không phải người gọi

- `LlmModule._require_reload_window()` là cửa duy nhất: `release()` và
  `load_model()` raise `RuntimeError` nếu không nằm trong `model_reload()`.
- `bootstrap_runtime()` bọc lần load đầu tiên trong `model_reload()`; test
  bootstrap cũng vậy. Không còn đường tắt "lúc startup chưa có lease".
- Đây là guard của **owner** (ADR 0032/0041), nên nó sống cùng lớp sở hữu
  thay vì phụ thuộc từng route nhớ gọi đúng.

### 2. Thiếu cấu hình thì từ chối, không đoán

- `scopes_for_capability(capability, None)` trả `READ_ONLY_SCOPES`. Bỏ hằng
  `DEFAULT_TRUSTED_SCOPES`: một host quên inject map không còn chạy với
  `project.write` / `project.execute`.
- `tool_spec_from()` raise `ValueError` khi entry không có `policy`; policy
  lạ đã fail từ trước. Không có "mặc định read-only" cho tool chưa được tác
  giả.
- `create_tool_registry()` raise `ConfigurationError` khi thiếu
  `Config/agent_tools.json`. Server không được tự nhận "ready" với catalogue
  rỗng rồi fail ở mọi request.

### 3. Bề mặt RAG công khai

- `RagModule.reranker_active` là property chỉ đọc; `release_ml_memory()` là
  operation có tên. Adapter `RAGModuleLike` và hai route (`/health`,
  `/v1/chat/...`) chỉ dùng hai thành viên này, không chạm `_reranker` /
  `_release_ml_memory`.
- Chuyển state transition vào module giữ nguyên ADR 0032: module sở hữu trạng
  thái index, adapter chỉ hỏi.

### 4. Dừng êm thay vì `os._exit`

- `RuntimeModules.shutdown_requested` (một `threading.Event`) và
  `request_shutdown()` là tín hiệu duy nhất.
- `/admin/shutdown` trả lời HTTP trước, rồi (sau 0,3 s, trong thread nền) gọi
  callback `request_shutdown` được inject — không còn `os` trong router.
- `_stop_server_when_requested(server, event)` đặt
  `uvicorn.Server.should_exit = True`; `main()` dựng `uvicorn.Server` +
  `uvicorn.Config`, chạy watcher và gọi `server.run()`. Lifespan `finally`
  chạy bình thường; không có đường nào bỏ qua cleanup.

### 5. Một kiểu cho một hợp đồng

- `LLMRuntime.model_lease()` trả `AbstractContextManager[LLMBackend | None]`;
  `LLMModuleLike` **kế thừa** `LLMRuntime` thay vì khai báo lại `llm` /
  `llm_lock` / lease bằng `Any`.
- `ToolSpec.plugin_id` mặc định `BUILTIN_PLUGIN_ID`, không còn chuỗi
  `"builtin"` thứ hai.
- Docstring `classification_at_least` mô tả đúng chiều so sánh (clearance của
  principal ≥ trần dữ liệu của tool).

### 6. Phạm vi composition root

- `bootstrap/server.py` là composition root **cấp process** (một `_COMPOSED`,
  một graph). Hai server trong cùng process là **không hỗ trợ**; embedding
  host phải compose ở process riêng. Docstring nói đúng điều đó thay vì tuyên
  bố "không còn state cấp module".

## Hệ quả

- Không thể vô tình giải phóng model khi một run đang giữ lease: primitive
  không còn gọi được ngoài cửa sổ drain, kể cả từ code mới.
- Một deployment thiếu scope map / policy / catalogue **dừng với lỗi rõ ràng**
  thay vì chạy với quyền rộng hơn hoặc catalogue rỗng.
- `/admin/shutdown` giờ là graceful stop thật; test pin cả ba mắt xích
  (route → event → `should_exit`).
- Adapter không còn phụ thuộc private member; đổi tên field nội bộ của
  `RagModule` không làm vỡ HTTP layer.
- Không đổi wire/disk format, không nới approval/scope/transport/sandbox.

## Ghim bằng test và mutant

- `tests/unit/test_bootstrap_coverage_gaps.py`: swap ngoài cửa sổ drain raise;
  `release_ml_memory()` gọi đúng helper; `reranker_active` phản ánh field;
  `request_shutdown()` set event.
- `tests/integration/test_composition_root_bootstrap.py`: startup load chạy
  trong `model_reload()` (fake loader gọi `_require_reload_window()` như bản
  thật).
- `tests/unit/test_final_edges_a.py`: host không inject map → principal chỉ có
  `project.read`.
- `tests/unit/test_gateway_bootstrap_guards.py`: entry thiếu policy bị từ chối.
- `tests/unit/test_final_edges_b.py`: thiếu catalogue → `ConfigurationError`.
- `tests/integration/test_main_entry_points.py`: `Server`/`Config` được dựng,
  `run()` được gọi, watcher đặt `should_exit`; runtime fail thì exit 1.
- `tests/unit/test_http_admin_routes.py`: route gọi `request_shutdown`, không
  còn `os._exit`.
- Mười mutant mới: cửa sổ reload (guard + startup), scope map, missing policy,
  missing catalogue, event, watcher, route shutdown, `reranker_active`,
  `/health` reranker. `MIN_MUTANTS` 150 → **160**.

Đo lại ngày 2026-10-24: xem `Docs/SCORECARD.md` — mục "Independent review
follow-up (2026-10-24, ADR 0042)".

## Không thay đổi

- ADR 0032 vẫn sở hữu drain contract; ADR này chỉ đặt nó vào đúng chủ sở hữu.
- ADR 0041 vẫn sở hữu single model owner và admission trước thread.
- Không đổi endpoint, schema, hay định dạng task/session trên đĩa.
- 72 `except Exception` vẫn là boundary barrier (HTTP, tool, optional SDK,
  worker): chúng dịch lỗi thành kết quả có cấu trúc hoặc log, không nuốt một
  quyết định an toàn. Thu hẹp từng site là thay đổi hành vi riêng, không phải
  điều kiện của ADR này.
