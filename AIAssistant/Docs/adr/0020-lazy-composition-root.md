# ADR 0020 — Composition root lazy: import chỉ define, `compose()` mới build

- **`bootstrap/server.py` không còn side effect lúc import.** Trước đây một
  lần import module đã đồng thời: mở file log (`setup_logging`), đọc model
  registry + `RAGSettings` từ env, monkey-patch asyncio proactor, dựng
  container/`AgentService`/`McpToolServer` và build cả FastAPI app + mount
  toàn bộ router. Hậu quả: test import module = thực thi graph thật; composition
  không discoverable, không có thứ tự phụ thuộc tường minh, và bất kỳ thay
  đổi thứ tự nào cũng là regression câm.
- **Một hàm duy nhất, idempotent, có thứ tự.** `compose()` build cả object
  graph theo đúng thứ tự phụ thuộc (logging → proactor patch → registry →
  env toggles → instances → container → services → app → router → A2A/MCP).
  Khoá `RLock`; cờ `_COMPOSED` đặt trước body để chặn back-edge
  `compose() → _register_optional_protocols() → compose()`; nếu body raise,
  wrapper công khai reset cờ về `False` và propagate lỗi, nên lần access sau
  retry thay vì coi graph dở dang là đã composed (retry an toàn: handler log
  cũ bị thay, không nhân đôi output).
- **Ba lối vào, một semantics.** `main()` và `bootstrap_runtime()` gọi
  `compose()` không điều kiện; accessor đầu tiên của một tên trong
  `_COMPOSED_NAMES` (`server.app`, `from server import llm_module`, …) kích
  hoạt qua module `__getattr__` (PEP 562). Helper nội bộ đọc global composed
  (`_adopt_backend`, `load_model`, `_get_chatbot_agent`,
  `_register_optional_protocols`, …) tự guard `compose()` ở dòng đầu — gọi
  trực tiếp từ test không phụ thuộc thứ tự chạy.
- **Giữ nguyên ở import, có lý do.** `load_local_dotenv` +
  `apply_startup_environment_defaults` chạy trước mọi import đọc env (banner
  A2A — ADR 0001/0014); hai chèn `sys.path` là source-launch plumbing;
  `_SERVER_START_TIME` là hằng số thuần. `_chatbot_agent_holder` là list
  rỗng — cheap, không phải side effect.
- **Khai báo type dạng bare annotation** (`logger: logging.Logger`, `app: FastAPI`,
  …) ở module scope: mypy không nhận binding sinh từ `global` trong hàm
  (131 lỗi `name-defined` nếu thiếu), còn annotation không bind tên ở
  runtime nên `__getattr__` vẫn kích hoạt cho tới khi `compose()` gán thật.
- **Pin hành vi.** `LazyCompositionTests`
  (`tests/integration/test_composition_root_bootstrap.py`): một subprocess
  interpreter mới chứng minh import không tạo `app`/`logger`/`platform`,
  `compose()` dựng graph đúng một lần (identity của `app` không đổi sau lần
  gọi thứ hai), và tên lạ vẫn raise `AttributeError` bình thường.
