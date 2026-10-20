# ADR 0026 — A2A registry và remote lookup thuộc composition root

## Bối cảnh

Hai mục cuối của review kiến trúc là state toàn tiến trình trong đường A2A:

- `adapters/a2a_protocol.py` giữ `_remote_registry` (dict) + `_registry_lock`
  ở module scope; `discover_remote_agents()`, `get_remote_registry()` và
  `resolve_remote_specialist()` là các hàm module đọc/ghi dict đó.
- `orchestration/supervisor.py` giữ `_remote_specialist_lookup` và cho
  composition root cài bằng `set_remote_specialist_lookup()`.

Cách này tránh được import cycle (orchestration không import adapter), nhưng
đổi lại: hai lần `compose()` trong cùng process (test, embedded host, restart
mềm) chia sẻ và có thể ghi đè state của nhau; lookup có thể còn trỏ vào
adapter đã bị thay; và một test phải reset global sau mỗi ca.

## Quyết định

1. `RemoteAgentRegistry` là một lớp sở hữu `dict` + `RLock` **theo instance**,
   với `discover()`, `find()`, `resolve()`, `snapshot()`.
2. `compose()` tạo đúng một `RemoteAgentRegistry`, rồi:
   - truyền cùng instance đó vào `A2ARouter(registry=...)`;
   - truyền `functools.partial(delegate, remote_lookup=registry.resolve)` vào
     `AgentService(delegate_fn=...)`;
   - gắn instance vào `app.state.a2a_registry` để lifespan discovery dùng lại.
3. `orchestration.supervisor.delegate()` nhận `remote_lookup` như keyword-only
   argument và gọi nó trực tiếp. Không còn setter toàn cục; không truyền
   lookup thì routing ở lại local.
4. `ApprovalService`/`UIActionService` lấy `delegate_fn` từ `AgentRunContext`
   thay vì import hàm `delegate` trực tiếp, nên resume path dùng đúng lookup
   đã tiêm.
5. Không còn hàm module `discover_remote_agents`,
   `resolve_remote_specialist`, `get_remote_registry`; test tạo registry riêng
   và truyền vào router.

## Hệ quả

- Không còn state A2A toàn tiến trình trong production path. Hai composition
  độc lập không nhìn thấy registry của nhau.
- Import cycle vẫn bị chặn: `orchestration` chỉ biết callable
  `RemoteSpecialistLookup`, không biết adapter.
- Discovery và routing dùng chung một instance, nên kết quả discovery có hiệu
  lực ngay cho `A2ARouter` lẫn supervisor mà không cần đồng bộ toàn cục.
- Test phải tiêm registry thay vì patch `_remote_registry`; regression được
  pin bởi `tests/unit/test_a2a_import_order.py`,
  `tests/unit/test_a2a_protocol_guards.py`, `tests/unit/test_a2a_router.py`,
  `tests/unit/test_a2a_http_router.py` và
  `tests/integration/test_server_module.py`.
- `A2ARouter()` không truyền registry vẫn hợp lệ: nó nhận một registry rỗng,
  nên các test/CLI local-only không cần dựng graph.

## Không giải quyết

At-rest encryption của task/pending-action content vẫn là roadmap item riêng;
ADR này chỉ đóng phần process-wide state.
