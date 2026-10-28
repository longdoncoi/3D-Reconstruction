# ADR 0045 — Bề mặt agent cũng là một port (`AgentToolCatalog`)

- Status: Accepted
- Date: 2026-10-28
- Supersedes: none (nối tiếp ADR 0032)

## Bối cảnh

ADR 0032 đã biến **đường thực thi** thành một port: `ToolCatalog` che
`plugins.registry.PluginRegistry` khỏi tầng application, nên policy chokepoint
chỉ phụ thuộc hợp đồng. Nhưng **bề mặt agent** thì vẫn buộc vào lớp cụ thể:
`agents/service.py`, `agents/deps.py`, `agents/graph_wiring.py` và
`agents/prompts.py` khai báo `ToolRegistry` (class trong
`tools/registry.py`), còn `AgentRunContext.tool_registry` là `Any`.

Hệ quả đo được:

- Tầng `agents` import một class cụ thể của gói `tools` ở module load, nên
  một catalogue khác (test double, embedder, bản thay thế) không thể thay
  vào mà không kế thừa đúng class đó.
- `Any` trên `AgentRunContext` làm mypy không kiểm tra được surface mà
  prompt builder, grammar và dispatch guard thực sự đọc.
- Review 2026-10-27 ghi đây là design debt còn mở (scorecard, mục "still
  open"), dù mọi cổng vẫn xanh.

Ràng buộc: `ToolDefinition` là `frozen dataclass`; protocol không được đòi
thuộc tính **ghi được** (mypy từ chối structural subtyping vì attribute
mutable là invariant), và việc thêm protocol không được kéo `tools` vào
import của `agents` (một bare facade/context vẫn phải dựng được catalogue
mặc định).

## Quyết định

### 1. Hai port chỉ-đọc, không mở rộng `ToolCatalog`

`ports.AgentToolSpec` mô tả **một** record mà agent đọc: `name`,
`description`, `parameters`, `requires_approval`, `handler`,
`to_openai_format()`. `ports.AgentToolCatalog` mô tả catalogue:
`get_all()`, `get(name)`, `get_openai_tools()`, `grammar`, `models`.

Đây là port **thứ hai**, không phải cha của `ToolCatalog`: execution
chokepoint không cần listing dạng prompt, và một catalogue chỉ phục vụ
execution vẫn là `ToolCatalog` hợp lệ. `ToolRegistry`/`ToolDefinition` thoả
cả hai bằng structural typing — không có `register()` nào bị kéo vào hợp
đồng agent.

### 2. Thuộc tính trong `AgentToolSpec` là `@property` chỉ-đọc

`ToolDefinition` frozen nên thuộc tính của nó chỉ đọc. Khai báo protocol bằng
attribute ghi được làm mypy báo `Following member(s) ... have conflicts`;
`@property` (covariant, read-only) khớp và vẫn chặn consumer ghi đè.

### 3. Default catalogue nằm ở `agents/catalog.py`, import lười

`default_agent_catalog()` import `tools.factory.create_tool_registry` **bên
trong hàm**. Vì vậy:

- `AgentService()` và `AgentRunContext().run(...)` vẫn tự dựng catalogue khi
  không có composition root (test, embedder).
- Import `ai_assistant.agents` không kéo `tools.factory`/handler built-in;
  edge `agents -> tools` chỉ tồn tại ở thời điểm gọi, không phải module load.

### 4. Đổi type ở đúng các điểm tiêu thụ

`AgentService.tool_registry`, `AgentRunContext.tool_registry`,
`RunDeps.tool_registry`, `_parse_tool_call(..., AgentToolCatalog, ...)` và
`build_agent_system_prompt(AgentToolCatalog | str | None)`. Không đổi hành
vi runtime: `ToolRegistry` vẫn là implementation mặc định.

## Hệ quả

- Tầng agent phát biểu đúng hợp đồng nó dùng; một catalogue khác chỉ cần
  thoả protocol, không cần kế thừa class cụ thể.
- `mypy --strict` kiểm tra được surface prompt/grammar/dispatch; tổng số file
  mypy 147 → 148 ở bước này (148 → 151 sau ADR 0046).
- Không có nhánh runtime mới, nên **không thêm mutant**: quyết định này là
  type-level và được ghim bằng mypy + các test hành vi sẵn có
  (`test_agent_run_context`, `test_agent_service_guards`, `test_agent_prompts`,
  `test_graph_wiring_delegations`, `test_agent_architecture` — 61 test).
  `MIN_MUTANTS` giữ nguyên 226.
- Coverage không đổi cách đo: thân `...` của protocol vẫn nằm trong
  `excluded_lines` (đã kiểm tra bằng `coverage json`), nên sàn 100 % không
  bị ảnh hưởng.

## Không thay đổi

- `ToolCatalog` (ADR 0032) và đường thực thi duy nhất (ADR 0003).
- `ToolRegistry` vẫn là catalogue mặc định do composition root dựng.
- Không đổi payload/grammar gửi model; chỉ đổi type của tham số.
