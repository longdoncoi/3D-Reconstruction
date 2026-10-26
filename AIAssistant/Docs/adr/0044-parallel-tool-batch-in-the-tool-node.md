# ADR 0044 — Parallel tool batch: song song trong node tool, không đổi topology

- Status: Accepted
- Date: 2026-10-26
- Supersedes: none

## Bối cảnh

Audit 2026-10-26: service **đã** chạy song song *giữa* các task
(`TaskService` thread + `BoundedSemaphore`, `AgentService.max_concurrent_runs`,
`SessionLocks`) nhưng *bên trong* một workflow thì tuyệt đối tuần tự: graph
LangGraph lấy một tool call mỗi lượt (`plan_step_index`), không có
`asyncio.gather` / `ThreadPoolExecutor` / `Send` ở đâu. Một plan với vài truy
vấn RAG độc lập hoặc vài research song song phải xếp hàng từng bước, mỗi bước
một vòng reason→tool→reflect.

Bốn ràng buộc cứng quyết định thiết kế:

1. **Một model lease** (ADR 0040/0041): llama.cpp là chủ sở hữu đơn. Song
   song LLM completion là phá hợp đồng — nên **chỉ tool được song song**.
2. **Một cổng thực thi duy nhất** (ADR 0003): mọi nhánh vẫn đi qua
   `ToolExecutionService` gateway — không có đường tắt.
3. **HITL một grant = một invocation** (ADR 0009): grant được cấp cho đúng
   lệnh người dùng đã duyệt và tiêu khi chạy; fan-out một lệnh cần approval
   làm grant đó không còn rõ nghĩa.
4. **Checkpoint an toàn** (ADR 0023/0027): state là một bản ghi duy nhất cho
   mỗi node; resume/HITL đọc đúng nó.

## Quyết định

### 1. Fan-out nằm trong node `tool`, không dùng LangGraph `Send`

`Send` đẩy state đi nhiều nhánh rồi chờ merge ở graph level: đỉnh topology
thay đổi, checkpointer phải ghi nhiều nơi, HITL resume và route `after_tool`
đều phải học lại hình dạng mới. Fan-out **bên trong** một node giữ nguyên
topology (plan → reason → tool → reflect), state chỉ được ghi **một lần**
sau khi merge — checkpoint, resume và router không đổi hình.

### 2. Planner phát biểu hợp đồng song song, validate fail-closed

`normalise_plan_payload` nhận thêm
`plan_spec["parallel_groups"] = [[idx...], ...]` — danh sách nhóm bước chỉ
số. Mọi nhóm không hợp lệ (chỉ số ngoài plan, trùng lặp, nhóm ≤ 1 phần tử,
phần tử không phải số) bị **loại bỏ entirely**, không phải sửa chữa: một planner
hỏng không được quyền bật behavior mới mà contract không mô tả.

### 3. Điều kiện vào nhóm: deterministic, rag-only, approval-free — không ngoại lệ

Một nhóm chỉ chạy song song khi **tất cả** thành viên:

- là bước `rag_search` **theo khai báo** (`plan_spec["step_kinds"]`), không
  phải theo contract đã resolve. Bước `model_selected` dù keyword trông giống
  RAG vẫn **không bao giờ** vào nhóm: nó cần model chọn tool/đối số giữa
  chừng (model lease, ADR 0040/0041), và một worker thread không được phép
  thay model quyết định đó;
- không thuộc `tools_requiring_approval`.

**`direct` cũng không vào nhóm.** Ban đầu ADR này xếp `direct` vào nhóm
"deterministic", nhưng nó hỏng theo một đường khác: bước `direct` chỉ được
chấp nhận khi reasoner trả `step_answer`, mà `completed_plan_steps` chỉ tiến
khi các bước được chấp nhận **liên tiếp**. Một batch chứa `direct` sẽ để bước
k đó chưa có câu trả lời, trong khi bước k+1 đã có evidence; reflection không
thể vượt qua k, nên vòng lặp không tiến được (deadlock tiến độ) — và batch
cũng chẳng có payload tool nào để chạy cho `direct`. Vì vậy điều kiện cuối
cùng là **rag-only**.

**Một thành viên không hợp lệ ⇒ cả nhóm rơi về tuần tự.** Không phải "nhánh
đó hỏi riêng": grant được cấp cho một invocation chính xác, người dùng phải
thấy rõ trình tự từng bước như trước, và một nhóm nửa song song nửa tuần tự
sẽ làm thứ tự `plan_step_index` không còn khớp với thứ tự thực thi. Fail-closed
theo nhóm là lựa chọn có chủ đích — an toàn hơn là từng nhánh tự quyết.

Điều kiện được kiểm tra lại **từ checkpoint** bằng đúng hàm validate của
parser (`valid_parallel_group`): spec trên đĩa do process khác ghi, compiler
không bảo chứng được hình dạng của nó.

### 4. Bounded: `AGENT_PARALLEL_MAX` (mặc định 4, trần cứng 8)

Mỗi nhánh là một thread OS chạy gateway/A2A. Không giới hạn là đổi
`max_concurrent_tasks` thành một lỗ hổng tài nguyên mới; trần cứng 8 là số
một host desktop chịu được cho tool I/O-bound.

### 5. Mỗi nhánh đi qua đúng đường cũ, gộp step có nhãn

- Nhánh gọi `ctx.execute` → `_execute_tool` (gateway) hoặc `_remote_dispatch`
  (A2A, registry đã thread-safe) như một lượt thường; không có executor thứ
  hai, không có gateway thứ hai. Head dùng đúng params model đã chọn; params
  của các sibling được suy ra từ chính text bước của chúng
  (`plan_step_execution_contract(kind="rag_search")`).
- Kết quả merge thành N cặp `tool_call` + `tool_result`, mỗi cặp mang
  `batch_id` và `branch_index` để UI/audit truy ngược. Head được đóng dấu tại
  **đúng vị trí** của nó trong log (dispatch giữ `head_index` theo identity),
  các sibling được tổng hợp thành `tool_call` để cặp call/result vẫn khớp
  từng một.
- **Chỉ head xin `step_answer`**: directive tổng hợp nêu tên bước hiện tại, và
  reflection của head mới là thứ `completed_plan_steps` chấp nhận. Bằng chứng
  của các sibling đi kèm trong **một** message phụ của cùng update; vòng
  `reason` kế tiếp trả lời từng bước từ context đó thay vì search lại (xem
  §8). Không sinh reflection cho từng thành viên: tiến độ plan vẫn do các
  lượt `step_answer` quyết định, đúng như đường tuần tự.
- **Cancel**: kiểm tra `cancel_checker()` trong thread dispatch, theo thứ tự
  nhóm, trước mỗi nhánh — nên *nhánh nào bắt đầu* là deterministic; nhánh bị
  bỏ nhận `{"error": "Tool batch branch cancelled."}` và batch trả
  `done=True, cancelled=True`. Mọi nhánh đã submit được `join`
  (`shutdown(wait=True)`) trước khi node trả state — không một nhánh nào sống
  sót sau node.
- Exception của một nhánh được contained trong chính nhánh (biến thành
  `{"error": ...}` như `_dispatch` vẫn làm): một tool nổ không giết luồng
  main, không nuốt kết quả các nhánh khác, và circuit breaker vẫn thấy lỗi
  (một batch sạch reset streak, batch lỗi cộng dồn).

### 6. Vá `_delegation_cache`

`graph_wiring._delegation_cache` là plain dict check-then-set — hai nhánh
đồng thời có thể cùng một lúc tính delegation, hoặc ghi đè kết quả của nhau.
Cache được bọc `threading.Lock` (lock chỉ giữ qua phép tra cứu/tính toán,
không giữ qua thực thi tool), và mọi phép tính delegation cho một batch chạy
**trước** fan-out qua `warm_delegations` (precompute tuần tự) để các nhánh
chỉ đọc entry đã ấm.

### 7. Observability: một span cha, ngữ cảnh sao chép mỗi nhánh

Span cha `agent.tool_batch` (thuộc tính `branches=N`) bao quanh cả fan-out;
mỗi thread chạy `contextvars.copy_context()` (pattern `TaskService._start`)
nên `traceparent`/OTel context lan đúng theo nhánh thay vì lẫn sang nhau, và
span `agent.tool` của gateway trong từng nhánh là con của span cha đó.

### 8. Vá hai lỗi tiềm ẩn mà batch làm lộ ra

- **Schema planner thiếu `step_kinds`.** `PLANNER_PROMPT` đã yêu cầu
  `step_kinds` từ trước, nhưng `PLANNER_JSON_SCHEMA` không khai báo nó, nên
  grammar của constrained decoder không thể sinh ra trường này và parser âm
  thầm rơi về heuristic keyword. Schema nay khai báo `step_kinds` (enum ba
  giá trị) và `parallel_groups` — hợp đồng prompt/schema/parser khớp nhau,
  nếu không thì điều kiện rag-only ở §3 không bao giờ có dữ liệu để đọc.
- **Directive mâu thuẫn với synthesis request.** Khi batch đã lấy evidence
  cho bước hiện tại, directive cũ vẫn bắt "phải dùng rag_search trước khi trả
  lời", trong khi tool node vừa yêu cầu `step_answer`. Directive nay kiểm tra
  `has_rag_evidence_for_step(steps, done_count)`: có evidence thì đòi
  `step_answer`, chưa có thì mới đòi search. Nếu không, model sẽ search lại
  đúng bước vừa chạy — lặp vô ích và trái với evidence vừa xác minh.

## Hệ quả

- Plan có các bước RAG/độc lập chạy nhanh hơn theo độ rộng nhóm, không đổi
  một dòng topology, checkpoint, router hay HITL.
- Không có đường nào thoát gateway, không có grant nào bị tiêu hai lần, không
  có LLM nào bị gọi song song.
- Nhóm không hợp lệ hoặc cần approval luôn rơi về tuần tự — hành vi cũ là
  mặc định an toàn, hành vi mới phải được chứng minh hợp lệ trước khi bật.

## Ghim bằng test và mutant

- `tests/unit/test_plan_parallel_groups.py`: hợp đồng `parallel_groups`
  (hợp lệ, chỉ số ngoài plan, trùng, giảm dần, nhóm lẻ, phần tử không phải
  số, `bool`), parser (`normalise_plan_payload`) và clamp
  `AGENT_PARALLEL_MAX` (sàn 1, trần 8, giá trị trong khoảng).
- `tests/unit/test_parallel_batch.py`: từng điều kiện loại nhóm (không plan,
  spec hỏng, head lạ, member `direct`, nhóm lệch plan sau checkpoint, cần
  approval, width ≤ 1), happy path 2 nhánh (2 cặp call/result, thứ tự
  `plan_step_index`, `batch_id`/`branch_index`, head giữ params gốc, sibling
  suy từ text, `top_k` cho role lookup, `tool_call_count` = số nhánh,
  sibling evidence trong message phụ, `completed_plan_steps` không tự tăng),
  concurrency thật (barrier + thứ tự merge ổn định), cap qua pool
  `max_workers`, exception một nhánh, verify từng nhánh, warm delegation,
  cancel giữa batch (nhánh bị bỏ + `done`/`cancelled` + join), circuit
  breaker và reset streak, head được đóng dấu đúng vị trí trong log.
- `tests/unit/test_graph_wiring_delegations.py`: `warm_delegations` phân loại
  một lần cho mỗi cặp khác nhau; hai thread cùng một cặp chỉ tính một lần
  (lock); wiring truyền `max_parallel`/`requires_approval` xuống graph.
- `tests/unit/test_reason_prepare_guards.py` (`RagDirectiveTests`): directive
  đòi `step_answer` khi evidence đã có, đòi search khi chưa có.
- `tests/unit/test_completion_guard_paths.py`: `PLANNER_JSON_SCHEMA` khai báo
  `step_kinds` (đúng enum) và `parallel_groups`.
- Mutant (22, `MIN_MUTANTS` 199 → 221): bỏ điều kiện approval-free, bỏ
  fallback width, bỏ kiểm tra cancel từng nhánh, bỏ join, bỏ reset streak,
  bỏ sibling evidence, gộp cả nhóm thành 1 call, đóng dấu head vào `steps[0]`,
  bỏ kiểm tra kind của member, bỏ cổng rag-only của head, bỏ khớp head của
  nhóm từ checkpoint, bỏ lock cache, bỏ warm-up, bỏ sàn/cap width, bỏ bốn
  guard của `valid_parallel_group` (kích thước, thứ tự/trùng, biên plan,
  `bool`), directive quay lại luôn đòi search, schema mất `step_kinds`.

## Không thay đổi

- ADR 0040/0041 (single model owner) — LLM không nằm trong batch.
- ADR 0003 (single execution core), ADR 0009 (single-use grants),
  ADR 0023/0027 (checkpoint), ADR 0019 (step vocabulary — thêm key
  `batch_id`/`branch_index` là additive).
- Không đổi request/response schema; step mới vẫn là `tool_call`/
  `tool_result` quen thuộc với client Qt.
