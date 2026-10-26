# ADR 0043 — Memory: một chủ sở hữu cho ngữ cảnh ngắn hạn, kho ba loại nhớ dài hạn

- Status: Accepted
- Date: 2026-10-26
- Supersedes: none

## Bối cảnh

Đối chiếu bảy năng lực của một agent (audit 2026-10-26), `AIAssistant` có
checkpoint, HITL, observability, evaluation, supervisor + A2A — nhưng **không
có memory**. Cái được gọi là "nhớ" chỉ là lịch hội thoại do client Qt gửi kèm
mỗi request, cộng vài xử lý rải rác:

1. `llm/prompts.py:trim_history` cắt lịch theo token (drop-oldest).
2. `orchestration/nodes/summarize.py:summarize_messages` nén bớt theo ký tự.
3. `orchestration/state.py` giữ ba hằng số ngân sách (`CONTEXT_*_LIMIT`).

Ba quyết định về **cùng một** ngữ cảnh nằm ở ba module khác layer: một transport
(`llm`), một node (`orchestration`) và một file state. Hai nửa của một chính
sách có thể trôi dạt (đúng bệnh mà `domain/context_budget.py` đã đóng ở lần
trước). Ngoài ra:

- một session sống lâu mất ngữ cảnh ngay khi client gửi history cắt ngắn;
- không có gì vượt qua ranh giới session: run hôm trước làm gì, biết gì, làm
  thế nào — lần sau hỏi lại từ đầu;
- không có chỗ nào ghi lại "cách đã làm được" để planner học lại.

## Quyết định

### 1. `domain/memory.py` là chủ sở hữu duy nhất của cả hai nửa

- **Ngắn hạn (STM)**: `trim_history`, `summarize_messages`,
  `estimate_tokens` và các hằng số ngân sách chuyển về `domain/memory.py`
  (stdlib-only, không IO, không LLM). Các đường import cũ
  (`llm.prompts`, `orchestration.nodes.summarize`, `orchestration.state`)
  trở thành delegate/re-export mỏng — tiền lệ ADR 0019 với `Step`.
- **Dài hạn (LTM)**: từ vựng `MemoryKind = semantic | episodic | procedural`
  (đóng), dạng bản ghi `MemoryRecord`, và ranker thuần túy `recall_rank`.

### 2. STM thêm buffer per-session có giới hạn

Client vẫn gửi lịch nó giữ (không đổi wire format), nhưng
`ShortTermMemoryPort` giữ thêm một dải bounded per-session: khi lịch client
gửi ngắn hơn những gì server từng thấy, buffer lấp khoảng trống. Giới hạn
số tin nhắn là ràng buộc sống còn — session sống lâu không được phát triển
không giới hạn (cùng triết lý `TaskCoordinator._evict_locked`).

### 3. Recall semantic v1 = keyword + recency, thuần deterministic

`recall_rank(records, query, top_k)` chấm theo
`overlap * (0.5 + 0.5 * decay) + 0.01 * salience * decay`, nửa đời
`MEMORY_HALF_LIFE_DAYS = 30`. Không embedding, không FAISS trong v1:

- truy vấn **rỗng** = duyệt theo recency ("tao nhớ gì về session này");
- truy vấn có token mà **không có record nào khớp** → loại bỏ, không xếp cuối:
  tiêm một "memory" không liên quan vào prompt là nhiễu, và một ranker trả về
  hàng không-overlap sẽ làm điều đó cả ngày;
- hòa điểm phá bởi `created_at` giảm dần rồi `id` tăng dần — thứ tự không bao
  giờ phụ thuộc thứ tự duyệt dict/set.

Embedding (tái dùng stack RAG sẵn có) là v2, thay `recall_rank` sau cổng port
không đổi.

### 4. Ghi LTM: chỉ run thành công, trích xuất deterministic — không gọi LLM

Consolidation chạy sau khi run pass reflection/thành công, trên worker thread,
trích xuất bằng rule trên chính state đã có (goal, outcome, dãy tool, artifact,
plan đã pass critic). Gọi LLM để tóm tắt sẽ phải lấy model lease
(ADR 0040/0041) và ghi rác của run thất bại vào kho — cả hai đều không đáng
giá trong v1. Distillation bằng LLM để v2, đúng luật lease.

### 5. Kho LTM đóng kín bằng chữ ký, fail-closed khi redaction

- Bản ghi đi qua `redact_value` / `has_sensitive_content`
  (ADR 0010): **không redact được thì bỏ bản ghi** — không bao giờ ghi
  plaintext nhạy cảm.
- LTM sealed at-rest bằng `FieldCipher` với context gắn row
  `"memory.v1:<id>:payload"` (ADR 0027 + quy tắc gắn id dòng của ADR 0036):
  bản ghi bị lấy khỏi store không mở được ở chỗ khác, và một payload chép
  sang dòng khác fails authentication.
- Dữ liệu rehydrate là JSON qua biên tin cậy, nên `is_memory_kind()` là song
  sinh runtime của `MemoryKind` — y hệt `validate_steps` với `StepType`
  (ADR 0019). Kind lạ bị loại ngay ở cửa.

### 6. Hai port, inject như `checkpointer`

`ShortTermMemoryPort` / `LongTermMemoryPort` trong `ports.py`;
`AgentService` nhận cả hai qua constructor, mặc định `None` = **tắt** —
không có store/cipher thì không có ghi gì (fail-closed như checkpointer).
`MemoryService(None)` cũng bất hoạt (`enabled` False). Adapter SQLite nằm ở
`adapters/persistence`, composition root inject; tầng agent không biết adapter.

### 7. Procedural memory seed planner theo task signature

Plan đã pass reflection, khóa theo chữ ký của task, được nạp lại ở
`plan_node._reuse_prior_plan` — phiên bản cross-session của cơ chế reuse
sẵn có trong một run. Cùng chữ ký thì cùng công thức; khác chữ ký thì planner
hỏi model như bình thường.

V1 chỉ giữ **text các bước** của plan (`" | ".join(steps)[:600]` khi ghi),
nên `plan_spec` khi nạp lại là `{"steps": plan}` — không có `step_kinds`, và
`plan_step_execution_contract` rơi về heuristic từ khoá đúng như nó làm với
plan không có kind (`helpers.py`). Plan nạp lại được đánh dấu
`plan_verified=True` **không chạy lại critic**: critic đã pass lúc ghi, và
chạy lại chính là chi phí mà seed tồn tại để tiết kiệm — các cửa chặn thực
thi (approval, validation, supervisor) vẫn áp dụng y như plan mới. Đường đi
không đụng adapter: `MemoryService.procedural_plan(task)` (application) →
`AgentService` tính trước khi run → `prior_plan` đi cùng `initial_steps`
qua `AgentRunContext.run` → `run_langgraph_agent` → `graph.run` → key
`prior_plan` của `AgentState`, và `_reuse_prior_plan` đọc từ state.

### 8. Step type mới `"memory_recall"` (additive)

Một recall được quan sát được trên step log UI. `StepType` là Literal đóng;
client Qt so sánh bằng chuỗi trong chuỗi `if/else` **không có nhánh bắt buộc**
nên type lạ bị bỏ qua an toàn; `validate_steps` giữ nguyên mọi payload key
(ADR 0019).

### 9. Cấu hình qua `settings.py`

`AGENT_MEMORY_ENABLED` (mặc định bật), `AGENT_MEMORY_STM_MAX_MESSAGES`,
`AGENT_MEMORY_LTM_TOP_K`, `AGENT_MEMORY_RETENTION_DAYS`,
`AGENT_MEMORY_RETENTION_MAX_MB` — đọc đúng một chỗ, đúng lối
`AgentRuntimeSettings.from_env` (ADR 0001).

## Hệ quả

- Một chính sách ngữ cảnh, một địa chỉ: thêm một ngân sách là thêm một hằng số
  trong `domain/memory.py`, không phải sửa ba file và hi vọng chúng khớp.
- Session sống lâu giữ ngữ cảnh ngay khi client gửi history ngắn; bộ nhớ vẫn
  có giới hạn.
- Cross-session recall được — "lần trước bạn đã làm X", "người dùng thích Y" —
  không phụ thuộc việc client còn giữ lịch hay không.
- Run thất bại không làm nhiễm kho; bản ghi nhạy cảm không bao giờ ra đĩa
  plaintext.
- Tắt memory = một cờ, không cần đụng graph.

## Ghim bằng test và mutant

- `tests/unit/test_memory_domain.py`: từ vựng ba kind, `is_memory_kind` từ
  chối kind lạ, `recall_rank` (top_k=0, loại không-overlap, duyệt recency khi
  truy vấn rỗng, decay nửa đời, tie-break, record thiếu trường, truy vấn chỉ
  có ký tự đặc biệt), `trim_history` (không cắt khi vừa vặn, drop-oldest,
  không bao giờ rỗng, bỏ qua content không phải str), `summarize_messages`
  (histogram ngắn giữ nguyên, history dài nén vào ngân sách), trích xuất
  (`task_signature` bất biến với dấu/cách/hoa-thường, `build_run_records`:
  episodic luôn có, procedural chỉ khi có plan và khóa theo chữ ký, semantic
  ≤ `MAX_SEMANTIC_PER_RUN`, id semantic ổn định để thay thế không tích lũy,
  `format_memory_block` rỗng → `""`).
- `tests/unit/test_short_term_memory.py`: buffer không phát triển vô hạn
  (cửa sổ, bản đồ session, forget/cleanup) và facade đọc/ghi đúng hợp đồng.
- `tests/unit/test_summarize_guards.py`, `tests/unit/test_prompts.py`,
  `tests/unit/test_prompt_builder_guards.py`: hành vi STM cũ giữ nguyên qua
  delegate — cùng test, cùng kỳ vọng, địa chỉ đổi.
- `tests/unit/test_memory_store.py`: khosealed — từ chối khi không có key,
  round-trip, bền vững, payloadsealed (không plaintext), context gắn với id
  dòng, từ chối kind lạ khi ghi, thay thế theo id, cách ly owner, thứ tự +
  giới hạn, bỏ qua dòng không mở được / không phải object / context lạ /
  identity lệch, prune theo tuổi và theo byte (oldest-first), close idempotent.
- `tests/unit/test_memory_service.py`: service fail-closed (store None, top_k=0,
  store ném lỗi, kind lạ bị loại ở biên tin cậy), redaction gate (token shape
  sống sót bị bỏ whole; keyword assignment được mask và bản ghi vẫn vào),
  ghi lỗi bị containment, facade recall (block + step `memory_recall`), chỉ
  consolidate run completed; procedural lookup exact-key (tắt memory/top_k=0,
  store ném lỗi, sai signature, kind khác bị loại dù store lọc sai, metadata
  không phải mapping, text không phải str/tách rỗng, bản ghi đầu tiên thắng)
  và facade offer `prior_plan` cho runner.
- `tests/unit/test_plan_node_guards.py`: seed procedural — plan nhớ được reuse
  mà không gọi planner (kèm plan step trên log, `plan_verified=True`,
  `plan_attempts+1`), plan trong run vẫn thắng plan nhớ, mid-task vẫn skip
  trước seed, plan nhớ rỗng rơi về planner.
- `tests/unit/test_graph_run_guards.py`, `tests/unit/test_runner.py`,
  `tests/unit/test_agent_run_context.py`: `prior_plan` đi hết bốn hop
  (context → runner → `graph.run` → `input_state`), thiếu hop nào cũng đứt.
- `tests/integration/test_server_module.py`: lifespan đóng store khi có và
  bỏ qua khi tắt.
- `tests/integration/test_composition_root_bootstrap.py`: compose với
  `AGENT_MEMORY_ENABLED=0` không xây store, service bất hoạt.
- Mutant (15): recall ném lại lỗi store, bỏ re-check kind, redaction gate
  tắt, ghi lỗi không được containment, `enabled` với top_k=0, kho không key,
  kind lạ được ghi, identity lệch được phục vụ, byte cap bỏ, tuổi không
  clamp, cap semantic bỏ, procedural khi không có plan, facade không recall,
  consolidate mọi status, announce recall rỗng.
- Mutant procedural (13): lookup bỏ off-switch top_k, ném lại lỗi store,
  nhận mọi kind, bỏ so signature, split text không phải str, trả `[]` khi
  tách rỗng; seed bị tắt, không verified, không tính attempt; ba hop
  `prior_plan` bị drop (facade, context, runner) và `graph.run` gửi None.
  `MIN_MUTANTS` 171 → 199 (xem `scripts/mutation_check.py`).

## Không thay đổi

- Wire format của request/response và lịch do client gửi: không đổi.
- ADR 0019 (vocabulary domain), ADR 0027 (at-rest seal), ADR 0010 (redaction),
  ADR 0040/0041 (single model owner) đều được tôn trọng, không bị thay thế.
- Checkpoint/LangGraph vẫn là nguồn state của một run; memory là **bổ sung**,
  không thay thế nó.
