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
`MemoryService` **bắt buộc** một `owner` tường minh (xem §10); store `None`
vẫn bất hoạt (`enabled` False). Adapter SQLite nằm ở
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
không đụng adapter: `MemoryService.procedural_plan(task, owner=...)`
(application) → `AgentService` tính trước khi run → `prior_plan` đi cùng
`initial_steps`
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

### 10. Owner đi theo lời gọi, không nằm trong process

Composition root chỉ dựng **một** `MemoryService` cho cả process, nên biến
"của ai" thành trạng thái của service là biến mọi tenant A2A thành một
namespace chung: tenant B đọc được run của tenant A, và một task trùng chữ ký
ghi đè plan của A. Vì vậy:

- `MemoryService.__init__` **bắt buộc** `owner`; `recall`,
  `procedural_plan` và `consolidate_run` nhận `owner` per-call (keyword-only)
  để deployment nhiều tenant truyền tenant của chính lời gọi. Owner rỗng bị
  từ chối ngay tại cửa (`ValueError`) — nó không phải "không có owner" mà là
  một namespace không ai đọc.
- `AgentService.execute` / `_execute_locked` và `AgentRunContext.run`
  **bắt buộc** `owner` keyword-only:
  không có mặc định để một caller quên scope mà rơi im lặng vào memory của
  tenant khác. Đường loopback `/v1/agent/execute` (JSON và SSE) truyền
  `DEFAULT_MEMORY_OWNER`; `LangGraphAgentOrchestrator` truyền `task.owner`
  (tenant đã resolve từ bearer token), **không** truyền
  `principal.subject` — subject là `a2a:<task id>`, mỗi task một giá trị, sẽ
  làm mất memory tích luỹ của tenant.
- Id bản ghi mang namespace owner: `sha256(owner)[:16]` làm tiền tố cho
  episodic/procedural/semantic. `id` là primary key và `INSERT OR REPLACE`
  là cơ chế "chạy lại thì làm mới plan", nên nếu id không mang owner thì hai
  tenant chạy cùng task sẽ ghi đè nhau. `DEFAULT_MEMORY_OWNER` giữ hình dạng
  id lịch sử (không tiền tố) để bản desktop nâng cấp vẫn đọc và thay thế được
  dòng cũ.
- `MemoryStore.add` từ chối thay thế dòng của owner khác (`ValueError`): lớp
  namespace phải đúng, nhưng store không được là nơi xoá memory của tenant
  khác nếu lớp đó có bug.
- Consolidation chạy retention sau khi ghi thành công (khi bật), và lỗi prune
  được containment như mọi lỗi memory khác — nếu chỉ prune lúc khởi động,
  kho sẽ phình giữa hai lần restart.
- **Quyết định consolidate có đúng một chủ.** `AgentRunContext` giữ
  `memory_service` và `consolidate_completed(result, *, task, session_id,
  owner)`; `run` **bắt buộc** `owner` và gọi helper sau mọi graph completion.
  Nhờ vậy run mới, run resume sau approve và run resume sau desktop-ack cùng
  đi qua một quyết định, thay vì chỉ `_execute_locked` biết consolidate.
- **Owner đi cùng parked action.** `run_langgraph_agent` bắt buộc `owner`,
  ghi nó vào payload pending; action desktop được chain copy owner sang entry
  kế tiếp. Entry cũ (trước khi có owner) fallback về `DEFAULT_MEMORY_OWNER`
  thay vì crash hoặc ghi vào namespace rỗng.
- **Hai đường completion không có graph cũng gọi helper.** `ApprovalService.
  _approved` (chạy trực tiếp khi không resume được) và `UIActionService.
  _finalize` (kết thúc bằng text) consolidate envelope `completed`; run
  parked/rejected/cancelled và desktop-ack `failed` vẫn không ghi gì.

## Hệ quả

- Một chính sách ngữ cảnh, một địa chỉ: thêm một ngân sách là thêm một hằng số
  trong `domain/memory.py`, không phải sửa ba file và hi vọng chúng khớp.
- Session sống lâu giữ ngữ cảnh ngay khi client gửi history ngắn; bộ nhớ vẫn
  có giới hạn.
- Cross-session recall được — "lần trước bạn đã làm X", "người dùng thích Y" —
  không phụ thuộc việc client còn giữ lịch hay không.
- Run thất bại không làm nhiễm kho; bản ghi nhạy cảm không bao giờ ra đĩa
  plaintext.
- Hai tenant A2A không đọc được memory của nhau, kể cả khi chạy cùng một task:
  owner đi theo lời gọi và id bản ghi mang namespace.
- Mọi đường hoàn tất — run mới, approve, desktop-ack, fallback không graph —
  đều ghi memory dưới đúng tenant; run parked/failed/cancelled không ghi.
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
  id mang namespace owner và `DEFAULT_MEMORY_OWNER` giữ hình dạng lịch sử,
  `format_memory_block` rỗng → `""`).
- `tests/unit/test_short_term_memory.py`: buffer không phát triển vô hạn
  (cửa sổ, bản đồ session, forget/cleanup) và facade đọc/ghi đúng hợp đồng.
- `tests/unit/test_summarize_guards.py`, `tests/unit/test_prompts.py`,
  `tests/unit/test_prompt_builder_guards.py`: hành vi STM cũ giữ nguyên qua
  delegate — cùng test, cùng kỳ vọng, địa chỉ đổi.
- `tests/unit/test_memory_store.py`: khosealed — từ chối khi không có key,
  round-trip, bền vững, payloadsealed (không plaintext), context gắn với id
  dòng, từ chối kind lạ khi ghi, thay thế theo id, cách ly owner (thay thế
  cùng id khác owner bị **từ chối**, dòng của owner kia còn nguyên), thứ tự +
  giới hạn, bỏ qua dòng không mở được / không phải object / context lạ /
  identity lệch, bỏ qua **cột số bị sửa** (chuỗi và blob: dòng đó bị skip,
  các dòng lành vẫn trả về), prune theo tuổi và theo byte (oldest-first),
  close idempotent.
- `tests/unit/test_memory_service.py`: service fail-closed (store None, top_k=0,
  store ném lỗi, kind lạ bị loại ở biên tin cậy), redaction gate (token shape
  sống sót bị bỏ whole; keyword assignment được mask và bản ghi vẫn vào),
  ghi lỗi bị containment, facade recall (block + step `memory_recall`), chỉ
  consolidate run completed; procedural lookup exact-key (tắt memory/top_k=0,
  store ném lỗi, sai signature, kind khác bị loại dù store lọc sai, metadata
  không phải mapping, text không phải str/tách rỗng, bản ghi đầu tiên thắng)
  và facade offer `prior_plan` cho runner; quyền sở hữu (owner bắt buộc khi
  khởi tạo, owner rỗng bị từ chối, owner per-call thắng owner cấu hình, một
  run scope cả recall lẫn consolidate, retention chạy sau ghi thành công và
  lỗi prune được containment).
- `tests/integration/test_memory_tenant_isolation.py`: chuỗi production thật
  (`AgentTask.owner` → `LangGraphAgentOrchestrator` → `AgentService.execute`)
  trên store SQLite sealed thật — tenant B không recall được run của tenant A
  dù cùng task, consolidation của B không thay plan của A, và namespace
  desktop mặc định tách khỏi tenant.
- `tests/unit/test_plan_node_guards.py`: seed procedural — plan nhớ được reuse
  mà không gọi planner (kèm plan step trên log, `plan_verified=True`,
  `plan_attempts+1`), plan trong run vẫn thắng plan nhớ, mid-task vẫn skip
  trước seed, plan nhớ rỗng rơi về planner.
- `tests/unit/test_graph_run_guards.py`, `tests/unit/test_runner.py`,
  `tests/unit/test_agent_run_context.py`: `prior_plan` đi hết bốn hop
  (context → runner → `graph.run` → `input_state`), thiếu hop nào cũng đứt.
- `tests/unit/test_agent_run_context.py`: consolidation thuộc context — run
  `completed` ghi đúng owner, run parked không ghi, context không có memory
  vẫn chạy; `run` bắt buộc `owner`.
- `tests/unit/test_agent_resume_services.py`: approve resume và desktop-ack
  resume consolidate dưới owner đã park; fallback không graph (approve trực
  tiếp, desktop-ack finalize text) cũng consolidate; desktop-ack `failed`
  không ghi; action legacy thiếu owner dùng `DEFAULT_MEMORY_OWNER`; action
  desktop được chain giữ owner.
- `tests/unit/test_runner.py`: payload pending của cả approval lẫn UI-ack
  mang `owner` của run, không phải owner mặc định.
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
- Mutant cột-số (1, follow-up 2026-10-28): dòng có cột `created_at`/
  `salience` không phải số bị **phục vụ** thay vì skip — bản cũ để `float()`
  ném ra ngoài guard, nên một dòng bị sửa làm mất **toàn bộ** recall thay vì
  đúng dòng đó. Cùng hai mutant của ADR 0044 §9, mutant `Referer` normalization
  và mutant shutdown close containment, `MIN_MUTANTS` 221 → 226.
- Mutant ownership (12, follow-up 2026-10-28): cả **lớp** quyết định tenant,
  không chỉ một đại diện — id không mang namespace (hai tenant ghi đè nhau),
  owner rỗng được nhận, owner per-call bị bỏ qua, consolidation ghi dưới
  owner cấu hình thay vì owner của run, retention không chạy sau ghi, store
  thay thế dòng của owner khác, ba lời gọi memory ở facade bỏ `owner`,
  orchestrator không forward `task.owner`, và hai route loopback gán một
  tenant bịa thay vì `DEFAULT_MEMORY_OWNER`. `MIN_MUTANTS` 226 → 238.
- Mutant consolidation-on-resume (6, follow-up 2026-10-29): parked action
  quên owner của run, context consolidate cả run parked/failed, context bỏ
  owner khi gọi `consolidate_run`, action desktop chain reset owner, fallback
  approve không graph không consolidate, và fallback desktop-ack finalize
  không consolidate. `MIN_MUTANTS` 238 → 242. Bốn file mang quyết định này
  (`agents/context.py`, `agents/response.py`, `agents/approval_service.py`,
  `agents/ui_action_service.py`) được thêm vào `PINNED_DECISION_FILES`: một
  file không thể âm thầm mất mutant cuối trong khi tổng số vẫn trên sàn.

## Không thay đổi

- Wire format của request/response và lịch do client gửi: không đổi.
- ADR 0019 (vocabulary domain), ADR 0027 (at-rest seal), ADR 0010 (redaction),
  ADR 0040/0041 (single model owner) đều được tôn trọng, không bị thay thế.
- Checkpoint/LangGraph vẫn là nguồn state của một run; memory là **bổ sung**,
  không thay thế nó.
