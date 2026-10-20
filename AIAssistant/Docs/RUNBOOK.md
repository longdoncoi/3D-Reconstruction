# RUNBOOK — AIAssistant vận hành production

Tài liệu này dành cho profile **server/nhiều người dùng**. Desktop single-user
không cần runbook này: mọi cổng dưới đây là opt-in (ADR 0023/0024) và mặc định
giữ nguyên hành vi cũ.

## 1. Checklist bật profile production

Đặt trong `.env` (hoặc secret store) rồi restart server:

| Biến | Giá trị production | Vì sao |
|------|--------------------|--------|
| `AGENT_SANDBOX_RUNTIME` | `docker` + `AGENT_SANDBOX_IMAGE=<image đã duyệt>` | OS boundary cho tool execution |
| `AGENT_SANDBOX_REQUIRE_ISOLATION` | `1` | Từ chối chạy nếu container runtime không bật (fail closed) |
| `AGENT_CHECKPOINT_BACKEND` | `sqlite` hoặc `postgres` (+ `AGENT_CHECKPOINT_URL`) | Durable resume sau restart |
| `AGENT_CHECKPOINT_REQUIRED` | `1` | Từ chối rơi về memory im lặng |
| `AGENT_MAX_CONCURRENT_RUNS` | `2`–`8` tuỳ RAM/GPU | Trần burst nhiều session, hết slot trả 503 |
| `AGENT_RATE_LIMIT_PER_MINUTE` | `60`–`300` tuỳ client | Chặn vòng lặp client lỗi, trả 429 + `Retry-After` |
| `AGENT_OBSERVABILITY` | `1` (+ `AGENT_OTLP_ENDPOINT` nếu có collector) | Bật counter/histogram cho cảnh báo bên dưới |

Kiểm tra sau khi bật:

```powershell
python AIAssistant/scripts/verify_models.py --require-all   # pin model, exit 0
python AIAssistant/scripts/generate_sbom.py --output sbom.json
```

Log khởi động **không được** có `Checkpoint backend unavailable` hay
`Using MemorySaver`; có nghĩa là cổng chưa ăn.

## 1.1 Dữ liệu lưu cục bộ (threat model hiện tại)

`tasks.sqlite` (message/result/metadata/event log) và
`pending_agent_actions.json` (đối số tool, bước suy luận, ngữ cảnh chờ duyệt)
được mã hoá **at-rest theo field** bằng AES-256-GCM (ADR 0027): envelope
`enc:v1:<base64(nonce||ciphertext)>`, tên field làm associated data nên
ciphertext bị chuyển sang cột khác sẽ fail authentication. Cột chỉ mô tả
(`id`, `status`, `owner`, `context_id`, `classification`, `origin`, timestamp)
vẫn plaintext để SQL filter/retention/tenant còn chạy trên chỉ mục thật.

`Cache/` cũng đã được niêm phong (ADR 0028) — **cùng một key**: `chunks.pkl`
(nội dung tài liệu + ảnh base64), `bm25.pkl` và `faiss_index.bin` được seal
bằng AES-256-GCM, nên **bản sao volume dữ liệu không mang theo corpus tài
liệu ở dạng đọc được**. Tên file giữ nguyên để không đổi đường dẫn, nhưng
**nội dung không còn là pickle** — đừng mở bằng công cụ khác. Cache không
xác thực được (sai key, bị sửa, hoặc cache của version cũ) sẽ bị **dựng
lại**; cache là artifact dẫn xuất nên cái giá chỉ là thời gian.

**Key** lấy theo thứ tự, và mọi nhánh đều **fail closed** (server không khởi
động được nếu key hỏng — đó là chủ ý, vì một key sinh lại sẽ làm mọi state cũ
không đọc được):

| Nguồn | Dùng khi | Ghi chú |
|-------|----------|---------|
| `AI_STORAGE_KEY` | production có secret store | `base64:<url-safe>` / `hex:<digits>` / base64 trần; phải giải ra đúng 32 byte |
| `AI_STORAGE_KEY_FILE` | production mount secret file | File thiếu/không đọc được → **abort lúc khởi động**, không sinh key thay thế |
| tự sinh một lần | desktop single-user | `%LOCALAPPDATA%\3d-reconstruction\ai_assistant.key` (Windows) hoặc `$XDG_CONFIG_HOME/3d-reconstruction/ai_assistant.key` (POSIX), 0600 — **ngoài** `APP_DATA_DIR` nên backup volume không tự mang theo key |

Vì vậy, trước khi coi deployment là production:

- **backup key cùng quy trình backup volume** — mất key là mất state, và
  chính sách đó là *fail closed* chứ không phải *cache miss*;
- đặt `APP_DATA_DIR` trên volume được mã hoá toàn đĩa (BitLocker/LUKS) và chỉ
  cấp quyền cho account chạy service — mã hoá field-level **không** thay thế
  mã hoá ở tầng volume, nó chỉ phủ các field nêu trên;
- không mount chung volume dữ liệu giữa nhiều tenant hoặc giữa host/container
  không cùng trust boundary;
- mã hoá backup của volume; coi bản backup là dữ liệu RESTRICTED;
- không đưa `tasks.sqlite`/`pending_agent_actions.json` vào log, ticket hay artifact
  CI.

Cơ chế này chống được sao chép volume, đọc backup và account khác trên cùng
máy; **không** chống được attacker đã chạy được code dưới user đó (họ đọc
được file key). Cùng giới hạn đã nêu cho HMAC cache RAG. Xoay key (re-key) và
gọi KMS/DPAPI trực tiếp chưa có trong repo: xem
[`Docs/adr/0027-at-rest-encryption-local-state.md`](adr/0027-at-rest-encryption-local-state.md).

### 1.2 Xoay khoá at-rest (ADR 0030)

Khoá hiện tại **xoay được** — trước đây đổi khoá là mất toàn bộ state.
`scripts/rekey_storage.py` làm ba việc, theo đúng thứ tự:

```powershell
# 1. Kiểm tra khoá đang cầm có phải khoá state được seal không (read-only).
python AIAssistant/scripts/rekey_storage.py --verify

# 2. Xem trước sẽ xoay bao nhiêu dòng (không ghi gì).
python AIAssistant/scripts/rekey_storage.py --old-key <khoá-cũ> --new-key <khoá-mới> --dry-run

# 3. Xoay. Task store được mã hoá lại trong MỘT transaction; payload pending
#    được ghi lại nguyên tử; cache RAG bị XOÁ (nó là artifact dẫn xuất, dựng
#    lại được).
python AIAssistant/scripts/rekey_storage.py --old-key <khoá-cũ> --new-key <khoá-mới>
```

Sau đó **cài khoá mới** vào `AI_STORAGE_KEY` (hoặc `AI_STORAGE_KEY_FILE`) rồi
khởi động lại. Cho tới lúc đó app **fail closed** bằng `StorageKeyError` — đó
là chủ ý: khoá cũ không mở được state nữa, và im lặng chấp nhận state không
đọc được là tệ hơn dừng hẳn.

Quy tắc vận hành:

- **Luôn chạy `--verify` trước.** Nếu khoá không mở được state, đừng xoay —
  bạn đang cầm nhầm khoá.
- **Sao lưu volume trước khi xoay.** Công cụ báo trước số dòng sẽ bị
  mã hoá lại; nếu có sự cố, trả lại bản sao lưu.
- **Xoay là hành động của người vận hành**, không tự động: đổi khoá mà
  quên cài khoá mới là mất state.
- Cache RAG bị xoá có chủ ý — lần khởi động kế tiếp sẽ dựng lại, tốn thời
  gian chứ không mất dữ liệu.

## 2. SLO tối thiểu (khởi điểm, chỉnh theo tải thật)

Các ngưỡng dưới đây là *đề xuất khởi điểm* cho deployment nội bộ; số đo thật
phải lấy từ metrics của chính bạn. Không có con số nào ở đây được đo sẵn.

| SLO | Nguồn đo | Gợi ý |
|-----|----------|-------|
| Availability (`/readyz`) | probe của orchestrator | ≥ 99% theo tuần |
| p95 độ trễ node plan/reason/tool/reflect | `p95_latency_by_node()` (SCORECARD #2) | Theo baseline máy chạy; cảnh báo khi tăng > 30% so với tuần trước |
| Tỉ lệ 5xx | `agent_requests_total{status="503"}` / tổng | < 1% giờ cao điểm |
| Tỉ lệ throttle | `rate(agent_http_throttled_total[5m])` | < 5% request; cao hơn là quá tải hoặc client lỗi |
| Recovery cancel-to-stop | p95 `agent_recovery_seconds` (SCORECARD #4) | < 2s |
| Nguyên vẹn artifact | log `ModelIntegrityError` | 0 lần |

## 3. Cảnh báo (Prometheus)

```promql
# Client bị chặn liên tục — xem peer nào, đừng nới limit ngay
sum by (surface) (rate(agent_http_throttled_total[5m])) > 0.5

# Hết slot run (503 do budget): điều tra tải, không phải lỗi hệ thống
rate(agent_requests_total{status="503"}[5m]) > 0.1

# Trade-off: tỉ lệ lỗi tổng
sum(rate(agent_requests_total{status=~"5.."}[5m])) / sum(rate(agent_requests_total[5m])) > 0.01
```

Log-based (chưa có counter riêng, xác nhận trước khi trông cậy):

- `ModelIntegrityError` / `does not match the pinned SHA-256` → dừng rollout,
  chạy mục 4.5; **không** tắt verify.
- `Sandbox isolation is required by policy` → runtime docker chết; xem mục 4.4.
- `Checkpoint backend unavailable` khi `AGENT_CHECKPOINT_REQUIRED=1` → startup
  đã fail (đúng thiết kế); khi =0 là đang chạy memory, không durable.

## 4. Quy trình

### 4.1 Chỉnh quota (rate limit / run budget)

1. Xác định nguồn: 429 (rate limit) hay 503 (hết slot).
2. Nới bằng env rồi restart: `AGENT_RATE_LIMIT_PER_MINUTE`,
   `AGENT_MAX_CONCURRENT_RUNS`. Không cần deploy code.
3. Rate limiter là **per-process**: chạy nhiều replica thì giới hạn thật nằm ở
   ingress/reverse proxy — đặt ở đó trước, env chỉ là lớp trong cùng.
4. Probe `/livez`, `/readyz` được miễn 429; nếu chúng đỏ thì không phải limiter.

### 4.2 Checkpoint: durability, backup, retention, restore

- Backend: `sqlite` mặc định ghi tại `AIAssistant/Cache/checkpoints.sqlite`
  (`AGENT_CHECKPOINT_PATH`); `postgres`/`redis` qua `AGENT_CHECKPOINT_URL`.
- Retention: `cleanup_old_checkpoints(max_days=30, max_size_mb=50)` chạy trong
  lifespan lúc startup; file sqlite phình > 50MB sẽ được VACUUM. Giữ nguyên
  hành vi này trừ khi có lý do đo được.
- Backup sqlite: dừng server (hoặc `sqlite3 checkpoints.sqlite ".backup <đích>"`)
  rồi copy file — copy nóng có thể ra bản hỏng.
- Backup postgres: `pg_dump` theo lịch của bạn; restore là quy trình của DBA.
- Restore sqlite: thay file khi server dừng, start lại với
  `AGENT_CHECKPOINT_REQUIRED=1`, chạy lại một phiên đã hủy trước đó để xác
  nhận resume.

### 4.3 Cổng checkpoint từ chối khởi động

`RuntimeError: AGENT_CHECKPOINT_REQUIRED=1 refuses the in-memory checkpointer`
→ đặt `AGENT_CHECKPOINT_BACKEND=sqlite|postgres|redis`. Backend cấu hình hỏng
(`RuntimeError: … failed to initialize`) → xem nguyên nhân chuỗi `from` trong
log (đường dẫn/DSN/package thiếu). Cả hai trường hợp là fail closed có chủ
đích: sửa cấu hình, không hạ cổng trừ khi chấp nhận mất durability.

### 4.4 Sandbox isolation

- Bật `AGENT_SANDBOX_RUNTIME=docker` + image đã duyệt trước khi bật
  `AGENT_SANDBOX_REQUIRE_ISOLATION=1`.
- Docker argv đã cố định `--network none`, `--cap-drop ALL`,
  `--security-opt no-new-privileges`; thêm cờ nới quyền là thay đổi bảo mật —
  cần review riêng.
- Khi lệnh bị từ chối vì policy: sửa lệnh/allow-list hoặc image; **không**
  hạ `AGENT_SANDBOX_REQUIRE_ISOLATION` để "chạy tạm".

### 4.5 Luân chuyển pin model (ADR 0023)

1. Chọn commit mới trên Hugging Face; lấy SHA-256 nội dung từ tree API
   (`?expand=true`, trường `lfs.oid`) hoặc hash file đã tải:
   `Get-FileHash -Algorithm SHA256 <file>`. Không dùng `x-linked-etag`.
2. Cập nhật **cùng lúc** `AIAssistant/config/models.toml` và hằng số tương ứng
   trong `src/ai_assistant/config/model_registry.py`.
3. `python AIAssistant/scripts/verify_models.py --require-all` → exit 0.
4. Chạy eval có so sánh: tải artifact `model-in-loop-report` của CI gần nhất,
   rồi `python AIAssistant/evals/run_model_in_loop_eval.py --model-path <gguf>
   --baseline <report.json>` — tụt quá `--baseline-tolerance` (mặc định 5%)
   thì dừng, điều tra trước khi rollout.

### 4.6 Chuỗi cung ứng định kỳ (mỗi release)

```powershell
cd AIAssistant
uv pip compile requirements.txt --python-version 3.11 --universal --generate-hashes -o requirements.lock
uv pip compile requirements-test.txt --python-version 3.11 --universal --generate-hashes -o requirements-test.lock
python scripts/generate_sbom.py --output sbom.json
git diff requirements.lock requirements-test.lock   # review version nhảy có chủ đích
```

CI job `test` cài `requirements-test.lock` bằng `pip install --require-hashes`
— hash lệch thì job đỏ ngay ở bước install, trước cả khi test chạy. Nếu pip
báo `THESE PACKAGES DO NOT MATCH THE HASHES`, dừng lại: hoặc wheel bị
republish (điều tra trước khi cập nhật hash), hoặc lock sinh từ index khác.
Sửa range trong `requirements*.txt` mà quên regenerate lock thì
`test_lockfile_sync` (trong suite CI) đỏ kèm tên requirement — không có
đường "cài bộ cũ trong im lặng". Hash ở đây chứng minh *danh tính artifact*,
không phải provenance: nếu audit cần provenance thì phải bật
attestation (GitHub artifact attestations hoặc cosign/SLSA) — repo này chưa
bật, ghi nhận thay vì giả vờ có.
SBOM (171 distribution kèm hash + 6 artifact model) cũng được CI ghi thành
artifact `aiassistant-sbom` mỗi lần chạy `agent-evals.yml`; diff SBOM giữa
hai release là cách kiểm tra dependency trôi.

### 4.7 Eval & regression

- CI (`agent-evals.yml`) chạy tool eval (16 case, gồm adversarial),
  agentic eval (25 scenario/114 case, gồm gateway policy) và model-in-loop
  (7 case vi/en, gate coverage 0.9 / accuracy 0.5, model pin theo revision).
  Fallback 3B đã pin đo cục bộ: 6/7 kind đúng, coverage 7/7, p95 ~3.9s.
- Muốn so theo thời gian: tải artifact `model-in-loop-report`, dùng `--baseline`
  như mục 4.5 bước 4. Eval đỏ nghĩa là hành vi đổi — điều tra, không cập nhật
  kỳ vọng cho tới khi hiểu nguyên nhân.
