# ADR 0027 — Mã hoá at-rest cho state cục bộ (task + pending action)

## Bối cảnh

`Docs/SCORECARD.md` ghi một gap còn mở sau vòng review cuối: nội dung task và
pending action tới đĩa dưới dạng **plaintext**. `tasks.sqlite` giữ prompt của
người dùng, đối số tool, kết quả model và event log; `pending_agent_actions.json`
giữ đối số tool và ngữ cảnh chờ phê duyệt. Cả hai nằm dưới `APP_DATA_DIR` và
thừa hưởng ACL của thư mục đó — không phải mã hoá, chỉ là phân quyền.

Gap này **không** là một cổng đo được (coverage/mutation/eval đều xanh), nên
theo luật ADR 0015 nó phải được **đóng bằng code và ghim bằng test**, không
được ghi thêm một dòng "roadmap item" vào tài liệu. Đây là bước đó.

Ràng buộc thiết kế: state cục bộ là của *tiến trình*, không phải của *service
nhiều người dùng*, nên phải giữ được ba điều: (1) desktop single-user không
phải cấu hình gì; (2) mở state cũ (plaintext) không được làm brick ứng dụng;
(3) thiếu/sai key phải **fail closed**, không bao giờ trả ciphertext cho tầng
trên.

## Quyết định

1. **Bọc theo field bằng AES-256-GCM.** Adapter mới
   `adapters/persistence/storage_crypto.py` cung cấp `AesGcmFieldCipher`
   (thi triển protocol `FieldCipher`: `seal` / `open` / `is_sealed`) — seam nhỏ
   để test tiêm cipher giả mà không cần chạm đĩa. Định dạng envelope:
   `enc:v1:<url-safe base64(nonce || ciphertext)>`, nonce 96-bit **random cho
   từng field** (không tái sử dụng nonce), và **tên field làm associated
   data** (`ai_assistant.local_state.v1:<context>`) nên một ciphertext bị
   chuyển sang cột khác sẽ **fail authentication** thay vì đọc ra rác.
2. **Quản lý key tường minh, fail-closed, theo thứ tự nguồn:**
   - `AI_STORAGE_KEY` — `base64:<url-safe>` / `hex:<digits>` / base64 trần,
     phải giải ra **đúng 32 byte**;
   - `AI_STORAGE_KEY_FILE` — secret do deployment mount; file thiếu hoặc
     không đọc được thì **abort lúc compose**, không sinh key thay thế (key
     mới sẽ làm mọi row cũ không đọc được);
   - nếu không có gì: sinh **một lần** tại `settings.storage_key_location()` —
     **ngoài** `APP_DATA_DIR` (`%LOCALAPPDATA%`/`$XDG_CONFIG_HOME`) để backup
     volume dữ liệu không tự động mang theo key — với quyền owner-only (0600
     trên POSIX, ACL account trên Windows). Tranh chấp tạo file dùng
     `O_EXCL`: tiến trình thua cuộc **dùng key trên đĩa**, không dùng key riêng
     của nó. File key được `fsync` **kèm thư mục chứa nó**: `fsync` file chỉ
     đảm bảo *dữ liệu* nằm trên đĩa, còn entry thư mục là metadata riêng — nếu
     không flush nó, mất điện có thể làm **file key biến mất**, và lần khởi
     động sau sẽ sinh key mới và orphan mọi row (đúng lỗ hổng mà chính sách
     fail-closed này sinh ra để chặn).
3. **`settings.py` vẫn là chủ sở hữu duy nhất của môi trường** (ADR 0001,
   0014): `storage_key()` và `storage_key_location()` chỉ *đọc* env; adapter
   crypto chỉ *kiểm tra* key đã giải. Không module nào khác đọc env lúc import.
4. **Cipher được tiêm, không phải global.** `compose()` gọi
   `load_storage_cipher()` **một lần**, đặt vào `PlatformContainer.storage_cipher`,
   rồi truyền vào `SqliteTaskStore(..., cipher=…)` và
   `PendingActionStore(..., cipher=…)`. Không có service locator, không có
   module-level mutable state; một composition thứ hai (test, embedded host) có
   key riêng.
5. **Migrate plaintext cũ đúng một lần.** SQLite dùng `PRAGMA user_version`:
   v3 = "đã niêm phong". Khi mở store **có** cipher mà version < 3, adapter seal
   các cột `message`/`result`/`error`/`metadata` và payload event còn plaintext
   (bỏ qua giá trị đã seal để không bọc hai lần). File pending JSON không có
   bảng nên giữ nguyên: `open()` trả **nguyên văn** giá trị chưa seal, và lần
   `save()` kế tiếp sẽ ghi lại dưới dạng envelope.
6. **Đọc thì fail closed.** Không có cipher mà gặp giá trị đã seal →
   `StorageKeyError` (không trả ciphertext lên tầng trên). Sai key hoặc row bị
   sửa → `StorageDecryptionError`. **Thông điệp lỗi không echo ciphertext hay
   key material** vì chúng được ghi log và có thể tới màn hình vận hành.
7. **Chỉ field chứa nội dung người dùng mới được mã hoá.** `id`,
   `capability`, `status`, `context_id`, `owner`, `classification`, `origin`,
   `created_at`/`updated_at` và `task_events.kind` vẫn plaintext, để SQL
   filter / phân trang / retention / kiểm tra quyền sở hữu theo tenant vẫn
   chạy trên chỉ mục thật. `task_events.data` và toàn bộ nội dung pending
   **có** mã hoá.

## Hệ quả

- **Gap của SCORECARD đã đóng.** Không còn prompt / đối số tool / kết quả
  model nào nằm plaintext trên đĩa. Phần còn lại của `Docs/RUNBOOK.md` §1.1
  (mã hoá toàn volume, ACL, bảo mật backup) **vẫn đúng** — field-level
  encryption không thay thế mã hoá ở tầng volume, nó chỉ phủ đúng các field
  nêu ở điểm 7.
- **Mô hình thù địch đã ghi rõ.** Cơ chế này chống được: sao chép volume dữ
  liệu, đọc backup, một account khác trên cùng máy. Nó **không** chống được
  attacker đã chạy được code dưới user đó — họ đọc được file key. Cùng giới
  hạn đã nêu cho HMAC cache RAG (ADR 0013/0021), và cùng cách ghi: nói rõ
  giới hạn thay vì ghi "đã mã hoá" rồi im lặng.
- **Mất key = mất state, và đó là chủ ý.** Không có đường "quên key thì tạo
  key mới và coi như cache miss" — đường đó sẽ biến mất state thành dữ liệu
  rác âm thầm. `AI_STORAGE_KEY_FILE` tồn tại chính là để production đặt key
  vào secret store và backup nó cùng quy trình backup volume.
- **Desktop không phải cấu hình gì:** không có env nào thì key tự sinh ở
  `%LOCALAPPDATA%\3d-reconstruction\ai_assistant.key` và dùng lại ở lần sau.
- **Chi phí:** `cryptography` trở thành runtime dependency (`requirements.txt`
  đã pin `cryptography>=42.0.0`; lock file giữ hash). Bọc theo field tốn
  overhead AES-GCM trên mỗi ghi — không đáng kể so với I/O SQLite, nhưng đây
  là lý do `AI_STORAGE_KEY` vẫn được khuyến nghị thay vì sinh key.
- **Ghim bằng test, không bằng lời.** `tests/unit/test_storage_crypto.py`
  (envelope + context binding + tamper + nguồn key + race),
  `tests/unit/test_sqlite_store_crypto.py` (seal trên đĩa, migrate một lần,
  fail-closed khi thiếu/sai key),
  `tests/unit/test_pending_store_crypto.py` (seal payload, đọc plaintext cũ,
  fail-closed khi thiếu key). **10 mutant mới** trong `scripts/mutation_check.py`
  đảo từng quyết định fail-closed của lớp này (key file thiếu bị thay thế,
  envelope bị sửa, key sai độ dài, race khóa, không flush thư mục chứa key,
  đọc sealed khi thiếu key, ghi plaintext, không migrate row cũ, seal hai lần) và
  bắt buộc test sở hữu phải fail.

## Không giải quyết

- **Xoay key (re-key).** Bộ key hiện tại không có đường re-encrypt state cũ khi
  đổi `AI_STORAGE_KEY`; đổi key nghĩa là state cũ không đọc được. Đây là
  việc vận hành (migrate bằng công cụ riêng), chưa phải hạ tầng.
- **DPAPI/KMS cho file key.** Mặc định file key dựa vào ACL của account; một
  deployment cần mạnh hơn thì mount secret và chỉ đường `AI_STORAGE_KEY_FILE`
  (hoặc `AI_STORAGE_KEY` từ KMS) — cơ chế đó đã có, nhưng bản thân việc gọi
  KMS/DPAPI chưa được implement trong repo này.
- **Mã hoá toàn volume.** Vẫn là việc của vận hành (BitLocker/LUKS), xem
  `Docs/RUNBOOK.md` §1.1.
