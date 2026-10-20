# ADR 0030 — Xoay khoá at-rest: re-encrypt state cũ thay vì vứt bỏ nó

## Bối cảnh

ADR 0027 bọc **mọi** nội dung state cục bộ bằng một khoá at-rest duy nhất:
task store (`tasks.sqlite`), pending-action store
(`pending_agent_actions.json`) và — theo ADR 0028 — cả cache RAG. Cùng một
khoá đó, theo đúng thứ tự nguồn `AI_STORAGE_KEY` → `AI_STORAGE_KEY_FILE` →
file khoá sinh một lần.

Nhưng chính ADR 0027 đã ghi ra gap của nó ở mục *Không giải quyết*:

> **Xoay key (re-key).** Bộ key hiện tại không có đường re-encrypt state cũ khi
> đổi `AI_STORAGE_KEY`; đổi key nghĩa là state cũ không đọc được.

Hệ quả cụ thể: **đổi khoá = mất toàn bộ state.** Không phải "mất cache" —
mất prompt người dùng, đối số tool chờ phê duyệt, kết quả model và event log.
Với một deployment đã chạy production, `AI_STORAGE_KEY` là *dữ liệu*: nó nằm
trong secret store, nó được backup, và đổi nó (vì chính sách yêu cầu đổi định
kỳ) sẽ làm toàn bộ state trên đĩa thành rác. Người vận hành chỉ còn hai lựa
chọn: **không bao giờ đổi khoá** (vi phạm chính sách) hoặc **xoá state**
(mất dữ liệu). Cả hai đều sai.

Còn một thiếu sót nữa, không nhìn thấy được nếu không có công cụ: trước khi
xoay, vận hành viên **không có cách nào hỏi "khoá tôi đang cầm có phải khoá
đã dùng để seal state trên đĩa không?"**. Sai khoá ⇒ state không đọc được,
và sai khoá là chuyện xảy ra thật (file khoá bị copy từ máy khác, secret
store trả bản cũ, biến môi trường bị sót).

Cả hai đều là **vận hành**, không phải kiến trúc: không có đường re-encrypt
thì không có đường xoay khoá.

Ràng buộc thiết kế giống ADR 0027: state cục bộ là của *tiến trình*, nên
phải giữ (1) không bao giờ để lại state *nửa* mã hoá — nửa đọc được bằng khoá
cũ, nửa bằng khoá mới, tức là **không đọc được bằng khoá nào**; (2) mọi thao
tác phải **fail closed**: sai khoá thì abort, không phải "bỏ qua row hỏng";
(3) phải **lặp lại được** — một lần xoay bị ngắt giữa chừng phải chạy lại được,
không để lại một state không nhất quán.

## Quyết định

1. **Re-key ở tầng store, không phải ở tầng tool.**
   `SqliteTaskStore.rekey(old, new)` và `PendingActionStore.rekey(old, new)`
   nhận **hai** cipher tường minh và trả về số row đã re-encrypt. Mỗi giá trị
   được mở bằng khoá cũ và seal lại bằng khoá mới với **đúng context** của nó
   (`task.message`, `task.result`, `task.error`, `task.metadata`,
   `event.data`, `pending.actions`) — context là associated data của AEAD, đổi
   nó là ciphertext không còn xác thực được. Cột `NULL` (`result`, `error`)
   vẫn là `NULL`.
2. **Atomic, fail-closed, và không bao giờ nửa mã hoá.**
   - Task store: **một transaction** bao trọn vòng re-key. Một giá trị mà khoá
     cũ không mở được ⇒ abort ⇒ rollback ⇒ database còn **nguyên trạng** (byte
     của các cột seal không đổi), và ứng dụng vẫn đọc được bằng khoá cũ.
   - Pending store: payload là **một** blob seal duy nhất, nên nó được mở
     trước, seal lại sau, rồi mới ghi — ghi qua file tạm rồi `replace` (đúng
     như `save()`). Sai khoá ⇒ file cũ **không bị đụng** tới.
   - Cột "nửa mã hoá" là trạng thái không tồn tại trong thiết kế này: hoặc
     transaction rollback trả về nguyên trạng, hoặc file không bao giờ bị thay.
3. **Lặp lại được (idempotent).** Một giá trị **đã** gắn với khoá mới được
   bỏ qua, không seal lần hai. Chạy cùng một cặp `(old, new)` hai lần: lần hai
   báo `0` row và state vẫn đọc được. Hệ quả thực hành: **xoay bị ngắt giữa
   chừng thì chạy lại là xong** — phần đã xoay được bỏ qua, phần chưa thì
   được xoay.
4. **Plaintext legacy được seal luôn.** Một giá trị chưa seal không gắn với
   khoá nào, nên nó được seal bằng khoá mới: sau khi xoay, **không còn giá trị
   nào đọc được bằng mắt thường** trên đĩa.
5. **Cache RAG bị XOÁ, không re-key.** Cache là **artifact dẫn xuất** — nó luôn
   dựng lại được từ tài liệu nguồn (ADR 0028 đã nói rõ: cache không xác thực
   được là *miss*, và caller dựng lại). Re-key cache là viết lại hàng chục MB
   để giữ một thứ không giữ gì; xoá nó chỉ tốn **thời gian dựng lại**, không
   tốn dữ liệu. Vì vậy công cụ **xoá** `faiss_index.bin`, `chunks.pkl`,
   `bm25.pkl` và `metadata.json` và nói rõ điều đó.
6. **Một công cụ vận hành, fail closed, không bao giờ lộ khoá.**
   `scripts/rekey_storage.py`:
   - `--verify` — **chỉ đọc**: báo bao nhiêu task row / event / pending payload
     mà khoá đang cấu hình mở được, và **exit ≠ 0** nếu mở không nổi. Đây là
     câu hỏi "khoá của tôi có đúng không?" mà vận hành viên cần trước khi
     xoay.
   - `--dry-run` — báo những gì *sẽ* được re-encrypt, **không ghi gì**.
   - `--new-key` / `--new-key-file` — khoá mới, đúng các định dạng mà
     `AI_STORAGE_KEY` chấp nhận (`base64:<url-safe>`, `hex:<digits>`, base64
     trần → **đúng 32 byte**). Bắt buộc cho hành động xoay.
   - `--old-key` / `--old-key-file` — khoá cũ; mặc định lấy khoá **đang cấu
     hình** (`AI_STORAGE_KEY` → `AI_STORAGE_KEY_FILE` → file khoá sinh sẵn).
     Công cụ **không** sinh file khoá mới: một khoá mới sinh ra không mở
     được state nào.
   - `--data-dir` — ghi đè nơi state nằm (mặc định: đúng đường dẫn mà
     composition root dùng).
   - Mọi lỗi — khoá sai định dạng, khoá không mở được state, không tìm thấy
     state — đều **abort với thông điệp rõ và exit ≠ 0**. Công cụ **không bao
     giờ** ghi một state nửa xoay.
   - **Khoá không bao giờ được in ra.** Chỉ có *bước tiếp theo* được in:
     cài khoá mới vào `AI_STORAGE_KEY` / `AI_STORAGE_KEY_FILE` trước khi khởi
     động lại. (Khoá nằm trong history của shell và trong log nếu bị in.)
7. **Quy trình vận hành (đúng thứ tự này):**
   1. `--verify` với khoá **cũ** → phải báo OK. Không OK thì dừng: khoá đang
      cầm không phải khoá đã seal state.
   2. `--dry-run` với khoá cũ + khoá mới → xem số row/payload sẽ bị re-encrypt.
   3. Chạy xoay thật (khoá cũ + khoá mới) → in ra số row đã re-encrypt, số
      pending payload, số file cache đã xoá.
   4. **Cài khoá mới** vào `AI_STORAGE_KEY` (hoặc `AI_STORAGE_KEY_FILE`) và
      khởi động lại ứng dụng.
   5. Nếu bước 3 bị ngắt, chạy lại **cùng cặp khoá** — phần đã xoay được bỏ
      qua, phần chưa thì được xoay.

## Hệ quả

- **Gap của ADR 0027 đã đóng.** Xoay khoá không còn là "mất state": nó là một
  lệnh vận hành có kiểm tra trước, có báo cáo sau, và **lặp lại được**.
- **Chi phí của lần xoay là thời gian, không phải dữ liệu.** Cache RAG bị
  dựng lại; task/pending state thì giữ nguyên nội dung.
- **Trạng thái sau khi xoay:** đúng như một lần cài mới — state trên đĩa được
  seal bằng khoá mới, và **khoá cũ không mở được nữa** (đã được test). Nếu
  vận hành viên quên cài khoá mới, ứng dụng **fail closed** với
  `StorageKeyError` — đúng như ADR 0027, không phải "chạy tiếp với rác".
- **Mô hình thù địch không đổi.** Xoay khoá không làm state tự nhiên an toàn
  hơn: nó chỉ đổi khoá. Kẻ đọc được file khoá vẫn đọc được state (cùng giới
  hạn đã nêu ở ADR 0027/0028).
- **Ràng buộc vẫn giữ:** `AI_STORAGE_KEY` vẫn là **một** khoá cho mọi state
  at-rest; không phát sinh "khoá thứ hai" và không có đường "khoá cũ làm
  fallback" (fallback đó biến việc xoay thành việc *không bao giờ* xoay).
- **Ghim bằng test, không bằng lời.** `tests/unit/test_storage_rekey.py`
  (round trip + khoá cũ bị tước quyền đọc, khoá cũ sai ⇒ abort và state còn
  nguyên, store rỗng là no-op, xoay hai lần không seal hai lần, `NULL` vẫn
  `NULL`, plaintext legacy được seal, công cụ verify/dry-run/rotate và các
  nhánh lỗi của nó). **3 mutant mới** trong `scripts/mutation_check.py` đảo
  từng quyết định fail-closed của lớp này: re-key chấp nhận giá trị mà khoá
  cũ không mở được, re-key không còn là **một** transaction (xoay bị ngắt để
  lại state nửa mã hoá), và công cụ bắt đầu xoay mà **không** kiểm tra khoá cũ
  có mở được state hay không.

## Không giải quyết

- **KMS / DPAPI cho file khoá.** File khoá mặc định vẫn dựa vào ACL của
  account (ADR 0027). Xoay khoá *không* thay thế được việc đó: một deployment
  cần khoá trong KMS thì vẫn phải mount nó qua `AI_STORAGE_KEY_FILE` (hoặc
  `AI_STORAGE_KEY` từ KMS) và gọi công cụ này khi đổi.
- **Xoay khoá không tự động (unattended).** Không có scheduler, không có
  daemon, không có chính sách hạn dùng. Việc xoay là hành động của vận hành
  viên, có `--verify` và `--dry-run` để họ làm đúng.
- **Lịch sử khoá.** Chỉ có **một** khoá tại một thời điểm; không có "khoá
  trước" để thử khi nghi ngờ, và không có cách đọc state bằng cả hai khoá.
  Một khoá bị mất là mất state (đúng như ADR 0027 đã chấp nhận).
- **Cache RAG không được re-key** — nó bị xoá và dựng lại. Xem mục 5 ở trên.
- **Tài liệu nguồn và chat attachment.** Tài liệu nguồn là file của người
  dùng (mã hoá volume là việc vận hành). Chat attachment dùng **chung** khoá
  at-rest (ADR 0031), nên nó đi theo ADR này: xoay khoá là xoay luôn.
