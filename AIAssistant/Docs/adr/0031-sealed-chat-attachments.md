# ADR 0031 — Chat attachment được niêm phong bằng chính key at-rest của dịch vụ AI

## Bối cảnh

ADR 0027 niêm phong *state* của agent, ADR 0028 niêm phong *cache RAG*. Còn một
kho thứ ba trong dữ liệu người dùng mà app tự quản lý: **file đính kèm của
chat**.

`FileUtilities::processAttachment` copy file người dùng chọn vào
`Upload/Images|Files/` và sinh thumbnail ở `Thumbnails/`. Cả hai là **plaintext**
trên đĩa. Lịch sử chat (lời thoại, phân tích của model về tài liệu của
người dùng) đã được niêm phong bằng DPAPI — nhưng file đính kèm thì không, nên
một bản sao thư mục dữ liệu vẫn lộ ra đúng những file đó (một bản chụp
CMND, một hợp đồng quét).

Vấn đề không đơn giản là "bọc file". Có ba ràng buộc kỹ thuật khiến cách
naive nhất (bọc bằng khoá riêng của app) **không dùng được**:

1. **Dịch vụ AI phải đọc được file.** `chat_routes._scan_attachments_for_image`
   và `llm/prompts.DefaultPromptBuilder.build_vision_messages` đều mở file
   đính kèm **theo đường dẫn** để đưa ảnh cho model. Nếu app niêm phong bằng
   khoá mà dịch vụ không có, ảnh sẽ không còn tới được model.
2. **Không thể gửi bytes inline.** Thân request bị chặn bởi
   `BodySizeLimitMiddleware` (2 MiB mặc định, `AGENT_MAX_REQUEST_BYTES`). Ảnh
   điện thoại 12 MP là 3–6 MB → vượt trần → 413 → tính năng hỏng. Hạ trần là
   tự làm yếu chống DoS.
3. **Hai tiến trình, hai hệ khoá.** App C++ dùng DPAPI (khoá của user profile);
   dịch vụ AI dùng khoá tệp AES-256-GCM ngoài `APP_DATA_DIR`. Chúng không chia
   sẻ khoá.

Điểm mấu chốt: **(1) và (3) cùng ép một hướng lời giải duy nhất** — dịch vụ
AI phải đọc được file đã niêm phong, nên **khoá phải là chung**.

## Quyết định

1. **Một khoá at-rest cho cả hai tiến trình.** App C++ dùng **đúng** khoá mà
   ADR 0027 quy định, theo đúng thứ tự nguồn: `AI_STORAGE_KEY` →
   `AI_STORAGE_KEY_FILE` → file khoá sinh một lần ở
   `%LOCALAPPDATA%\3d-reconstruction\ai_assistant.key` (Windows) hoặc
   `$XDG_CONFIG_HOME/3d-reconstruction/ai_assistant.key` (POSIX). Quy tắc này
   được **hiện thực hai lần** — một lần ở Python, một lần ở C++ — và được
   ghim bằng test.
   ⚠️ App C++ **không** dùng `QStandardPaths::AppDataLocation` cho đường dẫn
   này: nó cho ra `AppData/Local/<Org>/<App>`, khác đường dẫn phía Python.
2. **Cùng một định dạng envelope, byte-for-byte.**
   `enc:v1:` (7 byte) ‖ nonce 12 byte ‖ ciphertext ‖ tag GCM 16 byte, với
   AAD = `"ai_assistant.local_state.v1:" + context`. `context` cho file đính
   kèm là `"chat.attachment"`, cho thumbnail là `"chat.thumbnail"`. Phía C++
   dùng **BCrypt của Windows SDK** (`bcrypt.h`) — không thêm dependency thứ ba.
   Marker và context sống ở `ai_assistant/ports.py` vì chúng là **hợp đồng
   giữa hai tiến trình**, không phải chi tiết cài đặt.
3. **Cơ chế mở file là tiêm vào (DI), không phải import.**
   `image_utils.read_sealed_or_plain(path, unseal=...)` là *kernel leaf* — nó
   không được import adapter, nên caller truyền vào callable mở file.
   `DefaultPromptBuilder` và `build_chat_router` nhận `unseal` từ composition
   root. Đây là cùng nguyên tắc đã dùng cho `FieldCipher`: contract ở
   `ports`, implementation ở adapter.
4. **File cũ vẫn đọc được.** `open()` trên blob không phải envelope trả về
   nguyên bytes → attachment viết trước khi có tính năng này vẫn hiển thị, và
   lần ghi sau sẽ niêm phong lại.
5. **POSIX không có backend OS.** `SecretVault` (DPAPI) và `AtRestCipher`
   (BCrypt) đều là Windows. Trên POSIX chúng ghi plaintext kèm tiền tố
   `plain:` / trả nguyên bytes, và **điều đó được ghi rõ** trong
   `Docs/ARCHITECTURE_SRC.md` §4.1 chứ không được giấu. State của dịch vụ AI
   vẫn được niêm phong trên mọi nền tảng vì `cryptography` là đa nền tảng.
6. **Không đổi gì về hành vi người dùng.** Thứ tự byte mà model nhận **không
   đổi**: dịch vụ AI vẫn tự giải mã rồi resize/Encode bằng PIL, y như cũ.
   Đây là lý do phương án "dịch vụ AI nhận ảnh đã resize của app" bị loại: nó
   đổi encoder nên đổi input của model.

## Hệ quả

- **Kho plaintext cuối cùng trong dữ liệu người dùng đã đóng.** Lịch sử
  chat, cache RAG, file đính kèm, thumbnail: tất cả đã niêm phong.
- **Một khoá, ba nơi dùng** (state của agent, cache RAG, file đính kèm). Sao
  lưu phải giữ **một** file khoá, không phải nhiều.
- **App C++ giữ khoá của dịch vụ AI.** Đây là sự kết dính có chủ ý: hai tiến
  trình cùng đọc một kho at-rest. Hệ quả: app C++ phải đọc được file khoá ở
  `%LOCALAPPDATA%` — tức là **bất kỳ tài khoản nào chạy được app cũng đọc
  được khoá**. Với DPAPI thuần, khoá gắn với profile; với khoá tệp, nó gắn
  với ACL. Đây là mức bảo vệ tương đương phần state của dịch vụ AI, và được
  ghi lại thay vì coi là không quan trọng.
- **Hai hiện thực của cùng một hợp đồng** (Python `cryptography`, C++ BCrypt)
  phải giữ khớp byte-for-byte. Rủi ro lệch được giảm bằng cách đặt marker
  và context vào `ports.py` và ghim bằng test ở cả hai phía.
- **Chi phí:** một lớp crypto mới ở `src/core` (~150 dòng), BCrypt (Windows
  SDK, không dependency mới), và việc đọc file đính kèm phải đi qua đường mở
  file thay vì `QPixmap(path)`.
- **Một hệ quả hiển thị được:** link `file://` tới một file **đã seal** sẽ
  mở ciphertext ra bằng trình ứng dụng của hệ điều hành. Vì vậy lịch sử
  chat hiện file đính kèm đã seal **dưới dạng tên, không có link**; file
  plaintext cũ (trước khi có tính năng này) vẫn giữ link cũ. Không có đường
  "giải mã ra file tạm để mở" — nó sẽ tạo lại bản plaintext trên đĩa, đúng
  thứ mà tính năng này sinh ra để chặn.
- **Ghim bằng test:** hai mutant ở `tests/mutation_check.py` (bỏ bước seal khi
  ghi, bỏ bước unseal khi đọc) phải làm đỏ `TestChatSessionManager`; phía
  Python, `tests/unit/test_sealed_attachment_read.py` phủ vòng tròn mở file đã
  seal, file plaintext cũ, và việc thiếu khoá.
- **Hợp đồng chéo ngôn ngữ được ghim bằng vector, không bằng lời.** Đây là
  hợp đồng giữa **hai tiến trình**, nên "trông giống nhau" không đủ: mỗi bên
  giữ **output của bên kia** làm vector cố định
  (`TestAtRestCipher::thePythonServiceEnvelopeIsReadable` mở envelope do
  Python tạo; `test_sealed_attachment_read.py` mở envelope do C++ tạo). Một
  thay đổi làm lệch hợp đồng sẽ làm đỏ một test, thay vì hiện ra giữa chừng
  dưới dạng "trợ lý không nhìn thấy ảnh".

## Không giải quyết

- **POSIX vẫn không mã hoá** ở phía app C++ (xem trên). Muốn có thì cần một
  backend crypto cho POSIX trong `AppCore` — đó là một quyết định dependency
  riêng, chưa làm.
- **Tài liệu nguồn** vẫn là file của người dùng; mã hoá volume là việc vận
  hành (xem `Docs/RUNBOOK.md` §1.1).
- **Xoay key** cho các file đính kèm: khoá của file đính kèm là khoá at-rest
  chung, nên nó đi theo ADR 0030 (xoay khoá).
