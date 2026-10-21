# ADR 0034 — Những control mà gate không nhìn thấy

- Status: Accepted
- Date: 2026-10-21
- Supersedes: none

## Bối cảnh

Một review kiến trúc chạy trên **source**, không chạy trên `SCORECARD.md`.
Nó chạy lại mọi gate trước, rồi đọc code đằng sau chúng. Gate **thật sự xanh**:
lint sạch, mypy strict 0 lỗi, 9/9 import contract, 2 034 test, coverage
**100,00 %** (9 095 statement / 2 388 branch), mutation 74/74 bị giết.

Kiến trúc lõi đứng vững. Nhưng review tìm ra tám lỗ hổng, và chúng có
**cùng một đặc điểm**: mọi gate đều xanh *vì chúng không thể nhìn thấy lỗi*.

| Lỗi | Vì sao gate xanh |
|---|---|
| `read_file` đọc được `AIAssistant/.env` — khoá at-rest và **mọi** bearer token — **không cần approval** | Block-list là *extension*, mà dotfile không có extension (`splitext(".env") == (".env", "")`) — nhánh guard không bao giờ chạy |
| `shutil.which` trên Windows tìm CWD trước PATH → binary agent cài được chạy thay binary thật | Test "giữ" control này **mock `shutil.which` trả về `None`** — một tiền đề không bao giờ xảy ra ngoài đời |
| Deny-list ghi file so khớp **phân biệt hoa thường** → `PYPROJECT.TOML` ghi đè `pyproject.toml` thật | Không có test nào thử biến thể hoa thường |
| `AesGcmFieldCipher.open()` trả plaintext chưa seal → **AEAD downgrade** | `open_bytes` (cùng class) từ chối; nhánh text thì không, và không test nào chạm |
| A2A discovery **theo redirect**, dispatch thì không | Hai chân transport dùng hai opener khác nhau; test chỉ phủ chân dispatch |
| Egress chỉ dò **tên key** → secret nằm trong *value* (chính là prompt của người dùng) đi ra ngoài | `has_sensitive_keys` đúng theo tên của nó — và đó là tên của điều nó làm |
| Composition root **ghi được** → agent sửa được cổng OpenAPI, đường dẫn dotenv, nơi nạp cipher | Deny-list phủ "cây policy" nhưng không phủ chính lớp lắp ráp |
| Guard symlink **không đo được trên Windows** → mutation gate đỏ trên đúng nền tảng sản phẩm | Test `skip` khi hệ điều hành không cho tạo symlink |

Đây đúng là lớp lỗi ADR 0029 đã vá một lần cho `/v1/chat/completions`:
*"mọi case đều mock collaborator — mà mock trả lời mọi attribute"*. Lần này nó
quay lại ở lớp sandbox: **control có thật, test có thật, nhưng test không thể tái
hiện tấn công**.

Điểm mấu chốt: `SCORECARD.md` là một bản đo trung thực. Nó đo *gate*, không đo
*hệ thống*. "100,00 %" là con số thật — về một bộ test mà không thấy được tám
lỗi trên.

## Quyết định

### 1. Secret file bị chặn theo **tên**, ở một chặn duy nhất

`tools/builtin/path_utils.py` giờ có `is_secret_name(name)`: một tập tên cấm
rõ ràng (`.env`, `.envrc`, `.npmrc`, `secrets.json`, `id_rsa`, …) cộng với họ
`.env*` (`.env`, `.env.local`, `.env.production`, …).

Guard nằm trong `agent_safe_path` — hàm mà **mọi** công cụ đọc đi qua — và nó
so cả hai cách viết: cách *lexical* người gọi yêu cầu và cách *thật* một
symlink trỏ tới. Một link tên `notes.txt` chỉ vào `.env` không phải là lối vào.

`search_text` dùng `os.walk` nên không đi qua `agent_safe_path`; nó dùng chung
`is_secret_name` để bỏ qua file theo tên — vì bước quét chạm `secrets.json`
(một file `.json`) *trước* mọi dotfile, và bộ lọc extension không chặn được nó.

Phía *ghi* vốn đã chặn `.env` đúng; giờ phía *đọc* cũng vậy.

### 2. Resolver PATH tự quét, không bao giờ chạm CWD

`shutil.which` bị loại khỏi `sandbox.py`. Trên Windows nó **chèn thư mục hiện
tại vào đầu** search path, nên:

* binary cài trong CWD thắng binary của operator;
* giá trị trả về là **tương đối** (`.\\git.BAT`), không phải hit tuyệt đối mà
  docstring hứa;
* lời từ chối "một tên PATH không mang" **không bao giờ kích hoạt**, vì
  `which` đã tìm thấy file cài rồi.

`_resolve_executable` giờ tự quét các entry của PATH, theo thứ tự, và **không
bao giờ nhìn CWD**. Kết quả là đường dẫn tuyệt đối, nên hệ điều hành không thể
sắp lại thứ tự tìm kiếm khi tiến trình con khởi động: chương trình chạy đúng
là chương trình đã được kiểm.

`_executable_candidates` tách riêng phần giải `PATHEXT` của Windows, và quyết
định nền tảng là **hằng module** (`_IS_WINDOWS`) để cả hai nhánh đều đo được
trên cả hai nền tảng — coverage chạy trên Linux *và* Windows, một nhánh chỉ
một nền tảng chạm được sẽ kẹt ở 99 % vĩnh viễn trên nền tảng kia.

### 3. Deny-list so khớp không phân biệt hoa thường

`_DENIED_WRITE_NAMES` là bản gấp lowercase của `_DENIED_WRITE_PATHS`, và phía
so sánh dùng `target_name.lower()`. Hệ điều hành tập tin Windows không phân
biệt hoa thường, nên `PYPROJECT.TOML` **chính là** `pyproject.toml` — so
sánh phân biệt hoa thường để biến thể lọt qua và bản ghi thật bị thay.

Chuỗi substring dự phòng không che được trường hợp này: `pyproject.toml` là
entry duy nhất không phải một mảnh đường dẫn.

### 4. Cột đã seal phải *được* seal

`SqliteTaskStore` giờ theo dõi **chế độ sealed**: sau khi migration một lần
chạy xong (`PRAGMA user_version = 3`), mọi cột nội dung đều được ghi qua
cipher, nên một giá trị **không** phải envelope seal thì **không chính danh**.

Trước đây `open()` trả plaintext một cách trung thực, nên bất kỳ ai có quyền
ghi `tasks.sqlite` đều thay được một cột đã seal bằng JSON plaintext và ứng
dụng **phục vụ nó như thể chính nền tảng đã ghi** — thẻ GCM không bao giờ có
cơ hội từ chối. Đây là AEAD downgrade.

Artifact nhị phânn (`open_bytes`) đã từ chối envelope chưa seal với đúng lý do
đó; các cột text giờ hành xử như vậy. Lối thoát legacy được **giới hạn trong
migration**: một store *không* có cipher vẫn là store plaintext theo thiết kế
(đường opt-in).

`PendingActionStore` migrate theo hướng khác: một file legacy được **niêm
phong lại và ghi ngược lại** ngay lần đọc đầu tiên, nên bản plaintext không còn
tồn tại trên đĩa thay vì nằm lại cho tới lần `save()` kế tiếp — có thể là
nhiều ngày, hoặc không bao giờ.

### 5. Discovery và dispatch dùng **cùng một** opener

`a2a_protocol` giờ dùng chính opener chặn redirect của `a2a_client`. Trước đây
chân dispatch cố ý từ chối 30x (*"following it would turn one compromised peer
into SSRF against anything on the host's network"*) còn chân discovery dùng
opener mặc định của urllib — tức là **theo** redirect. Discovery là chân chạy
lúc startup với một peer chưa được thẩm định, nên đó đáng lẽ phải là chân
*không* yếu hơn.

### 6. Egress hỏi cả *key* lẫn *value*

`domain/governance.py` có hai quy tắc được đặt tên, cùng một chủ sở hữu
(ADR 0032: one owner, two named rules):

* `has_sensitive_keys` — quy tắc *key*: tên của thuộc tính có nói lên đây là
  credential không;
* `has_sensitive_content` — quy tắc *rộng hơn*: key **hoặc** hình dạng của
  giá trị.

Quy tắc value cố ý **không mơ hồ**: một credential có hình dạng rõ (một phép
gán `api_key=…` với giá trị đủ dài, hoặc một định dạng token tự mang dấu hiệu
— `sk-`, `ghp_`, `AKIA`, PEM, JWT). Một mẫu văn xuôi (*"the password is stored
in the vault"*) không thể phân biệt được với secret thật, và một biên giới từ
chối theo văn xuôi sẽ từ chối cả những yêu cầu thường lệ.

Hai đường đi khác nhau, đúng như tài liệu vốn đã nói: tham số để *thực thi* thì
**từ chối** (mask nó là giao cho peer một payload bịa đặt để chạy); message là
*văn xuôi tự do* của người dùng thì **mask** (peer không bao giờ cần mật khẩu
của họ). `TrustedA2AClient.submit` — chặn cuối trước khi ra dây — hỏi lại câu
hỏi thay vì tin rằng caller đã hỏi.

### 7. Composition root không ghi được

`_DENIED_WRITE_TREES` giờ phủ `bootstrap`, `llm`, `orchestration`, `plugins`,
`rag`, `observability` và `tools` bên trong gói `ai_assistant`.

ADR 0032 nêu tên các *file* policy. Nhưng lớp lắp ráp chính là policy:
`bootstrap/app_factory.py` quyết định có phục vụ bề mặt OpenAPI hay không,
`bootstrap/server.py` quyết định file dotenv nào chứa secret được nạp,
`bootstrap/modules.py` quyết định khoá at-rest dựng từ đâu. Một agent sửa được
chúng thì tắt được cổng mà không cần đụng vào một file policy nào.

### 8. Gate phải đo được trên đúng nền tảng

Guard symlink không thể chạy trên một Windows không bật Developer Mode, nên test
`skip` và mutant sống sót — mutation gate **đỏ trên đúng nền tảng sản phẩm**.
Guard đó giờ được đo bằng một đường khác không cần đặc quyền: `_canonical_path`
(ADR 0033) canonicalise *parent* rồi gắn lại *leaf*, nên đường dẫn tới một
file **chưa tồn tại** vẫn được so trên cùng một cách viết — và test case-variant
của `write_file` (một tên khác trong cùng thư mục policy) đập vào đúng nhánh đó
mà không cần tạo symlink.

## Hậu quả

* Tám lỗi ở trên đã đóng và được ghim bằng test hồi quy.
* Mỗi guard mới có **một mutant** trong cùng commit, và `MIN_MUTANTS` được nâng
  lên 79 — xoá một mutant bây giờ là một quyết định *nhìn thấy được*.
* Những control mà test chỉ chứng minh được *qua mock* được ghi lại trong
  `SCORECARD.md`, dưới mục "control được kiểm bằng mock" — để lần review
  sau không phải đoán.

### Gap còn lại, đã ghi nhận

* **Một pending-action store gốc không có neo bên ngoài.** Một envelope toàn
  file không phân biệt được "plaintext do phiên bản cũ viết" với "plaintext do
  kẻ tấn công thay vào" mà không có một cách hiệu dụng bên ngoài file đó. Cửa
  sổ bị thu hẹp còn *một lần đọc*; phần còn lại thuộc về operator (ACL, mã
  hoá ổ đĩa) — vốn đã là trách nhiệm của operator.
* **Bí mật trong văn xuôi ngắn không phát hiện được.** Mẫu hình dạng cần một
  giá trị đủ dài; một mật khẩu bốn ký tự trong câu văn không có hình dạng nào
  để nhận ra. Đây là giới hạn của bộ dò theo mẫu, được ghi rõ thay vì giấu.
