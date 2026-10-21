# ADR 0035 — Một quy tắc, một chủ sở hữu: đóng lớp lỗi thay vì từng thực thể

- Status: Accepted
- Date: 2026-10-21
- Supersedes: none

## Bối cảnh

ADR 0034 đóng tám lỗi, và kết luận của nó là: *"mọi cổng đều xanh vì chúng
không thể nhìn thấy lỗi"*. Để kiểm chứng cái kết luận đó, một audit mới được
chạy trên **các lớp chưa từng được đụng tới** — RAG, HTTP routes, plugin
loader, orchestration, các tool còn lại — với đúng lăng kín của ADR 0034:
*control bảo mật chỉ được ghim bằng mock* và *so sánh sai thứ*.

Audit tìm thêm **năm lỗi cùng lớp**. Và điều đáng nói nhất: **một trong số đó
là anh em của lỗi ADR 0034 vừa sửa** — cùng một lớp "so sánh sai thứ", cách
hai mươi dòng, trong một quy tắc khác của *cùng một hàm*.

| Lỗi | Vì sao gate không thấy |
|---|---|
| `.GIT/config` đọc được (không approval) — chứa remote URL và credential helper | Quy tắc *thư mục bị chặn* so **phân biệt hoa thường**, trong khi quy tắc *tên file bí mật* ngay cạnh đó so **không phân biệt** — một sự bất đối xứng nhìn thấy được ngay trong một hàm |
| `lsp_client` vẫn dùng `shutil.which` — binary cài trong CWD được chạy | Test mock `shutil.which` trả về **một tên tương đối** — đúng hình dạng mà một CWD-hit thật tạo ra. Đây là **chỗ thứ hai** giải chương trình, và ADR 0034 chỉ sửa chỗ thứ nhất |
| Cổng đọc của LSP so *chính tả* đường dẫn → junction trong project đọc ra ngoài | Mọi cổng đọc khác đều **fail-closed** (resolve rồi mới so) — riêng cổng này không |
| `vcs_tools` vứt bỏ hit tuyệt đối đã verify bằng `or "git"` → quay lại thứ tự tìm của OS | Test *khẳng định* hành vi không an toàn là đúng, với `subprocess.run` bị mock |
| Route chat mở **bất kỳ đường dẫn nào client đặt tên**, giữ cả at-rest key | Test mock `image_to_data_uri` — chưa từng có ca nào đọc file thật |

Hai điều này cộng lại định hình quyết định của ADR này.

### Điều thứ nhất: sửa từng thực thể là cách sửa sai

Nếu tám lỗi của ADR 0034 và năm lỗi này cùng một lớp, thì *đếm số lỗi đã
sửa* là một con số vô nghĩa: lớp lỗi đó vẫn còn nguyên, và thực thể thứ mười
bốn chỉ là một dòng code nữa. Đóng từng lỗi mà không đóng *lớp* là để lại
cửa sổ mở.

### Điều thứ hai: "10/10" không phải một kết luận, nó là một *phương pháp*

Điểm mấu chốt của ADR 0034 vẫn đúng: một bảng cổng xanh là một phép đo của
*gate*, không phải của *hệ thống*. Chứng minh điều đó lần thứ hai bằng chính
công việc vừa rồi không làm nó yếu đi — nó làm nó mạnh lên: **một hệ thống
đạt 10/10 là một hệ thống mà lớp lỗi đã được đóng ở mức *cấu trúc*, không
phải ở mức *thực thể*.**

## Quyết định

### 1. Hai module kernel làm chủ sở hữu duy nhất

`ai_assistant/path_policy.py` và `ai_assistant/exec_policy.py` là **kernel
leaf** — chỉ thư viện chuẩn, không thuộc lớp nào — nên mọi lớp đều phụ thuộc
được và chúng không phụ thuộc ai:

* `path_policy`: `canonical_path`, `resolve_within`, `is_blocked_dir`,
  `is_secret_name`;
* `exec_policy`: `resolve_executable`, `executable_candidates`.

Ba nơi từng tự giải chương trình (`sandbox`, `vcs_tools`, `lsp_client`) nay
dùng chung một quy tắc; bốn nơi từng tự so đường dẫn (`path_utils`,
`lsp_client`, `chat_routes`, và ba walker) nay dùng chung một quy tắc.

### 2. So sánh phân biệt hoa thường — *một quy tắc, mọi nơi*

`is_blocked_dir` và `is_secret_name` đều gấp về lowercase trước khi so.
Windows filesystem không phân biệt hoa thường, nên `.GIT` **chính là** `.git`
và `PYPROJECT.TOML` **chính là** `pyproject.toml`.

Bộ test viết theo **dạng tính chất**: nó duyệt chính tập quy tắc, nên thêm
một tên vào tập tự động đòi hành vi không phân biệt hoa thường — một guard
*chống trôi* thay vì một danh sách ví dụ có thể lỗi thời.

### 3. So *mục tiêu thật*, không so *chính tả*

`resolve_within` kiểm tra **mục tiêu đã resolve** (symlink/junction đã theo),
và canonicalise **cả hai vế** trước khi so — nên dạng 8.3 và dạng dài của
cùng một thư mục không còn đối chiếu với nhau (ADR 0033), và một junction
trong project trỏ ra ngoài là một escape thật.

Ba cổng đọc trước đây dùng ba cách khác nhau; nay cùng một cách, và cách đó
**fail-closed**.

### 4. `shutil.which` bị cấm — bằng guard tĩnh

`tests/integration/test_executable_resolution_guard.py` quét AST toàn bộ
`src/ai_assistant/**` và **fail build** nếu bất kỳ module nào gọi
`shutil.which`. Đây là điểm mấu chốt của ADR này: lỗi thứ tư không còn có
thể xuất hiện.

Guard này phải là *tĩnh* vì lỗi nó ngăn chặn **không nhìn thấy được lúc
chạy**: mock trong test cũ trả về một tên *tương đối*, đúng hình dạng mà một
CWD-hit thật tạo ra, nên bộ test vẫn xanh trong khi control không hoạt động.

### 5. Fail closed thay vì degrade

`vcs_tools` không còn `resolved or "git"`. Khi `git` không có trong PATH, lệnh
bị **từ chối** — không phải chạy tiếp bằng một tên trần. Container image
đang ship (`python:3.11-slim`) **không có git**, nên ở đó fallback là đường
*mặc định*, và nó mở lại đúng cái hijack mà resolver sinh ra để chặn.

### 6. Cổng attachment

Route chat mở một đường dẫn do *client* đặt tên, và route giữ at-rest key.
Nay đường dẫn chỉ được mở khi nó resolve **bên trong một thư mục upload** mà
desktop client dùng để lưu attachment đã seal, **hoặc** khi file là một
envelope **đã seal** — chỉ nền tảng giữ khoá, nên seal chứng minh chính nền
tảng đã tạo ra file đó.

Các thư mục upload được suy ra **đúng như phía C++** (`AppConfig::uploadDir`):
`<data root>/Upload`, với data root là project root khi chạy từ source và
`<APP_DATA_DIR>/3D-Reconstruction` khi cài đặt. Tên thư mục được ghim bằng
`tests/unit/test_upload_dir_contract.py`, đọc thẳng từ mã nguồn C++.

### 7. Mỗi guard mới có một mutant

`MIN_MUTANTS` 80 → **88**. Tám mutant mới: tên chương trình mang đường dẫn,
quy tắc thư mục bị chặn, **hai** phép kiểm confinement (nửa *chính tả* và
nửa *mục tiêu thật* là hai quy tắc khác nhau), việc giải chương trình của
LSP, việc giải git fail-closed, cổng attachment, và khai báo *đã đăng ký
nhưng chưa kích hoạt*.

## Hậu quả

* Năm lỗi đã đóng và được ghim bằng test hồi quy *chạy thật*.
* **Thực thể thứ mười bốn của lớp lỗi không còn có thể xuất hiện** — đó là
  điểm khác biệt giữa "đã sửa mười ba lỗi" và "đã đóng một lớp".
* Hai module kernel mới: `path_policy` và `exec_policy`. Chúng là leaf, nên
  không phá vỡ contract layer nào (9/9 import contract vẫn giữ).
* `SCORECARD.md` ghi lại các control nào **chỉ** được kiểm bằng mock, và
  bây giờ thêm cột "đã đóng ở mức cấu trúc chưa".

### Gap còn lại, đã ghi nhận

* **Hai công cụ LSP chưa được supervisor cho phép.** `go_to_definition` và
  `find_references` được quảng bá cho mô hình và đăng ký trong factory, nhưng
  bảng `ALLOWED_TOOLS` của supervisor không có chúng, nên mọi lời gọi bị từ
  chối (*"No specialist policy exists for tool"*). Điều này **che** lỗi
  `shutil.which` của LSP client trong thời gian chờ. Nguyên tắc bất biến đã
  được ghi lại: **không bật công cụ LSP trước khi `lsp_client` dùng resolver
  chung** — nay nó đã dùng, nên đường mở lại đã an toàn, nhưng việc bật công
  cụ vẫn là một quyết định riêng.
* **ADR 0013 §6 mô tả sai trạng thái hiện tại** của hai công cụ đó.
* **Cổng attachment chỉ nhận đường dẫn tuyệt đối resolve vào thư mục upload.**
  Một bản triển khai khác (client lưu chỗ khác) sẽ phải khai báo thêm gốc.
