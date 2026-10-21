# ADR 0033 — Cổng ghi so trên *một* cách viết của đường dẫn

## Bối cảnh

ADR 0032 chốt: *"cây policy không ghi được — deny-list phủ các thư mục chứa
policy của chính nền tảng, khớp trên đường dẫn **đã resolve** để symlink không
lách được"*. Claim đó được ghim bằng
`tests/unit/test_sandbox.py::test_a_symlink_into_a_policy_tree_is_denied`.

Job `windows` của `Build-Python-ci.yml` đỏ:

```
FAIL: test_a_symlink_into_a_policy_tree_is_denied
AssertionError: 'denied by sandbox policy' not found in
'Write target escapes the project root.'
```

Đây **không** phải hồi quy của guard: symlink vẫn bị chặn. Nhưng nó là một
lỗi thật, vì nó nói sai *lý do* — và lý do là phần được ghim.

**Nguyên nhân.** Windows có **hai** cách viết cho cùng một thư mục: dạng dài
chuẩn (`C:\Users\runneradmin`) và dạng 8.3 (`C:\Users\RUNNER~1`). Runner của
GitHub đặt `TEMP=C:\Users\RUNNER~1\AppData\Local\Temp`, nên toàn bộ cây tạm
của test đi qua dạng 8.3.

`Path.resolve()` hỏi OS *đường dẫn cuối* của một path **đã tồn tại**
(`GetFinalPathNameByHandle`) → trả về dạng dài. Nhưng một symlink mà *target
chưa tồn tại* không mở được, nên nó được resolve bằng cách **đọc chính link** →
giữ nguyên dạng 8.3 mà link được tạo ra. `relative_to()` so hai *chuỗi*, nên
`C:\...\RUNNER~1\tmpX\AIAssistant\config\base.toml` không nằm trong
`C:\...\runneradmin\tmpX`: `ValueError` → lỗi *"escapes the project root"* đến
trước lỗi policy, và deny-list cây policy không bao giờ được hỏi tới.

Trên POSIX `realpath()` luôn trả về một đường dẫn canonical, nên lỗi này chỉ
xuất hiện trên **đúng nền tảng sản phẩm chạy** — và CI Linux vẫn xanh.

## Quyết định

1. **Một cách viết duy nhất cho cả hai vế.** `tools/sandbox._canonical_path()`
   resolve đường dẫn (theo link) rồi **canonicalize** nó: resolve *parent*
   (phần sâu nhất OS resolve được) và gắn lại *leaf*. Project root và write
   target dùng **cùng** cách viết, nên `relative_to()` so được hai thứ cùng
   nghĩa. Hàm này là nguồn duy nhất của "đường dẫn đã resolve" trong sandbox.
2. **Cổng ghi hỏi trên đường dẫn canonical.** Tên file lấy từ đường dẫn
   canonical, không phải từ đường dẫn đã cho: một link trỏ tới
   `pyproject.toml` bị chặn theo *tên thật* của file đích, không theo tên
   của link.
3. **`create_directory()` và `_path_escapes_workspace()` dùng cùng cách viết.**
   Cùng một lớp lỗi: một đường dẫn không hề ra khỏi workspace nhưng đọc ra
   như đã ra.
4. **Bản ghi vẫn ghi vào đường dẫn đã cho**, không theo link. `os.replace` thay
   *link*, không phải file đích — nên kể cả khi gate bị bỏ, một cây policy
   không bị ghi đè qua link.

## Hệ quả

- **Claim của ADR 0032 đúng trên cả Windows lẫn POSIX.** "Khớp trên đường
  dẫn đã resolve" giờ là một so sánh *cùng cách viết*, không phải hai chuỗi
  tình cờ giống nhau.
- **Ghim ở đúng nơi hỏng.** Test Windows-only
  `test_a_link_spelled_with_the_short_path_is_denied` dựng link với dạng 8.3
  một cách **cố ý** (`GetShortPathNameW`), nên nó đỏ nếu canonicalization bị
  gỡ — kể cả trên máy mà `TEMP` được viết đầy đủ. Job `windows` của CI là nơi
  ghim đó; job `test` (Linux) ghim phần *còn lại*: link vào cây policy bị chặn,
  và mutant mới (bỏ `path.resolve()`) bị giết ở đó.
- **Test không phụ thuộc đặc quyền tạo symlink.** Một Windows không có
  `SeCreateSymbolicLinkPrivilege` vẫn tạo được *junction*, và junction được
  resolve y hệt (nó giữ nguyên cách viết khi được tạo ra, và OS không mở
  được nó khi target chưa tồn tại). Nên case trên chạy được trên mọi máy
  Windows — kể cả máy của người phát triển — và mutant bị giết ở *mọi* nơi,
  không chỉ ở CI.
- **POSIX không có mutant riêng cho canonicalization** — `realpath()` ở POSIX
  đã canonical, nên bước đó là no-op và *không quan sát được* trên nền tảng
  đó. Điều này được ghi ra thay vì giấu: mutant của quyết định là **"target
  phải được resolve"**, còn canonicalization được ghim bằng test Windows.
- **Số đo thay đổi:** `MIN_MUTANTS` 74 → 75, và `Docs/SCORECARD.md` ghi lại
  số đo mới.

## Không giải quyết

- **Cổng ghi vẫn không chống được TOCTOU giữa lúc kiểm tra và lúc ghi.** Đây
  là giới hạn của mọi kiểm tra theo đường dẫn trên một hệ điều hành không có
  `openat(O_NOFOLLOW)`; việc thu hẹp nó cần một quyết định riêng.
- **Symlink tới thư mục không tồn tại** vẫn giữ dạng 8.3 trong `resolve()`, nên
  nó bị từ chối bằng thông điệp *"escapes the project root"* thay vì thông điệp
  policy. Hành vi là **fail-closed** (từ chối), chỉ là lý do được báo sai; sửa
  nó cần một bộ canonicalization đệ quy, chưa đáng giá.
- **Cổng đọc** (`tools/builtin/path_utils.agent_safe_path`) so hai đường dẫn
  `realpath` bằng `commonpath` và có cùng đặc điểm: một link mà target chưa
  tồn tại có thể bị từ chối dưới lý do sai. Nó **fail-closed** (đọc bị từ
  chối), nên không có lỗ hổng; sửa nó là một thay đổi riêng, không thuộc
  cổng ghi mà CI đang đỏ.
