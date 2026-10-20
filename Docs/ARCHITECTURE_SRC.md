# Kiến trúc `src/` — định nghĩa và trạng thái "10/10"

> Tài liệu này là **hợp đồng đo được** cho kiến trúc phía C++ (`src/`, `tests/`).
> Nó không tuyên bố một điểm số: mỗi tiêu chí dưới đây đều gắn với một
> lệnh kiểm chứng và một ngưỡng, ai cũng chạy lại được.
>
> Lịch sử: phía Python (`AIAssistant/`) có thang điểm 10/10 riêng
> (`AIAssistant/Docs/adr/0021`, `0022`). Phía C++ **không** có thang điểm
> mang tính đánh giá — chỉ có bộ gate. Tài liệu này chuyển "10/10" thành
> một danh sách tiêu chí có thể chứng minh hoặc bác bỏ.

## 1. Bốn gate (đã có sẵn, bắt buộc xanh)

| # | Gate | Lệnh | Chứng minh điều gì |
|---|------|------|-------------------|
| 1 | Ranh giới kiến trúc | `cmake -DSOURCE_DIR=<repo> -P tests/CheckArchitectureBoundaries.cmake` | 7 nhóm luật: plugin↔service, service↔app, confinement của singleton, khớp heavy/light PCH, `src/utils` không heavy SDK, `waitForFinished()` phải có timeout |
| 2 | Bộ test | `ctest --test-dir build -C Release --output-on-failure` | Hành vi quan sát được của toàn bộ hợp đồng đã test |
| 3 | Coverage gate | `python tests/coverage_gate.py` | Mọi TU đo được có floor riêng; không TU nào "thoát" khỏi phép đo |
| 4 | Mutation check | `python tests/mutation_check.py` | Mọi quyết định bảo vệ (guard) đã được *ghim*: đảo quyết định → test phải đỏ |

Chi tiết cách chạy, biến môi trường và bảng số đo: `tests/README.md`.

## 2. Tiêu chí "10/10" (5 điều kiện, tất cả đều đo được)

| # | Tiêu chí | Điều kiện kiểm tra | Trạng thái |
|---|----------|--------------------|-----------|
| **C1** | 4 gate xanh | Cả 4 lệnh trong §1 exit 0 | ✅ |
| **C2** | **Không TU nào thoát khỏi phép đo** | Mọi `.cpp` trong `src/` (trừ 5 mục miễn trừ có lý do) xuất hiện trong báo cáo coverage | ✅ 95/95 TU |
| **C3** | Floor không bao giờ giảm | Mỗi file đo được có floor = giá trị đo; file mới phải được đưa vào danh sách | ✅ 95 file |
| **C4** | Nợ coverage được ghi nhận tường minh | File 0 % nằm trong bảng nợ (README) + in ra mỗi lần chạy | ✅ **0 TU** — mọi TU đã có dòng được thực thi |
| **C5** | Tài liệu khớp code | `AGENTS.md` + `tests/README.md` mô tả đúng trạng thái đo được | ✅ |

Điểm khác biệt so với trước đây: **C2** biến "tôi không biết chỗ nào chưa được đo" thành
một lỗi của gate. Trước đó 3 TU của app shell và toàn bộ lớp plugin (7 121 dòng)
không xuất hiện trong báo cáo — gate vẫn xanh.

### Số đo theo lớp (2026-10, MSVC 2022 / VS 17.12, Release)

| lớp | file .cpp | dòng đo được | tỉ lệ |
|-----|-----------|----------------|--------|
| `src/core` (kernel + guard) | 13 | 1 043 / 1 063 | 98,1 % |
| `src/app` (composition root) | 1 | 63 / 63 | 100 % |
| `src/services` | 37 | 1 859 / 3 906 | 47,6 % |
| `src/utils` (tiện ích dùng chung) | 9 | 374 / 447 | 83,7 % |
| `src/modules` (lớp plugin) | 35 | 796 / 4 277 | 18,6 % |

Trước đợt này: `src/utils` chỉ 10,8 %, `src/services` ~10 %, `src/modules`
**không xuất hiện trong báo cáo** (0 dòng được đo, vì plugin DLL nằm ngoài
`--modules`). Đợt này đã mở khoá cả ba:

* lớp plugin được đo qua vòng initialize/cleanup của từng plugin (mỗi plugin
  một tiến trình con) → 0 % → 18,6 %;
* lớp utils có test hợp đồng riêng → 10,8 % → 83,7 %;
* ba TU service cuối cùng chưa có test (`crosshairmanager`, `image2dloader`,
  `model3dloader`) đã có test → `src/services` 41,5 % → 46,9 %.

### Guard layer: không còn dòng thực thi nào chưa cover

Các dòng còn lại trong `src/core` rơi vào hai lớp đã được tài liệu hoá:
dòng **không có mã** (dấu `}` mà MSVC gán thành dòng, câu lệnh `continue`,
hàm bị optimizer inline) và nhánh **chỉ xảy ra khi môi trường hỏng**
(DPAPI hỏng, bản cài đặt thiếu file dịch, thư mục cấu hình không ghi được).
Không còn nhánh quyết định nào của guard layer chưa được kiểm chứng.

### Phát hiện được nhờ đo

Bộ gate không chỉ "giữ nguyên trạng" — nó đã bắt được hai lỗi nghiêm trọng
trong lớp plugin (xem `tests/README.md`):

* `AIProcessorPlugin::cleanup()` xóa một `QWidget` qua API `QDockWidget*` →
  **hỏng heap**, app chết khi thoát.
* `ViewerPlugin::cleanup()` rỗng → widget của plugin sống sót sau khi DLL
  được unload → **crash khi thoát app**.
* `MailMessageFormatter::decodeQuotedPrintable()` đưa cả thân mail qua byte
  Latin-1 rồi giải mã lại theo UTF-8 → **mọi ký tự tiếng Việt trong mail
  thuần bị thay bằng U+FFFD** (mail hiển thị rác).
* `SmtpMailer::sendMail()` kiểm tra `to.isEmpty()` trên *danh sách* người
  nhận, nhưng overload một-người-nhận bọc đối số rỗng vào list ⇒ **người
  nhận rỗng lọt qua và mail được gửi mà không có RCPT TO nào**.



## 3. Vì sao cần bỏ unity build ở một số target

Công cụ đo map dòng đã thực thi thông qua PDB. Nếu một file nguồn dùng chung
unity translation unit với file khác, MSVC có thể **inline hết** → không sinh
mã cho file đó → PDB không nhắc tên file → exporter trả về class rỗng → file đó
đo được 0 % một cách âm thầm.

Đã gặp đúng lỗi này: `CrosshairGeometry.cpp` bị đo 0 % dù có test.

Cách sửa đã áp dụng:

* `tests/CMakeLists.txt`: `UNITY_BUILD OFF` cho các test có *môn học* là file
  nguồn (`TestCrosshairGeometry`, `TestMailMimeParser`, `TestCameraParamsParser`,
  `TestPointCloudFilter`, `TestAIYoloPostProcessor`).
* `CMakeLists.txt`: `UNITY_BUILD OFF` cho target ứng dụng `3D-Reconstruction`
  — nhờ vậy `Main.cpp` / `MainWindow.cpp` của app shell cũng hiện diện trong
  báo cáo. PCH vẫn hấp thụ chi phí header nặng (build lại: ~57 s).

`AppCore` và plugin vẫn dùng unity: tất cả TU của chúng đều xuất hiện trong
báo cáo, và C2 sẽ bắt khi một TU mới bị "nuốt".

## 4. Rủi ro threading của mail stack (đã giảm, còn ghi nhận)

`IMailService` là API blocking; guard `mail::enforceBackgroundThread()` cảnh
báo/assert khi lời gọi chạy trên GUI thread. Trong lúc viết test SMTP, guard
bắt được một điều không thấy trước đó: **cả global pool lẫn pool riêng đều
có thể thực thi task *inline* trên thread gọi** (task đầu tiên được worker,
các task sau chạy trên GUI thread dù pool rảnh và `maxThreadCount=2`).

Đã đóng:

* `blocking::run()` (`src/utils/BlockingCall.h`) — helper dùng **thread riêng
  luôn**, trả về `QFuture` để watcher nhận kết quả trên thread của caller.
  5 call site của mail stack chuyển sang dùng nó ⇒ cam kết "không bao giờ
  chạy trên GUI thread" là chắc chắn, không còn phụ thuộc hành vi của pool.
* `TestSmtpMailer` ghim lại hợp đồng đó: lời gọi blocking phải chạy ngoài
  GUI thread (assert trên thread id) và không được phát cảnh báo của guard.

Pool riêng (`mail::workerPool()`) đã bị gỡ: thử nghiệm cho thấy kể cả pool
riêng rảnh cũng có thể chạy task inline, nên nó không thay thế được
`QThread` tường minh.

## 5. Nợ còn lại (có đo, chưa xử lý)

Xem bảng "coverage debt" trong `tests/README.md` — mỗi dòng là một file chưa
được test, kèm lý do (GUI-only, VTK-only, network-only) và hướng xử lý.

Nguyên tắc: nợ phải **được đo và được ghi**, không được biến mất khỏi báo cáo.

### 4.1 Dữ liệu at-rest phía C++

Nguyên tắc tương tự: **đã niêm phong thì phải ghi ra, chưa niêm phong thì
cũng phải ghi** — không có trạng thái "không biết".

| Dữ liệu | Trạng thái | Cơ chế |
|----------|-----------|---------|
| Lịch sử chat (`chat_sessions_<user>.json`) | **Đã niêm phong** | `SecretVault` (DPAPI CurrentUser trên Windows); file cũ dạng plaintext vẫn đọc được và được seal lại ở lần `save()` kế tiếp |
| Khoá licence, mật khẩu, SMTP credential | **Đã niêm phong** | `SecretVault` (`enc1:`) |
| Ảnh đính kèm & thumbnail (`Upload/`, `Thumbnails/`) | **Đã niêm phong** | ADR 0031 — AES-256-GCM bằng **cùng key at-rest** mà dịch vụ AI dùng, nên cả hai tiến trình đọc được file đã mã hoá. Envelope khớp byte-for-byte giữa hai ngôn ngữ, ghim bằng **vector chéo**: mỗi bên giữ output của bên kia |
| **Trên POSIX** (`SecretVault`, `AtRestCipher`) | **KHÔNG có OS crypto** | Không có backend mã hoá cho POSIX trong app này: `encrypt()` ghi `plain:<giá trị>` kèm cảnh báo. Đây là giới hạn nền tảng **đã biết**, không phải bí mật: app C++ nhắm Windows (DPAPI), còn state của dịch vụ AI được niêm phong bằng `cryptography` trên **mọi** nền tảng |

Hai mutant trong `tests/mutation_check.py` ghim hành vi của lớp chat: bỏ bước
seal khi ghi, và bỏ bước unseal khi đọc. Cả hai đều phải làm đỏ
`TestChatSessionManager`.

Nợ "ảnh đính kèm" được ghi ở đây để không biến mất: nó cần một thay đổi ở
tầng hiển thị, không phải một bước bọc file.

## 6. Cách đo lại

```powershell
$env:PATH = "C:/Qt/6.9.3/msvc2022_64/bin;F:/PROJECTS/QT/vcpkg/installed/x64-windows/bin;$env:PATH"
cmake -B build -S . -G "Visual Studio 17 2022" `
  -DQT_QMAKE_EXECUTABLE=C:/Qt/6.9.3/msvc2022_64/bin/qmake.exe `
  -DBUILD_TESTING=ON `
  -DFETCHCONTENT_SOURCE_DIR_ONNXRUNTIME=F:/PROJECTS/QT/onnxruntime-1.20.1
cmake --build build --config Release --parallel 2
ctest --test-dir build -C Release --output-on-failure
python tests/coverage_gate.py
python tests/mutation_check.py
```

Bốn gate đều chạy offline (không tải ONNX Runtime 323 MB, không cài vcpkg).
