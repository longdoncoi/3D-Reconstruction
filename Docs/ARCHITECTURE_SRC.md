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
| 1 | Ranh giới kiến trúc | `cmake -DSOURCE_DIR=<repo> -P tests/CheckArchitectureBoundaries.cmake` | 12 nhóm luật: plugin↔service, service↔app, confinement của singleton, khớp heavy/light PCH, `src/utils` không heavy SDK, `waitForFinished()` phải có timeout, event filter phải gỡ được, không dò cửa sổ shell, blocking I/O không qua thread pool, include path của plugin không chứa `src/core`, **mỗi lời gọi inference phải đi qua exception guard** |
| 2 | Bộ test | `ctest --test-dir build -C Release --output-on-failure` | Hành vi quan sát được của toàn bộ hợp đồng đã test (42 test executable) |
| 3 | Coverage gate | `python tests/coverage_gate.py` | Mọi TU đo được có floor riêng; không TU nào "thoát" khỏi phép đo; **floor không được hạ** |
| 4 | Mutation check | `python tests/mutation_check.py` | Mọi quyết định bảo vệ (guard) đã được *ghim*: đảo quyết định → test phải đỏ (63 mutant) |

**Tiền kiểm của gate 4 (đóng 2026-10-22).** Gate 4 từng **chết âm thầm**: một mutant trỏ vào `src/core/ServiceRegistry.h` sau khi file đã chuyển sang `src/include/`, nên script abort bằng `FileNotFoundError` *sau* khi pre-flight đã xanh — trông như gate đã chạy và không báo gì. Ba commit mang theo mutant hỏng, và tuyên bố "52/52 mutant ✅" chưa từng được kiểm chứng.

Nay gate 4 **tự kiểm mục tiêu của mình TRƯỚC khi rebuild bất cứ thứ gì**:

* mỗi mutant phải trỏ vào một file **tồn tại**;
* mỗi anchor phải xuất hiện **đúng một lần**;
* mỗi file header-only mang quyết định phải có **ít nhất một mutant** (xem §4.9).

Sai → gate **đỏ ngay lập tức** với danh sách mutant hỏng, thay vì im lặng bỏ sót.

Chi tiết cách chạy, biến môi trường và bảng số đo: `tests/README.md`.

## 2. Tiêu chí "10/10" — mỗi tiêu chí phải trả lời được câu hỏi phản ví dụ

> **Nguyên tắc (sửa 2026-10-22).** Trước đây §2 **tự chấm điểm**: bốn gate
> định nghĩa tiêu chí, rồi bốn gate đó tự báo "xanh". Đó là điểm tự tham
> chiếu — một gate chết làm tiêu chí của chính nó biến thành *"không có
> tiêu chí"*. Và nó đã xảy ra: gate 4 chết 3 commit liên tiếp mà §2 vẫn ghi
> "✅ 52/52 mutant".
>
> Nay mỗi tiêu chí có **ba phần**: lệnh kiểm chứng, ngưỡng, và **câu hỏi
> phản ví dụ**: *"nếu gate này xanh, lớp lỗi nào vẫn lọt qua?"*. Một tiêu chí
> chỉ đạt khi câu trả lời là **"không có gì"** — và câu trả lời phải được
> **kiểm chứng bằng một mutant hoặc một test**, không phải bằng lời hứa.

| # | Tiêu chí | Lệnh / ngưỡng | Lớp lỗi còn lọt qua? | Bằng chứng chống lọt |
|---|----------|---------------|----------------------|----------------------|
| **C1** | 4 gate xanh | §1 — cả 4 lệnh exit 0 | *Một gate chết **im lặng*** | Tiền kiểm tự-mục tiêu của gate 4 (§1): mỗi mutant phải trỏ vào file tồn tại với anchor đúng một lần → sai là **đỏ ngay** |
| **C2** | Không TU nào thoát khỏi phép đo | Mọi `.cpp` trong `src/` (trừ 5 mục miễn trừ có lý do) xuất hiện trong báo cáo coverage | *Code **header-only*** — không phải TU nên gate 3 không thấy | §4.9 — `PINNED_HEADERS`: mọi header mang quyết định đều phải có ≥ 1 mutant, gate 4 từ chối khi thiếu |
| **C3** | Floor không bao giờ giảm | Mỗi file đo được có floor = giá trị đo **và** ≥ baseline đã commit | *Một quyết định quan trọng nằm trong file có coverage cao* | coverage đo **dòng**, không đo **quyết định** → gate 4 ghim quyết định |
| **C4** | Nợ coverage được ghi nhận tường minh | File 0 % nằm trong bảng nợ (README) + in ra mỗi lần chạy | *Một lớp lỗi sống dưới một guard đã ghim ở **một thể hiện*** | §4.8 + §4.10 + §4.12 — mutant theo **lớp** quyết định, không theo một hàm |
| **C5** | Tài liệu khớp code | `AGENTS.md` + `tests/README.md` + file này mô tả đúng trạng thái đo được | *Tài liệu tự đánh giá mình* | Số liệu trong tài liệu **lấy từ output của gate**, không viết tay (§6) |
| **C6** | **Không lớp lỗi nào sống sót dưới một guard đã có** | Mỗi guard phải ghim **cả lớp** quyết định | — | §4.8 (khoá pipeline), §4.10 (exception barrier) và §4.12 (inference ra khỏi GUI thread) — 11 mutant mới, mỗi cái đảo một quyết định khác nhau trong cùng lớp |

### 2.1 Vì sao C6 là tiêu chí mới — và nó bắt được gì

Ba lần trong lịch sử repo này, một guard được thêm vào **một thể hiện** của
một lớp quyết định, còn anh em của nó thì không:

| lớp quyết định | được ghim | **không** được ghim → lỗi sống |
|------------------|-----------|------------------------------|
| khoá pipeline gây freeze GUI | `getImageList()` | 7 phương thức còn lại của `IReconstructionService` |
| exception từ ONNX giết app | 3 hàm **load** của `AIProcessor` | 4 hàm **inference** + `VideoTrackerThread::run()` |
| blocking I/O ngoài GUI thread | mail stack | AI inference (chạy đồng bộ trên GUI thread) — đã đóng ở §4.12 |

Cả ba đều **lọt qua toàn bộ bốn gate**: coverage đo dòng (các dòng đó đều
được thực thi), mutation ghim đúng *một* hàm, architecture guard là regex
trên mã nguồn. Gate không sai — chúng chỉ **không thấy** lớp.

C6 nói rõ: một guard chỉ được coi là đóng khi nó ghim **cả lớp**. §4.8, §4.10
và §4.12 là ba lần đóng lớp đó, và mỗi lần kèm **một mutant cho mỗi thể hiện
còn lại** — không phải một mutant đại diện.

### Số đo theo lớp (2026-10, MSVC 2022 / VS 17.12, Release)

| lớp | file .cpp | dòng đo được | tỉ lệ |
|-----|-----------|----------------|--------|
| `src/core` (kernel + guard) | 13 | 1 241 / 1 256 | 98,8 % |
| `src/app` (composition root) | 1 | 69 / 69 | 100 % |
| `src/services` | 40 | 2 404 / 4 206 | 57,2 % |
| `src/utils` (tiện ích dùng chung) | 9 | 390 / 458 | 85,2 % |
| `src/modules` (lớp plugin) | 35 | 2 404 / 4 399 | 54,6 % |

Đợt đóng **hai lớp lỗi** (§4.8 và §4.10) làm coverage **tăng** ở đúng
những file có guard mới — vì lỗi cũ nằm ở những dòng **chưa hề được thực
thi** (đường inference không có rào, đường truy vấn không đọc state đã
publish):

| file | trước | sau | vì sao |
|------|-------|-----|---------|
| `reconstructionservice.cpp` | 68,8 % (108/157) | **73,8 %** (118/160) | 7 phương thức chuyển sang state đã publish; worker publish kết quả |
| `reconstructionpipeline.cpp` | 49,5 % (94/190) | **52,5 %** (105/200) | `buildSnapshot()` (chuyển đổi ABI-neutral) có test riêng, rẻ |
| `videotrackerthread.cpp` | 50,8 % (30/59) | **73,1 %** (49/67) | rào exception của thread entry point + test có clip tổng hợp |
| `aiinferenceguard.cpp` | *(file mới)* | **71,4 %** (25/35) | một chỗ chứa exception cho **mọi** entry point inference |
| `tracksfm.cpp` | 93,2 % (55/59) | **96,6 %** | — |
| `aiservice.cpp` | 64,1 % (25/39) | 64,1 % (25/39) | không đổi (đường *load* đã có rào từ trước) |

Đợt này mở khoá lớp plugin: **18,6 % → 51,8 %**. Ba nguyên nhân:

* `TestPluginSmoke` trả về **panel thật** cho `getTabPanel()` (trước đó trả
  `nullptr`, nên toàn bộ đường ribbon không bao giờ chạy trong CI) và
  `addDockWidget`/`deleteDockWidget` thao tác lên `QMainWindow` thật;
* `TestPluginUi` biên dịch thêm `AITrainDockWidget.cpp`;
* `TestUserAuthDialogs` phủ luồng reset mật khẩu và lớp overlay nền.

Đợt §4.12 đo lại: `aiprocessorplugin.cpp` **34,4 % → 44,1 %** (144/419 →
193/438) — `TestPluginUi` giờ biên dịch cả `AIProcessorPlugin.cpp` (và
`VideoTrackerThread.cpp`) rồi gọi `initialize()`, `requestInference()`,
`finishInference()` thật; phần còn lại là dock train, cửa sổ chart và đường
tracking cần tương tác. `moderndialog.cpp` tăng 50 → 55 dòng vì test lỗi
inference dựng và đóng **hộp thoại cảnh báo thật**, kéo `src/utils`
**84,1 % → 85,2 %**. Floor `aiprocessorplugin.cpp` nâng **34 → 44** cùng
commit.

Đợt trước mở khoá lớp **service**: **49,6 % → 55,2 %**. Ba file được phủ bằng
test thật (không phải mock), vì cả ba là logic thuần CPU-side nên test được
không cần model, GPU hay server:

| file | trước | sau | vì sao test được |
|------|-------|-----|------------------|
| `reconstructionpostprocessor.cpp` | 20,0 % | **96,3 %** | densify (KNN + RNG cố định), Poisson meshing (bất biến "mặt không trỏ tới đỉnh đã bị trim"), texturing (best-view, nền đen, ghi file PLY) |
| `aiserverprocesscontroller.cpp` | 34,6 % | **94,7 %** | vòng đời tiến trình thật: nạp, start hoãn, kill dự phòng, thoát ngoài dự kiến, thu gom trong destructor |
| `reconstructionservice.cpp` | 60,9 % | **68,8 %** | khoá state publish tách khỏi khoá pipeline + hợp đồng "đọc được khi đang reconstruct" |

Cùng đợt, `aimodelloadservice.cpp` giữ **100 %** sau khi thêm nhánh bắt
exception (trước đó một exception từ ONNX làm kẹt `m_running` và không phát
tín hiệu kết thúc).

Mỗi floor được nâng **cùng commit với lần đo** (`coverage_gate.py
--update-baseline`): `reconstructionpostprocessor` 20 → 96,
`aiserverprocesscontroller` 34 → 94, `reconstructionservice` 60 → 68.

Đợt sau đó bật **Git LFS trong CI C++** (`lfs: true`) để có model ONNX thật,
và phủ thêm hai file bằng test thật:

| file | trước | sau | vì sao test được |
|------|-------|-----|------------------|
| `aiservice.cpp` | 10,3 % | **64,1 %** | `TestAIService` nạp model YOLO thật (`AIComputerVision/Models/*.onnx`) và chạy inference trên CPU |
| `aiprocessor.cpp` | 9,5 % | **17,7 %** | các đường từ chối model + khối `catch` mới |
| `chatmessagerenderer.cpp` | 5,6 % | **39,0 %** | `TestPluginUi` render lịch sử chat (user/assistant/agent trace) và kiểm HTML |

Hai phát hiện từ đợt này:

* **Lỗi bền vững thật**: `AIProcessor` chỉ bắt `Ort::Exception`. Một
  `std::bad_alloc` (máy hết RAM khi nạp session 227 MB) **thoát ra ngoài và
  giết app** — lời gọi đến từ UI thread. Nay bắt thêm `std::exception` và
  `...`: nạp model hỏng trả về "chưa nạp" thay vì làm sập ứng dụng.
* **Công cụ đo không host được session ONNX**: OpenCppCoverage nhân bộ nhớ
  của mọi phép cấp phát ⇒ tạo session ném `std::bad_alloc`. Vì vậy
  coverage gate đặt `AI_SERVICE_NO_MODELS=1` cho `TestAIService`: nó đo được
  **dòng của chính service** (chuyển tiếp, cờ sẵn sàng, từ chối model
  thiếu), còn **inference thật** vẫn được chứng minh ở lần CTest thường (không
  có exporter). Hai lớp bằng chứng, mỗi lớp ở đúng chỗ nó đúng.

### Guard layer: những nhánh còn lại đều đã được gọi tên

Các dòng còn lại trong `src/core` rơi vào hai lớp đã được tài liệu hoá:
dòng **không có mã** (dấu `}` mà MSVC gán thành dòng, câu lệnh `continue`,
hàm bị optimizer inline) và nhánh **chỉ xảy ra khi môi trường hỏng**
(DPAPI hỏng, bản cài đặt thiếu file dịch, thư mục cấu hình không ghi được).
Ngoài ra có một nhánh *nền tảng* đã được ghi rõ: `SecretVault` ghi
`plain:<value>` **chỉ** trên nền tảng không có backend OS — nền tảng có
backend thì **fail closed** (xem §4).

### Phát hiện được nhờ đo

Bộ gate không chỉ "giữ nguyên trạng" — nó đã bắt được các lỗi nghiêm trọng
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
* `src/core/AtRestCipher.cpp`: hàm đọc file khoá được đánh dấu
  `ATREST_NOINLINE` vì unity build gấp nó vào `loadKey()`.
* `src/core/CryptoUtils.cpp`: `verifyPassword()` được đánh dấu
  `CRYPTO_NOINLINE`, và hai nhánh (định dạng mới / legacy) được giữ tách
  biệt — nếu không, MSVC gộp hai lệnh `return` giống nhau và dòng so sánh
  của định dạng mới bị báo là "chưa bao giờ thực thi".

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
`QThread` tường minh. **Luật kiến trúc mới**: `src/modules/**` không được
dùng `QtConcurrent::run` (guard chặn).

### 4.1 Đồng bộ lớp service: state publish tách khỏi khoá pipeline (đã đóng)

`ReconstructionService` dùng **một** `shared_mutex` cho cả pipeline lẫn state
mà UI đọc (`m_imageList`, `m_snapshot`). Worker giữ khoá đó **suốt** lần
reconstruct, nên `getImageList()` — được `ReconstructionPlugin` gọi ngay sau
khi bắt đầu chạy — phải chờ hết cả quá trình: **UI đóng băng**, trái với
chính comment "a query must not wait for it" trong file. `pointCount()` /
`hasResult()` còn kiểm `m_running` **ngoài** khoá (check-then-act).

Đã đóng: state publish có khoá riêng (`m_publishedMutex`), tách hẳn khỏi
`m_mutex`; các truy vấn lấy khoá **trước** rồi mới kiểm `m_running`.
`TestReconstructionPipeline::serviceImageListStaysReadableDuringReconstruction`
ghim hợp đồng (đo thời gian trả về khi reconstruction còn chạy), và mutant
"getImageList() waits on the pipeline lock" phải làm đỏ test đó.

`AIModelLoadService` không bọc `try/catch` quanh lời gọi loader: một exception
từ ONNX để `m_running` kẹt `true` và **không phát** `finished`/`cancelled` —
mọi lần load sau bị từ chối và consumer tưởng load còn chạy mãi. Đã đóng:
worker bắt `std::exception` và `...`, luôn reset `m_running`, luôn phát tín
hiệu kết thúc; ba test mới ghim, và mutant "a throwing loader is reported as
success" phải đỏ. Join trong destructor **cố ý không biên**: worker gọi vào
`IAIService` mà AppShell chỉ huỷ *sau* service này, nên dừng chờ sớm sẽ là
use-after-free chứ không phải shutdown nhanh.

### 4.2 `AIServerProcessController::stop()` không chặn GUI thread (đã đóng)

`stop()` gọi `terminate()` rồi `waitForFinished(1000)`, và nếu hết hạn thì
`kill()` + `waitForFinished(STOP_SERVER_TIMEOUT_MS = 3000)` — **trên thread gọi**,
tức là GUI thread. Trên Windows một tiến trình console Python bỏ qua
`WM_CLOSE`, nên `terminate()` luôn hết hạn: mỗi lần `startServer()` /
`restartServer()` / `stopServer()` đứng cửa sổ ~1 s.

Đã đóng theo hướng *không chặn*:

* `stop()` chỉ `terminate()` + **bật timer kill dự phòng** (`STOP_SERVER_TIMEOUT_MS`);
  tiến trình được thu gom trong `onFinished()` của event loop ⇒ không còn
  `waitForFinished()` nào trên GUI thread.
* `start()` **hoãn** nếu tiến trình cũ chưa chết (`m_pendingModelIndex`), nên
  chuỗi `stop(); start()` vẫn chạy đúng mà không phải chờ.
* Một stop **do ta yêu cầu** không được báo như crash: cờ `m_stopping` chặn
  `processFinished` (nếu không, `AIAssistant::onServerFinished()` sẽ chạy
  recovery và tạo vòng restart).
* Destructor là **chỗ duy nhất** còn chờ có biên — app đang thoát, và
  `QProcess` con không được sống lâu hơn controller.

Ghim bằng `TestAIServerProcessController` (test executable tự chạy lại làm
tiến trình giả, không cần server thật) + 3 mutant: start hoãn bị rơi, stop
bị báo nhầm là crash, timer kill không được bật.

### 4.3 ABI gate tự động: hash nội dung bề mặt interface (đã đóng)

`IPlugin::kApiVersion` là **số do người viết phải nhớ bump**. Sửa một
`I*.h` mà quên bump ⇒ host vẫn nạp plugin cũ với ABI khác, và lỗi hiện ra
ở chỗ không liên quan (crash, đọc sai vùng nhớ) thay vì lúc nạp.

Đã đóng: `IPlugin::kAbiFingerprint` nay **trộn** fingerprint toolchain với
**hash nội dung của bề mặt interface plugin-facing** — không phải danh sách
`I*.h` mà là *bao đóng* include giao diến (transitive quoted-include closure)
của: mọi `src/core/I*.h`, core glue được allow-list, lớp tiện ích dùng chung
`src/utils/*.h` và `AgentActionManifest.h`. Bao đóng là cần thiết: plugin thấy
`IReconstructionService.h` **và** DTO nó include (`PointCloudData.h`) — hash
chỉ `I*.h` sẽ bỏ sót thay đổi của DTO đó.

Cơ chế:

* `cmake/PluginInterface.cmake` — **một** định nghĩa duy nhất của "bề mặt
  plugin-facing" (dùng chung cho generator và guard).
* `cmake/InterfaceAbi.cmake` — hash ổn định (tên file + nội dung file, không
  phụ thuộc đường dẫn tuyệt đối ⇒ cùng giá trị trên mọi máy).
* `cmake/GenerateInterfaceAbi.cmake` + custom command phụ thuộc **mọi** file
  của bề mặt ⇒ hash **không bao giờ cũ**.
* `TestInterfaceAbi` (CTest) ghim hai tính chất: **freshness** (hash sinh ra
  phải bằng hash tính lại từ file hiện tại) và **sensitivity** (đổi nội dung
  bất kỳ header nào ⇒ hash phải đổi).
* `TestPluginHost::abiFingerprintCarriesTheInterfaceHash` ghim phần còn lại:
  fingerprint thật sự chứa hash đó; mutant "the ABI fingerprint ignores the
  plugin-facing interface hash" phải làm đỏ.

Bằng chứng đầu-cuối: `TestPluginSmoke` nạp **DLL plugin thật** — nó đi qua
cổng ABI với hash sinh ra ở AppCore, nên nếu hash lệch giữa AppCore và
plugin thì test đỏ.

### 4.4 `ServiceRegistry`: đăng ký trùng bị TỪ CHỐI (đã đóng)

Đăng ký trùng cùng một interface trước đây chỉ `qWarning` rồi **ghi đè** —
một plugin (hoặc một lỗi wiring trong composition root) có thể âm thầm
cướp service mà composition root đã cài. Nay lần **đầu tiên thắng**: lần
đăng ký sau bị từ chối, service cũ giữ nguyên. Mutant "duplicate service
registration replaces the existing service" phải làm đỏ `TestServiceRegistry`.

### 4.5 Ranh giới plugin: từ regex của guard → **compiler** (đã đóng)

Trước đây ranh giới "plugin chỉ được include interface" được **regex** của
guard kiểm (rule 1). Regex soi *mã nguồn*: một include bắc cầu, một đường dẫn
tương đối, hay một dòng `target_include_directories` làm cho cả ranh giới bị
lỡ. Tệ hơn: `AppCore` **PUBLIC** toàn bộ `src/core`, nên plugin *biên dịch
được* `UserManager.h`, `SecretVault.h`, `AppConfig.h`, `PluginHost.h` — chỉ là
guard sẽ báo lỗi khi ai đó viết dòng include đó.

Đã đóng bằng **build graph**:

| | trước | sau |
|---|--------|-----|
| bề mặt công khai của AppCore | `src/core` + `src/utils` | **`src/include`** + `src/utils` |
| `src/core` (kernel cụ thể) | PUBLIC | **PRIVATE** |
| plugin include `UserManager.h` | biên dịch được (guard cảnh báo) | **không biên dịch được** |

29 header plugin-facing được chuyển sang `src/include/`: các interface `I*.h`,
core glue được allow-list (`Global.h`, `AppConstants.h`, `AIMode.h`,
`SignalBus.h`, `ServiceRegistry.h`) và các DTO chúng phơi ra
(`PointCloudData.h`, `UIConstants.h`, …). `src/core` giữ lại **chỉ** phần
cụ thể: `UserManager`, `SecretVault`, `AppConfig`, `PluginHost`, `CryptoUtils`,
`OtpStore`, `LicenseKeyLedger`, `UserAccountStore`, `UserSettingsStore`,
`LanguageManager`, `AtRestCipher`, `PluginDependencyResolver`.

Bằng chứng: thêm `#include "UserManager.h"` vào `ViewerPlugin.cpp` ⇒
`error C1083: Cannot open include file: 'UserManager.h'` — **compiler** từ chối,
không cần guard.

Và để một dòng include path không thể vô hiệu hoá tất cả, **rule 11** của
guard kiểm *include path* của từng plugin: không được có `src/core`. Mutant
"planted: a plugin's include path hands it the concrete kernel headers" phải
làm đỏ `TestArchitectureBoundaries`.

Hệ quả phụ đáng kể: bề mặt ABI của plugin **bằng đúng một thư mục**
(`src/include` + `src/utils`), nên hash ABI (giờ gồm 39 file) và ranh giới đều
lấy từ cùng một nguồn sự thật.

### 4.6 `IUserManager` là AGGREGATE, không phải hợp đồng để phụ thuộc (đã đóng)

`IUserManager` trước đây gộp 6 năng lực vào một interface 167 dòng: đăng
nhập/phiên, mật khẩu + OTP, hồ sơ, quản trị tài khoản, credential SMTP,
vòng đời cấu hình (cộng `IUserPreferences` + `ILicenseService` đã tách từ
trước). Một plugin chỉ cần "tôi là ai" + "tùy chọn của tôi" cũng phải kéo
theo cả bề mặt quản trị tài khoản.

Đã đóng theo Interface Segregation:

| interface | năng lực | ai dùng |
|-----------|-----------|---------|
| `IAuthService` | đăng nhập/đăng xuất, danh tính hiện tại, quyền admin, mật khẩu + OTP, vòng đời store | 5 plugin (AI Assistant, Chat, Reconstruction, Mail) |
| `IUserProfileService` | hồ sơ của tài khoản hiện tại | `AvatarCropperDialog` |
| `IUserAdminService` | quản trị tài khoản | `AdminUserManagerDialog` |
| `ISmtpCredentialService` | credential gửi mail | `AdminUserManagerDialog` |
| `IUserPreferences` | tùy chọn + secret theo user | 5 plugin |
| `ILicenseService` | tra cứu/kích hoạt licence | `AboutDialog`, `LicenseActivationDialog` |

`IUserManager` giữ lại làm **aggregate** (kế thừa cả sáu) để composition
root đăng ký một implementation dưới một key; mỗi interface hẹp có **ServiceKey
riêng** và composition root đăng ký **cùng một instance** `UserManager` dưới
tất cả. `UserTypes.h` chứa các kiểu giá trị dùng chung (`UserInfo`,
`AuthError`, `UserRole`) để các interface hẹp không phải include lẫn nhau.

Bằng chứng: 5 plugin không còn `#include "IUserManager.h"`; chúng resolve
đúng hai interface mà chúng dùng. Mutant "the composition root does not
register the segregated IAuthService" phải làm đỏ `TestAppShell`.

### 4.7 Tin nhắn chat có kiểu, không phải `QJsonObject` (đã đóng)

`ChatSession::messages` là `QList<QJsonObject>` với magic key (`"role"`,
`"content"`, `"attachments"`, `"timestamp"`). Vai trò là **chuỗi tự do**, nên
một vai trò chỉ-UI (`assistant_agent`) từng lọt vào request và quay về
**HTTP 422** — lỗi chỉ lộ ra ở tầng HTTP, cách xa nguyên nhân.

Đã đóng: `ChatMessage` (`src/include/ChatMessage.h`) — vai trò là enum đóng
(`ChatRole::User/Assistant/System/AgentTrace`), attachment là danh sách đường
dẫn, và `isModelRole()` là **quyết định duy nhất** về cái gì được phép tới
model. Trạng thái không hợp lệ không còn viết được ra.

Định dạng **wire và đĩa giữ nguyên** (`toJson()`/`fromJson()` ánh xạ đúng
shape cũ), nên lịch sử chat đã niêm phong vẫn mở được và dịch vụ Python vẫn
nhận đúng payload. `ChatSessionStore` chỉ là chỗ gọi mapping.

Ghim: `TestChatSessionManager` (round-trip qua định dạng wire + hợp đồng
"agent trace không phải model role") và mutant "the UI-only agent trace is
accepted as a model role" phải đỏ.

### 4.8 Đóng **cả lớp** "khoá pipeline gây freeze GUI" (đã đóng 2026-10-22)

§4.1 đã tách khoá *published* khỏi khoá *pipeline* — nhưng chỉ cho **một**
phương thức. Worker giữ `m_mutex` (khoá pipeline) **độc quyền suốt lần
reconstruct**, nên 7 phương thức còn lại của `IReconstructionService` vẫn lấy
khoá đó và **đóng băng GUI thread tới hết lần chạy**:

| phương thức | trước | sau |
|-------------|-------|-----|
| `getImageList()` | `m_mutex` → đã sửa ở §4.1 | `m_publishedMutex` ✅ |
| `setImages()` | `m_mutex` (chặn) | `try_to_lock` → **từ chối** |
| `loadCameraParams()` | `m_mutex` (chặn) | `try_to_lock` → **từ chối** |
| `configure()` | `m_mutex` (chặn) | `try_to_lock` → **từ chối** |
| `processPointCloud()` | `m_mutex` (chặn) | `try_to_lock` → **từ chối** |
| `pointCloudSnapshot()` | `m_mutex` (chặn) | chỉ đọc state đã publish ✅ |
| `pointCount()` | `m_mutex` (chặn) | chỉ đọc state đã publish ✅ |
| `hasResult()` | `m_mutex` (chặn) | chỉ đọc state đã publish ✅ |

Hai nguyên tắc mới:

* **Truy vấn chỉ đọc state đã publish.** Worker **publish** snapshot + số điểm
  ngay khi lần chạy kết thúc (`publishResult()`, trong khi vẫn giữ khoá
  pipeline). UI đọc bản publish ⇒ **không phương thức nào của UI phải chạm
  vào pipeline**, kể cả khi đang reconstruct. (Việc copy point cloud cũng
  chuyển khỏi GUI thread sang worker thread.)
* **Phép chuyển đổi nằm cạnh dữ liệu nó chuyển.** `ReconstructionPipeline::
  buildSnapshot()` dựng bản ABI-neutral (BGR→RGB swap nằm ở đó) — nên nó
  được đo bởi bộ test *nhanh* của pipeline, thay vì chỉ bởi một lần
  reconstruct hoàn chỉnh (hậu kỳ nặng, không kịp dưới exporter).
* **Mutator từ chối thay vì chờ.** `try_to_lock` + kiểm tra `m_running` **dưới
  khoá**: nếu pipeline đang bận, thay đổi bị **từ chối** (kèm `qWarning`),
  không phải chờ. Ý nghĩa giữ nguyên so với `if (m_running) return;` cũ —
  chỉ khác ở chỗ nó **không chờ**.

Hai điểm phụ đáng lưu ý:

* `reconstruct()` reset snapshot **và** số điểm đã publish, nên trong lúc
  chạy, `pointCount()` trả 0 và `hasResult()` false — UI không phải suy đoán.
* Thứ tự khoá bất biến vẫn là `m_mutex` → `m_publishedMutex`, không bao giờ
  ngược lại.

Ghim: **4 mutant** (mỗi cái đảo một phương thức, không phải một đại diện) +
3 test mới trong `TestReconstructionService`
(`serviceQueriesStayNonBlockingDuringReconstruction`,
`serviceMutatorsDoNotWaitForTheRunningPipeline`,
`servicePublishesNothingWhileTheRunIsInFlight`).

Bộ test service nằm trong **một executable riêng**
(`tests/TestReconstructionService.cpp`): trường hợp "lần chạy đã hoàn tất"
cần một lần reconstruct đầy đủ, mà hậu kỳ nặng để lại vài gigabyte trong
heap — chạy thứ hai trong cùng tiến trình sẽ vượt quá thời hạn mỗi hàm. Một
test chỉ xanh trên máy nhanh thì không phải là test.

### 4.9 Điểm mù của code **header-only** (đã đóng 2026-10-22)

Gate 3 liệt kê các **translation unit `.cpp`** — nên một header **hoàn toàn
vô hình** với nó. Gate 4 thì ghim theo *file*, nên có thể ghim một header —
nhưng không ai bắt buộc phải làm thế. Hệ quả: `src/utils/BlockingCall.h`
(primitive blocking-I/O duy nhất của app, dùng bởi 5 luồng UI ở 2 plugin)
**không có mutant nào** và **không có floor nào**.

Đã đóng theo hai lớp:

1. **Danh sách `PINNED_HEADERS`** trong `tests/mutation_check.py`: mọi file
   header-only mang quyết định phải có **≥ 1 mutant**; thiếu ⇒ gate 4 **đỏ**
   với danh sách file thiếu. Danh sách hiện tại:
   `src/include/IPlugin.h`, `src/include/ServiceRegistry.h`,
   `src/services/mail/MailThreadGuard.h`, `src/utils/BlockingCall.h`.
2. **Gate 4 tự kiểm mục tiêu của mình** trước khi rebuild bất cứ thứ gì:
   mỗi mutant phải trỏ vào file **tồn tại** với anchor **đúng một lần**.
   Sai ⇒ **đỏ ngay** với danh sách mutant hỏng.

Cái thứ hai là thứ đáng chú ý: trước đó một mutant trỏ vào
`src/core/ServiceRegistry.h` sau khi file đã chuyển sang `src/include/` làm
script abort bằng `FileNotFoundError` **sau** khi pre-flight đã xanh — trông
như gate đã chạy và không báo gì. Ba commit mang theo mutant hỏng, và tuyên
bố "52/52 mutant ✅" chưa từng được kiểm chứng.

Và khi gate chạy được, nó **tìm ra một mutant sống sót** — bằng chứng đầu
tiên rằng tuyên bố "52/52 mutant ✅" là sai:

> `AppShell: deferred services re-created on every call (held pointers dangle)`
> → **SURVIVED** (`TestAppShell` không nhận ra).

Vì sao? Test cũ so sánh **con trỏ trong registry** trước và sau lần gọi thứ
hai. Nhưng `ServiceRegistry` **từ chối** đăng ký trùng (lần đầu thắng, §4.4) —
nên registry **không thể** đổi: hai lần so sánh luôn bằng nhau. Lỗi thật của
một pha deferred không idempotent là: shell **thay** instance của mình ⇒
instance cũ bị **huỷ** trong khi registry vẫn trỏ vào nó ⇒ **dangling pointer**.

Đã đóng: test kiểm tra **sự sống của instance** (`QPointer` trên các service
thừa hưởng `QObject`), không phải danh tính của entry registry. Mutant đó
giờ **bị giết** — và đây là bằng chứng rằng gate, khi nó thật sự chạy, làm
được việc mà tài liệu từng tự nhận là đã làm.

### 4.10 Đóng **cả lớp** "exception ném ra ngoài giết ứng dụng" (đã đóng 2026-10-22)

ONNX Runtime (`Ort::Exception`), OpenCV (`cv::Exception`) và bộ cấp phát
(`std::bad_alloc` khi session 227 MB không vừa RAM) đều **ném**. Một
exception rời khỏi một Qt slot (GUI thread) hoặc `QThread::run()` (background
thread) sẽ tới `std::terminate()` ⇒ **abort toàn bộ ứng dụng** (0xC0000409).
Người dùng mất cả phiên làm việc vì một request thất bại.

Tình trạng cũ: lớp *load* đã có rào (`AIProcessor::loadDetectionModel` và
anh em bắt `Ort::Exception` + `std::exception` + `...`), nhưng **lớp
*inference* thì không**:

| đường | trước | sau |
|-------|-------|-----|
| `AIProcessor::runTracking` / `runSegmentation` / `runObjectDetection` / `detectObjects` | không có rào | qua `ai::runInferenceGuarded` / `ai::runDetectionGuarded` |
| `AIService::runDetection` / `runSegmentation` / `runTracking` / `detectObjects` | gọi thẳng `m_processor` | gọi qua guard (bề mặt plugin) |
| `VideoTrackerThread::run()` | không có rào | bọc toàn bộ vòng lặp, phát `errorOccurred` + `finishedTracking` |
| `blocking::run` (`src/utils/BlockingCall.h`) | không có rào | bọc, chuyển exception sang promise, **luôn** thả semaphore |

Đã đóng bằng **một chỗ duy nhất** cho đường inference:
`src/services/ai/AIInferenceGuard.{h,cpp}` — `ai::runInferenceGuarded()` và
`ai::runDetectionGuarded()`. Khi gọi ném, guard **chứa** exception, ghi nó vào
promise (để watcher thấy đây là lỗi, không phải kết quả rỗng) và trả về
**ảnh vào** cho caller — ảnh rỗng sẽ render thành "mất hình".

Và để **không thể** thêm một entry point inference mà quên guard: luật 12 của
`tests/CheckArchitectureBoundaries.cmake` đòi **số lời gọi thô bằng số guard**
trong `AIService.cpp` — thêm một lời gọi `m_processor->…` mà không bọc guard
là vi phạm, bị bắt ngay.

Ghim: **3 mutant** (guard của `AIInferenceGuard`, rào của `VideoTrackerThread`,
rào của `blocking::run`) + **1 mutant "planted"** cho luật 12 + 3 test mới
(`TestAIService::anInferenceExceptionIsContained`,
`TestAIService::aDetectionExceptionIsContained`,
`TestPluginWidgets::videoTrackerContainsAThrowingInference`).

### 4.11 `blocking::run`: hợp đồng hoàn tất thay vì "may mắn" (đã đóng 2026-10-22)

`blocking::run` trả về một `QFuture` mà **không bao giờ gọi
`promise->finish()`**. Hệ quả đo được: `QPromise::~QPromise()` gọi
`cancelAndFinish()` cho một promise chưa finish ⇒ future được giao **ở trạng
thái CANCELLED**, và tín hiệu `finished` chỉ tới khi *đối tượng thread* bị
xoá (qua `deleteLater`) — tức là hợp đồng "kết quả được giao" phụ thuộc vào
một chi tiết vòng đời mà không ai viết ra. Thêm một dòng
`if (watcher.isCanceled()) return;` — rất tự nhiên — là kết quả bị bỏ rơi âm
thầm.

Đã đóng: `run()` gọi `promise->finish()` **tường minh** (future là
*completed*, không phải *cancelled*), **chứa** exception, và **luôn** thả
semaphore của `TaskTracker` — kể cả khi callable ném.

Một chi tiết quan trọng phát sinh khi sửa: exception **không** được lưu trên
promise, vì `QFuture::result()` **ném lại** exception đã lưu — và lần ném đó
xảy ra **trong slot của caller**, tức là lại về `std::terminate()`, chỉ xa
nguyên nhân thêm một khung stack. Caller nhận **result mặc định** (với kiểu
"có thành công không" thì đó là giá trị thất bại) và lỗi được ghi log. Một
callable muốn báo lỗi *nghiệp vụ* thì bắt lỗi ngay trong callable và trả về
trong result — đúng như mail stack đã làm (`errorMsg` do service điền).

Ghim: 2 mutant (`BlockingCall: the promise is never finished`,
`BlockingCall: an exception in the callable escapes the worker thread`) + 2
test trong `TestUtilsLayer`.

### 4.12 Đóng **cả lớp** "inference chạy trên GUI thread" (đã đóng 2026-10-22)

Đây là dòng cuối của bảng §2.1. `AIProcessorPlugin::onObjectDetection()` và
`onSegmentation()` gọi thẳng `IAIService::runDetection()` /
`runSegmentation()` **trong slot GUI**. Một lần ONNX chạy hàng trăm mili-giây
tới vài giây, nên cửa sổ đứng hình suốt thời gian đó — và ba gate trước
**không thấy**: coverage đo dòng (dòng đó có chạy), mutation ghim lớp *mail*,
còn architecture guard chỉ cấm `QtConcurrent::run` chứ không cấm một lời gọi
đồng bộ.

Đã đóng bằng **một đường duy nhất** cho cả hai entry point:

| quyết định | trước | sau |
|------------|-------|-----|
| nơi chạy inference | trong slot GUI | `blocking::run(task, &m_inferenceTasks)` — QThread riêng, không phải pool (pool có thể chạy inline trên thread gọi) |
| kết quả | áp ngay sau lời gọi | `QFutureWatcher<cv::Mat>` → `finishInference()`, chỉ áp khi `!result.empty()` |
| số inference đồng thời | không giới hạn | **một** (`m_inferenceRunning`): ONNX session dùng chung, hai kết quả tranh nhau cùng viewer thì kết quả cũ có thể thắng |
| teardown | không chờ | `cleanup()` đặt `m_cleanedUp`, chờ `m_inferenceTasks` **có bound 3 s** trước khi xoá widget; task chỉ giữ service AppCore + giá trị, nên nó có thể kết thúc sau khi plugin unload mà không chạm object của plugin |
| backend ném | lan ra slot → `std::terminate()` | `blocking::run` chứa exception (§4.10/§4.11), watcher nhận `cv::Mat` rỗng, `finishInference()` báo `aiproc.inference_failed` thay vì render ảnh rỗng |

Ghim: **4 test** trong `TestPluginUi` — detection và segmentation chạy khác
GUI thread (so `QThread::currentThreadId()`), click thứ hai bị bỏ qua khi đang
chạy (và click sau khi xong vẫn chạy), backend ném được báo **đúng một lần**
và không đổi AI mode — cùng **2 mutant** (dispatch đồng bộ; bỏ busy guard).
`TestPluginUi` biên dịch thêm `AIProcessorPlugin.cpp` + `VideoTrackerThread.cpp`
để đo chính đường này.

Bài học lặp lại từ C6: guard kiến trúc là regex trên mã nguồn, nó **không**
thay được một test khẳng định thread id. Lớp lỗi này sống sót qua cả bốn gate
cho tới khi có một test hỏi đúng câu "inference chạy ở thread nào?".

## 5. Nợ còn lại (có đo, chưa xử lý)

Xem bảng "coverage debt" trong `tests/README.md` — mỗi dòng là một file chưa
được test, kèm lý do (GUI-only, VTK-only, network-only) và hướng xử lý.

Nguyên tắc: nợ phải **được đo và được ghi**, không được biến mất khỏi báo cáo.

### 5.1 Dữ liệu at-rest phía C++

Nguyên tắc tương tự: **đã niêm phong thì phải ghi ra, chưa niêm phong thì
cũng phải ghi** — không có trạng thái "không biết".

| Dữ liệu | Trạng thái | Cơ chế |
|----------|-----------|--------|
| Lịch sử chat (`chat_sessions_<user>.json`) | **Đã niêm phong** | `SecretVault` (DPAPI CurrentUser trên Windows); file cũ dạng plaintext vẫn đọc được và được seal lại ở lần `save()` kế tiếp |
| Khoá licence, mật khẩu, SMTP credential | **Đã niêm phong** | `SecretVault` (`enc1:`) |
| **Khoá API của nhà cung cấp AI** | **Đã niêm phong** | `UserSettingsStore::secretKeys()` — khoá nằm trong namespace `secret_`, mã hoá bằng `SecretVault`; bản plaintext cũ được dọn khi nạp |
| Ảnh đính kèm & thumbnail (`Upload/`, `Thumbnails/`) | **Đã niêm phong** | ADR 0031 — AES-256-GCM bằng **cùng key at-rest** mà dịch vụ AI dùng, nên cả hai tiến trình đọc được file đã mã hoá. Envelope khớp byte-for-byte giữa hai ngôn ngữ, ghim bằng **vector chéo**: mỗi bên giữ output của bên kia |
| **Trên POSIX** (`SecretVault`, `AtRestCipher`) | **KHÔNG có OS crypto** | Không có backend mã hoá cho POSIX trong app này: `encrypt()` ghi `plain:<giá trị>` kèm cảnh báo. Đây là giới hạn nền tảng **đã biết**, không phải bí mật: app C++ nhắm Windows (DPAPI), còn state của dịch vụ AI được niêm phong bằng `cryptography` trên **mọi** nền tảng |

Hai mutant trong `tests/mutation_check.py` ghim hành vi của lớp chat: bỏ bước
seal khi ghi, và bỏ bước unseal khi đọc. Cả hai đều phải làm đỏ
`TestChatSessionManager`.

### 5.2 Vault fail-closed (đã đóng)

Trước đây `SecretVault::encrypt()` **fail-open**: khi `CryptProtectData` thất
bại trên Windows, bí mật được ghi thẳng vào đĩa dưới dạng `plain:<giá trị>`
— và `decrypt()` bóc marker đó *không cảnh báo*, nên hạ cấp bảo mật là
**vô hình**. Đây không phải giới hạn POSIX đã ghi nhận: đó là một lỗi trên
chính nền tảng có backend.

Đã đóng: `encrypt()` trả về chuỗi rỗng khi bảo vệ thất bại, và các caller
giữ nguyên giá trị đang lưu (`UserSettingsStore::storeSecret`,
`LicenseKeyLedger::save`). `SecretVault::protectionFailureFallback()` là quyết
định có tên, `setProtectionFailureForTesting()` là seam để ghim hành vi
fail-closed của các caller bằng mutation.

### 5.3 Nợ còn lại đã ghi nhận

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

### 6.1 Thứ tự chạy, và điều gì phải **xanh** trước khi kết luận

| bước | lệnh | phải xanh | nếu đỏ thì nghĩa là gì |
|------|-------|-----------|-----------------------|
| 1 | `cmake -DSOURCE_DIR=<repo> -P tests/CheckArchitectureBoundaries.cmake` | exit 0 | một luật ranh giới bị vi phạm |
| 2 | `ctest --test-dir build -C Release --output-on-failure` | 42/42 | một hợp đồng quan sát được bị phá |
| 3 | `python tests/coverage_gate.py` | "98 file(s) at or above their floors" | một file đã **thoát** khỏi phép đo, hoặc coverage **tụt** dưới floor |
| 4 | `python tests/mutation_check.py` | "63/63 killed, 0 survived, 0 stale, 0 invalid" | một quyết định bảo vệ **không còn được ghim** |

Bước 4 **tự kiểm mục tiêu của mình trước khi rebuild** (file tồn tại, anchor
đúng một lần, header đã ghim) — nên nó đỏ **ngay** với danh sách mutant hỏng,
chứ không "chạy xong rồi không báo gì" như trước đây.

Số liệu trong tài liệu này lấy từ **output của các gate trên**, không viết
tay: đổi một số mà không đổi lệnh đo là tự ghi dối.
