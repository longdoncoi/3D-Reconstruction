# 3D Reconstruction & AI Medical Assistant

Phần mềm y tế tích hợp hiển thị ảnh DICOM MPR, tái tạo 3D từ ảnh 2D (Structure from Motion), trợ lý ảo AI thông minh chạy cục bộ (Local LLM + RAG), quản lý mail nội bộ và xác thực người dùng. Toàn bộ được xây dựng trên nền tảng **Qt 6** với kiến trúc **Service-Oriented Modular Plugin** — loại bỏ triệt để Anti-pattern "God Object".

---

## 🏗️ Kiến trúc Tổng quan

## 🤖 Hệ sinh thái AI Agent Workflow
Dự án được thiết lập sẵn một hệ thống quy trình chuẩn hóa dành riêng cho AI Agent (Copilot, Gemini, Cursor) khi tham gia lập trình:

- **`Instructions.md`**: Đóng vai trò là "System Prompt" cốt lõi. AI Agent khi mở project sẽ đọc file này để hiểu tổng quan kiến trúc (Service-Oriented, Plugin), các quy tắc SOLID bắt buộc, coding standards và những Anti-patterns bị cấm.
- **`.github/`**: Chứa CI/CD pipeline (`build.yml`). Hệ thống tự động kiểm tra cú pháp (Syntax check với chuẩn C++20) mỗi khi có Push/PR, đảm bảo code do người dùng hay AI sinh ra đều chuẩn xác.
- **`.prompts/`**: Chứa các template hướng dẫn cụ thể cho AI theo từng loại task (`bugfix.md`, `feature.md`, `refactor.md`). AI tuân theo các bước trong này để lập kế hoạch và viết code có hệ thống.
- **`.review/`**: Chứa `checklist.md` quy định tiêu chí đánh giá code (memory leak, thread safety, SOLID). AI sẽ dựa vào đây để tự kiểm tra (verify) lại code của mình trước khi hoàn thành task.
- **`.tasks/`**: Thư mục quản lý tiến độ. Gồm `README.md` hướng dẫn quy trình làm việc 8 bước và `task-template.md` chứa form khai báo Goal, Requirements cho từng task cụ thể, giúp AI làm việc tập trung và không đi lạc đề.
- **`scripts/`**: Chứa các đoạn script tự động hóa (vd: `agent_pipeline.ps1`, `agent_pipeline.sh`). AI Agent có thể gọi trực tiếp các script này để tự động build, chạy test và sinh commit mà không cần gõ lệnh thủ công lặp đi lặp lại.

---

## END      

### Service-Oriented Modular Plugin

Mọi tính năng trong dự án đều được tổ chức thành các Plugin độc lập, giao tiếp qua một bộ Interface trung tâm thay vì phụ thuộc trực tiếp lẫn nhau.

```
MainWindow (Shell)
    │
    ├── IAppContext ── Service Locator ──► ServiceRegistry<T>
    │       ├── ISceneService       (VTK 3D rendering)
    │       ├── IViewerService      (2D image navigation)
    │       ├── ISettingsService    (persistent settings)
    │       ├── IAIAssistantService (chat & LLM)
    │       ├── IReconstructionService (SfM pipeline)
    │       └── IMailService        (IMAP/SMTP)
    │
    ├── SignalBus ── Event Bus ──► decoupled UI events
    │
    └── Plugins (dynamic .dll)
            ├── ViewerPlugin
            ├── ReconstructionPlugin
            ├── AIProcessorPlugin
            ├── AIAssistantPlugin
            ├── SendMailPlugin
            └── UserAuthPlugin
```

**Các nguyên tắc thiết kế:**
- **`IAppContext`** — Service Locator duy nhất, Plugin chỉ nhận `IAppContext*` trong constructor, không cần biết đến `MainWindow`.
- **`ServiceRegistry<T>`** — Đăng ký và tra cứu service bất kỳ theo kiểu (type-safe). Tránh hoàn toàn Singleton toàn cục.
- **`SignalBus`** — Event Bus tập trung cho các sự kiện UI (đổi ảnh, thay đổi slice, đổi ngôn ngữ). Plugin phát sự kiện mà không cần biết ai lắng nghe.
- **`IPlugin`** — Giao thức lifecycle (`loadOrder()`, `onAppReady()`, `retranslate()`) để kiểm soát thứ tự khởi tạo và hỗ trợ đa ngôn ngữ động.
- **`AppConfig`** — Singleton quản lý tập trung toàn bộ đường dẫn hệ thống (Models, Logs, Predict, Config...). Loại bỏ macro phụ thuộc phần cứng.
- **`AppConstants`** — Hằng số toàn cục chia theo namespace (`AIServer`, `UI`, `App`).

---

## 🌟 Các Mô-đun & Tính năng

### 1. `ViewerPlugin` — Xem ảnh y tế & DICOM MPR
- **DICOM MPR**: Đồng bộ 3 mặt cắt Axial / Sagittal / Coronal qua Crosshair tâm điểm.
- **Volume Rendering**: Dựng khối 3D từ dữ liệu DICOM với VTK, điều hướng đa góc nhìn.
- **2D Viewer**: Hỗ trợ cuộn ảnh (Auto Prev/Next), phóng to/thu nhỏ, điều hướng theo bộ ảnh.
- **Thumbnail Strip**: Dải ảnh thu nhỏ phía dưới, click để nhảy đến frame bất kỳ.

### 2. `ReconstructionPlugin` — Tái tạo 3D từ ảnh (SfM)
- **Pipeline SfM**: Trích xuất đặc trưng (SIFT/ORB) → Khớp điểm (RANSAC) → Ước lượng Camera Pose → Triangulation.
- **Point Cloud**: Sinh đám mây điểm 3D có màu từ chuỗi ảnh 2D đa góc chụp, hiển thị qua PCL.
- **3D Model Viewer**: Xem và quản lý model 3D đầu ra (.ply, .obj).

### 3. `AIProcessorPlugin` — AI Detection, Segmentation & Tracking
- **YOLOv11 Inference**: Nhận diện đối tượng (Detection) và phân vùng ảnh (Segmentation) chạy GPU qua ONNX Runtime.
- **Auto re-run**: Tự động chạy lại inference khi người dùng cuộn sang ảnh mới.

### 4. `AIAssistantPlugin` — Trợ lý ảo AI (Local LLM + RAG)
- **Local LLM**: Tích hợp mô hình Qwen2.5 (text) và Qwen2.5-VL (vision) chạy hoàn toàn cục bộ qua `llama-cpp-python`.
- **RAG**: Retrieval-Augmented Generation — chatbot trích xuất thông tin từ tài liệu dự án (`Docs/`) và mã nguồn để trả lời chính xác.
- **Chat History**: Lưu lịch sử hội thoại theo session, hỗ trợ Retry và Edit từng tin nhắn, hiển thị thời gian.
- **Vision**: Phân tích ảnh đính kèm (DICOM, PNG) khi dùng Vision model.
- **Chat UI**: Markdown rendering, attachment, bong bóng chat, Image Viewer phóng to.

### 5. `SendMailPlugin` — Mail nội bộ (IMAP/SMTP)
- **Inbox**: Tải danh sách mail từ server IMAP với loading progress dialog.
- **Compose**: Soạn và gửi mail qua SMTP với đính kèm tệp.
- **Filter**: Lọc mail theo keyword (domain, no-reply...) để ẩn thư rác.
- **Rich Preview**: Hiển thị nội dung mail với format HTML, trích dẫn mail gốc khi reply.

### 6. `UserAuthPlugin` — Xác thực & Phân quyền Người dùng
- **Đăng nhập / Đăng xuất**: Quản lý phiên làm việc với giao diện login hiện đại.
- **Phân quyền**: Kiểm soát quyền truy cập tính năng theo role người dùng.
- **Quản lý tài khoản**: Thêm/xóa/sửa thông tin người dùng.

---

## 📂 Cấu trúc Thư mục Dự án

```
3D-Reconstruction/
│
├── src/                                    ← Toàn bộ mã nguồn C++ chính của ứng dụng
│   │   pch.h                               ← Precompiled Header — include sẵn Qt Core/Widgets để tăng tốc biên dịch
│   │   pch_dummy.cpp                       ← File rỗng bắt buộc cho CMake kích hoạt PCH
│   │
│   ├── app/                                ← Lõi ứng dụng — Shell khởi tạo, không chứa business logic
│   │   │   Main.cpp                        ← Entry point (hàm main), khởi tạo QApplication & Logger
│   │   │   AppShell.*                      ← Khởi tạo ServiceRegistry, nạp Plugin, wire dependencies
│   │   │   MainWindow.*                    ← Shell UI chính: Ribbon, DockWidget, StatusBar
│   │   │   MainWindow.ui                   ← Layout giao diện Qt Designer cho MainWindow
│   │   │   StyleManager.*                  ← Quản lý theme/stylesheet động (dark/light mode)
│   │   │   AppConstants.h                  ← Hằng số toàn cục: namespace AIServer, UI, App
│   │   │   CMakeLists.txt                  ← Cấu hình CMake cho target app
│   │   │   resources.qrc                   ← Khai báo tài nguyên Qt (icon, flag, ảnh)
│   │   │   app.rc                          ← Windows resource file (version info, icon)
│   │   │   app_icon.ico / app_icon.png     ← Icon ứng dụng
│   │   └── flag_en.png / flag_vn.png       ← Icon cờ quốc gia cho chuyển đổi ngôn ngữ
│   │
│   ├── core/                               ← Nền tảng hệ thống — Interface contracts & core services
│   │   │   IAppContext.h                   ← Service Locator interface — cổng giao tiếp duy nhất cho Plugin
│   │   │   IPlugin.h                       ← Lifecycle interface: loadOrder(), onAppReady(), retranslate()
│   │   │   ServiceRegistry.h              ← Template container type-safe để đăng ký/tra cứu service
│   │   │   SignalBus.*                     ← Event Bus tập trung — phát/nhận sự kiện UI decoupled
│   │   │   AppConfig.*                     ← Singleton quản lý đường dẫn hệ thống (Models, Logs, Config...)
│   │   │   LanguageManager.*               ← Quản lý đa ngôn ngữ động (EN/VI), đọc JSON translations
│   │   │   UserManager.*                   ← Quản lý tài khoản người dùng, xác thực, phân quyền
│   │   │   SmtpMailer.*                    ← SMTP client cấp thấp — gửi mail qua socket SSL/TLS
│   │   │   Global.h                        ← Macro APP_EXPORT cho DLL export/import symbols
│   │   │   AIMode.h                        ← Enum định nghĩa chế độ AI: Detection, Segmentation, Tracking
│   │   │   IAIService.h                    ← Interface cho YOLO inference (detection/segmentation)
│   │   │   IAIAssistantService.h           ← Interface cho LLM chatbot (gửi prompt, nhận response)
│   │   │   IMailService.h                  ← Interface cho IMAP client (fetch inbox, đọc mail)
│   │   │   IReconstructionService.h        ← Interface cho SfM pipeline (chạy reconstruction)
│   │   │   ISceneService.h                 ← Interface cho VTK 3D scene (render, camera, actor)
│   │   │   ISettingsService.h              ← Interface cho persistent settings (QSettings wrapper)
│   │   │   IViewerService.h                ← Interface cho 2D image navigation (next/prev, zoom)
│   │   └── CMakeLists.txt                  ← Cấu hình CMake cho thư viện core (static lib)
│   │
│   ├── services/                           ← Triển khai (Implementation) cụ thể của các Interface
│   │   │   CMakeLists.txt                  ← Cấu hình CMake cho thư viện services
│   │   │
│   │   ├── ai/                             ← Service xử lý AI — LLM chatbot & YOLO inference
│   │   │   │   AIService.*                 ← Impl IAIService — điều phối YOLO detect/segment qua ONNX Runtime
│   │   │   │   AIProcessor.*              ← Chạy ONNX inference trên GPU, pre/post-process tensor
│   │   │   │   AIYoloPostProcessor.*       ← Hậu xử lý output YOLO: NMS, vẽ bounding box, mask overlay
│   │   │   │   AIAssistant.*               ← Impl IAIAssistantService — gọi FastAPI LLM server qua HTTP
│   │   │   └── ChatSessionStore.*          ← Lưu/đọc lịch sử chat session (JSON file trên ổ đĩa)
│   │   │
│   │   ├── mail/                           ← Service xử lý email nội bộ
│   │   │   │   MailService.*               ← Impl IMailService — kết nối IMAP, fetch/parse email
│   │   │   └── MailMimeParser.*            ← Phân tích cấu trúc MIME: tách body, attachment, encoding
│   │   │
│   │   ├── reconstruction/                 ← Pipeline tái tạo 3D từ ảnh (Structure from Motion)
│   │   │   │   ReconstructionService.*     ← Impl IReconstructionService — API điều phối pipeline
│   │   │   │   ReconstructionPipeline.*    ← Orchestrator: điều phối toàn bộ bước SfM tuần tự
│   │   │   │   ReconstructThread.*         ← QThread chạy pipeline không block UI
│   │   │   │   ReconstructionConfig.h      ← Struct cấu hình pipeline: thuật toán, ngưỡng, tham số
│   │   │   │   CameraParamsParser.*        ← Đọc/ghi intrinsic camera parameters từ file
│   │   │   │   FeatureExtractor.*          ← Trích xuất đặc trưng ảnh (SIFT/ORB) qua OpenCV
│   │   │   │   FeatureMatcher.*            ← Khớp đặc trưng giữa cặp ảnh (BFMatcher + RANSAC)
│   │   │   │   FeatureTrack.h              ← Struct theo dõi feature points qua nhiều ảnh
│   │   │   │   PoseEstimator.*             ← Ước lượng Camera Pose (Essential Matrix, PnP)
│   │   │   │   Triangulator.*              ← Triangulation 2 view — tính tọa độ 3D từ cặp ảnh
│   │   │   │   MultiViewTriangulator.*     ← Triangulation đa góc nhìn (N-view) mở rộng
│   │   │   └── PointCloudFilter.*          ← Lọc nhiễu đám mây điểm: SOR, voxel grid downsampling
│   │   │
│   │   ├── scene/                          ← Quản lý VTK 3D scene (DICOM MPR & Volume Rendering)
│   │   │   │   SceneService.*              ← Impl ISceneService — quản lý renderer, camera, actor VTK
│   │   │   │   CrosshairManager.*          ← Quản lý crosshair đồng bộ 3 mặt cắt MPR
│   │   │   └── CrosshairGeometry.*         ← Tính toán hình học crosshair (line, plane intersection)
│   │   │
│   │   ├── settings/                       ← Persistent settings
│   │   │   └── SettingsService.*           ← Impl ISettingsService — đọc/ghi QSettings (INI file)
│   │   │
│   │   └── viewer/                         ← Service hiển thị ảnh 2D & tải dữ liệu
│   │       │   ViewerService.*             ← Impl IViewerService — điều hướng ảnh, auto prev/next
│   │       │   Image2DLoader.*             ← Tải và decode ảnh 2D (PNG, JPG, BMP)
│   │       │   DicomLoader.*               ← Tải và parse file DICOM qua VTK DICOM reader
│   │       │   Model3DLoader.*             ← Tải model 3D (.obj, .ply) thành vtkActor
│   │       └── ChatImageViewer.*           ← Widget phóng to ảnh đính kèm trong chat AI
│   │
│   ├── modules/                            ← Plugin động (.dll) — mỗi plugin là một tính năng nghiệp vụ độc lập
│   │   │   CMakeLists.txt                  ← Cấu hình CMake tổng cho tất cả plugin
│   │   │
│   │   ├── ViewerPlugin/                   ← Plugin xem ảnh y tế & DICOM MPR
│   │   │   │   ViewerPlugin.*              ← Impl IPlugin — đăng ký UI, kết nối service
│   │   │   │   ViewerRibbonUI.*            ← Thanh Ribbon: nút Open, DICOM, 3D Model, AI mode
│   │   │   │   ViewerListUI.*              ← Panel danh sách ảnh/file dạng thumbnail strip
│   │   │   │   ViewerNavigatorUI.*         ← Thanh điều hướng: prev/next, auto scroll, zoom
│   │   │   │   ViewerViewModel.*           ← ViewModel quản lý state hiển thị (MVVM pattern)
│   │   │   └── CMakeLists.txt              ← Cấu hình CMake cho ViewerPlugin.dll
│   │   │
│   │   ├── ReconstructionPlugin/           ← Plugin tái tạo 3D (SfM)
│   │   │   │   ReconstructionPlugin.*      ← Impl IPlugin — UI chạy pipeline, hiển thị kết quả
│   │   │   │   ReconstructionRibbonUI.*    ← Thanh Ribbon: nút Start Reconstruct, chọn dataset
│   │   │   └── CMakeLists.txt              ← Cấu hình CMake cho ReconstructionPlugin.dll
│   │   │
│   │   ├── AIProcessorPlugin/              ← Plugin AI Detection, Segmentation & Video Tracking
│   │   │   │   AIProcessorPlugin.*         ← Impl IPlugin — điều phối YOLO inference theo ảnh hiện tại
│   │   │   │   AIProcessorRibbonUI.*       ← Thanh Ribbon: chọn model, chế độ AI, nút Run
│   │   │   │   AITrainDockWidget.*         ← DockWidget hiển thị kết quả inference & log
│   │   │   │   AIPredictionLogger.*        ← Ghi kết quả dự đoán ra ảnh + metadata (timestamp)
│   │   │   │   VideoTrackerThread.*        ← QThread chạy YOLO tracking trên video frame-by-frame
│   │   │   └── CMakeLists.txt              ← Cấu hình CMake cho AIProcessorPlugin.dll
│   │   │
│   │   ├── AIAssistantPlugin/              ← Plugin trợ lý ảo AI (Local LLM + RAG)
│   │   │   │   AIAssistantPlugin.*         ← Impl IPlugin — khởi tạo chat UI, kết nối LLM service
│   │   │   │   AIAssistantRibbonUI.*       ← Thanh Ribbon: chọn model text/vision, nút New Chat
│   │   │   │   ChatBotDockWidget.*         ← DockWidget chính: chat bubble, input box, attachment
│   │   │   │   ChatMessageRenderer.*       ← Render tin nhắn: Markdown→HTML, code highlight, timestamp
│   │   │   │   ChatTemplates.h             ← HTML/CSS template cho bong bóng chat (user/bot style)
│   │   │   │   AIAttachmentPreviewFactory.* ← Tạo preview thumbnail cho file đính kèm (ảnh, doc)
│   │   │   └── CMakeLists.txt              ← Cấu hình CMake cho AIAssistantPlugin.dll
│   │   │
│   │   ├── SendMailPlugin/                 ← Plugin quản lý mail nội bộ (IMAP/SMTP)
│   │   │   │   SendMailPlugin.*            ← Impl IPlugin — khởi tạo mail UI, kết nối MailService
│   │   │   │   SendMailRibbonUI.*          ← Thanh Ribbon: nút Compose, Refresh Inbox, Filter
│   │   │   │   MailDockWidget.*            ← DockWidget chính: inbox list + mail content preview
│   │   │   │   MailFilterDialog.*          ← Dialog cấu hình bộ lọc mail (keyword, domain, no-reply)
│   │   │   │   MailInboxItemFactory.*      ← Factory tạo widget cho từng mail item trong inbox list
│   │   │   │   MailMessageFormatter.*      ← Format nội dung mail: HTML render, trích dẫn reply
│   │   │   │   MailSettingsDialog.*        ← Dialog cấu hình IMAP/SMTP server (host, port, SSL)
│   │   │   └── CMakeLists.txt              ← Cấu hình CMake cho SendMailPlugin.dll
│   │   │
│   │   └── UserAuthPlugin/                 ← Plugin xác thực & quản lý người dùng
│   │       │   UserAuthPlugin.*            ← Impl IPlugin — hiển thị login khi khởi động, phân quyền
│   │       │   LoginDialog.*               ← Dialog đăng nhập với username/password
│   │       │   ForgotPasswordDialog.*      ← Dialog khôi phục mật khẩu qua email
│   │       │   ChangePasswordDialog.*      ← Dialog đổi mật khẩu người dùng
│   │       │   AdminUserManagerDialog.*    ← Dialog quản trị: thêm/xóa/sửa tài khoản (Admin only)
│   │       │   SettingsDialog.*            ← Dialog cài đặt chung của ứng dụng
│   │       │   AboutDialog.*               ← Dialog thông tin phần mềm (version, credits)
│   │       │   AvatarCropperDialog.*       ← Dialog cắt và upload ảnh đại diện người dùng
│   │       │   ThemeSelectionDialog.*      ← Dialog chọn giao diện (dark/light/custom theme)
│   │       │   LicenseActivationDialog.*   ← Dialog kích hoạt bản quyền phần mềm
│   │       └── CMakeLists.txt              ← Cấu hình CMake cho UserAuthPlugin.dll
│   │
│   └── utils/                              ← Tiện ích dùng chung cho toàn bộ ứng dụng
│       │   Logger.*                        ← Hệ thống ghi log tập trung (file + console)
│       │   FileUtilities.*                 ← Xử lý file: copy attachment, tạo thumbnail, unique name
│       │   HtmlUtilities.*                 ← Chuyển đổi Markdown → HTML cho chat & mail
│       │   IconFactory.*                   ← Factory tạo icon gradient từ emoji (thay thế icon thủ công)
│       │   ModernDialog.*                  ← Base class dialog frameless có title bar tùy chỉnh
│       │   ModernMessageBox.h              ← Static helper: information/warning/question dialog
│       │   CustomProgressDialog.*          ← Dialog progress bar có nút Stop (cho tải mail, SfM)
│       │   PanStyle.*                      ← Custom VTK interactor: chuột phải = pan thay vì rotate
│       └── CMakeLists.txt                  ← Cấu hình CMake cho thư viện utils (static lib)
│
├── tests/                      ← Unit tests (Google Test)
│   │   CMakeLists.txt                  ← Cấu hình CMake cho test runner
│   │   TestAppConfig.cpp               ← Test quản lý đường dẫn hệ thống
│   │   TestCameraParamsParser.cpp      ← Test đọc/ghi camera parameters
│   │   TestLanguageManager.cpp         ← Test chuyển đổi ngôn ngữ EN/VI
│   │   TestMailMimeParser.cpp          ← Test phân tích cấu trúc MIME email
│   │   TestPointCloudFilter.cpp        ← Test lọc nhiễu đám mây điểm 3D
│   │   TestReconstructionPipeline.cpp  ← Test pipeline SfM end-to-end
│   │   TestServiceRegistry.cpp         ← Test đăng ký/tra cứu service type-safe
│   └── TestSignalBus.cpp               ← Test Event Bus phát/nhận signal
│
├── AIAssistant/                ← Hệ thống AI Server, Agent và RAG (Python)
├── AIComputerVision/           ← Huấn luyện YOLO, models và kết quả training
│   ├── StartChatbotServer.py   ← FastAPI server chạy LLM + RAG
│   └── requirements.txt        ← Python dependencies
│
├── Docs/                       ← Tài liệu dự án (PDF, DOCX, TXT)
│                               ← RAG đọc từ thư mục này để trả lời câu hỏi
│
├── Models/                     ← GGUF models (LLM, mmproj)
│                               ← ONNX models (YOLO detection/segmentation)
│
├── translations/               ← File đa ngôn ngữ JSON
│   ├── translations_en.json    ← Chuỗi tiếng Anh
│   └── translations_vi.json    ← Chuỗi tiếng Việt
│
├── Doxygen/                    ← Cấu hình và output tài liệu Doxygen
├── CMakeLists.txt              ← Build system gốc (CMake) — khai báo target, link thư viện
└── vcpkg.json                  ← Khai báo dependencies C++ (vcpkg manifest mode)
```

---

## 🛠️ Yêu cầu Hệ thống

| Thành phần | Phiên bản |
|---|---|
| OS | Windows 10/11 x64 |
| Compiler | MSVC 2022 (v143) |
| Qt | 6.9.3 |
| VTK | 9.6.0 |
| OpenCV | 4.x |
| PCL | 1.15.1 |
| ONNX Runtime | 1.20.1 (CUDA) |
| Python | 3.10+ (cho AI Server) |

---

## ⚙️ Biên dịch & Triển khai

1. **Cấu hình CMake**: Mở dự án trong Qt Creator. Kiểm tra đường dẫn VTK, OpenCV, PCL trong `CMakeLists.txt`.
2. **Build**: Dùng Kit **MSVC 2022 64-bit** — khuyến nghị **Release mode** để đạt hiệu suất GPU và render tốt nhất.
   > ⚠️ Nếu thay đổi bất kỳ file nào trong `src/core/`, hãy thực hiện **Clean & Rebuild** toàn bộ project.
3. **Plugin output**: Sau khi build, các `.dll` plugin tự động được copy vào `plugins/` cạnh file thực thi.
4. **AI Server**: Chạy `AIAssistant/StartChatbotServer.py` để khởi động server LLM trước khi dùng AI Assistant.

---

## 📝 Mở rộng Tính năng

Để thêm một tính năng mới:
1. Tạo thư mục Plugin mới trong `src/modules/`.
2. Implement interface `IPlugin` (hoặc service interface tương ứng trong `src/core/`).
3. Đăng ký Plugin trong `MainWindow` — **không cần sửa bất kỳ Plugin nào khác**.
4. Thêm chuỗi i18n vào `translations/translations_en.json` và `translations_vi.json`.

---

## 📚 Tài liệu Code (Doxygen)

```bash
cd Doxygen/
doxygen Doxyfile
# Mở Doxygen/html/index.html để xem tài liệu API đầy đủ
```
