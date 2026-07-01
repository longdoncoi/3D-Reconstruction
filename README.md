# 3D Reconstruction & AI Medical Assistant

Phần mềm y tế tích hợp hiển thị ảnh DICOM MPR, tái tạo 3D từ ảnh 2D (Structure from Motion), trợ lý ảo AI thông minh chạy cục bộ (Local LLM + RAG), quản lý mail nội bộ và xác thực người dùng. Toàn bộ được xây dựng trên nền tảng **Qt 6** với kiến trúc **Service-Oriented Modular Plugin** — loại bỏ triệt để Anti-pattern "God Object".

---

## 🏗️ Kiến trúc Tổng quan

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
- **Video Tracking**: Theo dõi đối tượng trong video với model YOLO-Track, lưu kết quả vào `VideoTracking/`.
- **Predict Logs**: Tự động lưu kết quả dự đoán vào `Predict/detection/` và `Predict/segmentation/` với timestamp.
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
├── src/                        ← Toàn bộ mã nguồn C++
│   ├── core/                   ← Interfaces & nền tảng hệ thống
│   │   ├── IAppContext.h       ← Service Locator — cổng giao tiếp chính
│   │   ├── IPlugin.h           ← Lifecycle interface cho mọi Plugin
│   │   ├── SignalBus.h/.cpp    ← Event Bus tập trung
│   │   ├── ServiceRegistry.h  ← Type-safe dependency injection
│   │   ├── AppConfig.h/.cpp    ← Quản lý đường dẫn tập trung (Singleton)
│   │   ├── LanguageManager     ← Đa ngôn ngữ động (EN/VI)
│   │   ├── UserManager         ← Quản lý tài khoản & xác thực
│   │   ├── SmtpMailer          ← SMTP client cấp thấp
│   │   └── I*Service.h         ← Toàn bộ Interface service contracts
│   │
│   ├── app/                    ← Lõi ứng dụng (MainWindow shell)
│   │   ├── MainWindow          ← Shell khởi tạo Plugin, không chứa business logic
│   │   ├── StyleManager        ← Quản lý giao diện (theme, stylesheet)
│   │   └── AppConstants.h      ← Hằng số hệ thống theo namespace
│   │
│   ├── services/               ← Triển khai (Implementation) của các Interface
│   │   ├── ai/                 ← AIAssistant: giao tiếp với LLM server
│   │   ├── mail/               ← MailService: IMAP client
│   │   ├── reconstruction/     ← SfM pipeline (OpenCV, PCL)
│   │   ├── scene/              ← SceneService: VTK 3D scene management
│   │   ├── settings/           ← SettingsService: QSettings wrapper
│   │   └── viewer/             ← ViewerService: 2D image navigation
│   │
│   ├── modules/                ← Plugin động (.dll), tính năng nghiệp vụ
│   │   ├── ViewerPlugin/       ← DICOM MPR, 2D viewer, thumbnail
│   │   ├── ReconstructionPlugin/ ← SfM UI, 3D model viewer
│   │   ├── AIProcessorPlugin/  ← YOLO detection/segmentation/tracking UI
│   │   ├── AIAssistantPlugin/  ← Chat UI, LLM integration, RAG client
│   │   ├── SendMailPlugin/     ← Inbox, compose, filter mail UI
│   │   └── UserAuthPlugin/     ← Login/logout, user management UI
│   │
│   └── utils/                  ← Tiện ích dùng chung
│       ├── ModernMessageBox    ← Dialog thông báo tùy chỉnh
│       └── ...                 ← Các helper UI khác
│
├── AITraining/                 ← Hệ thống AI Server (Python)
│   ├── StartChatbotServer.py   ← FastAPI server chạy LLM + RAG
│   ├── requirements.txt        ← Python dependencies
│   ├── Cache/                  ← FAISS index và BM25 cache
│   └── logs/                   ← Log file server
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
├── Config/                     ← Cấu hình ứng dụng (mail, user settings)
├── Predict/                    ← Kết quả dự đoán AI (detection, segmentation)
├── VideoTracking/              ← Kết quả video tracking
├── Thumbnails/                 ← Cache ảnh thu nhỏ cho viewer
├── Upload/                     ← Tệp đính kèm mail tạm thời
├── Dicom/                      ← Dữ liệu DICOM đầu vào
├── Dataset/                    ← Dataset ảnh 2D cho SfM reconstruction
├── 2DImages/                   ← Ảnh 2D đầu vào cho viewer
├── 3DModels/                   ← Model 3D đầu ra từ SfM
├── Doxygen/                    ← Cấu hình và output tài liệu Doxygen
├── tests/                      ← Unit tests
├── CMakeLists.txt              ← Build system (CMake)
└── vcpkg.json                  ← Khai báo dependencies (vcpkg)
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
4. **AI Server**: Chạy `AITraining/StartChatbotServer.py` để khởi động server LLM trước khi dùng AI Assistant.

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
