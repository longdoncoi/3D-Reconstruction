# ──────────────────────────────────────────────────────────────────────────────
# Architecture boundary guard — encodes the invariants from AGENTS.md so a
# refactor that forgets them fails CI instead of silently eroding the plugin ABI:
#
#   1. Plugin-facing surface = core interfaces (I*.h) + shared non-interface
#      core glue (Global/AppConstants/AIMode/SignalBus/ServiceRegistry) +
#      shared UI utilities (src/utils/**) + the plugin's own headers. Includes of concrete service headers (SceneService,
#      ShellService, …), app-shell headers (AppShell, MainWindow, …), or core
#      singleton headers (UserManager, AppConfig, SecretVault, …) are
#      boundary violations. Forbidden names are derived from the source tree
#      itself — there is no hand-maintained name list to drift out of sync.
#   2. Services must not reach up into the app shell (src/app/**).
#   3. UserManager::instance() chỉ được gọi ở composition root (AppShell.cpp)
#      và trong chính UserManager.{h,cpp}; mọi nơi khác phải resolve
#      IUserManager từ ServiceRegistry. AppConfig::bootstrap() tương tự: chỉ
#      Main.cpp / AppShell.cpp / Logger.cpp (bootstrap) và chính AppConfig.
#   4. Plugins must not call process singletons (AppConfig::instance /
#      AppConfig::bootstrap) — resolve IAppPaths from the ServiceRegistry
#      instead. LanguageManager không còn singleton; ILanguageService được
#      resolve từ registry.
#   5. Plugin nặng/nhẹ phải khớp với source của chính nó: plugin có include
#      VTK/OpenCV/PCL phải dùng SharedPCH + tự khai báo heavy libs; plugin
#      còn lại phải dùng SharedPCHLight và không được tham chiếu heavy SDK —
#      nếu không, "plugin nhỏ không kéo dependency nặng" lại bị phá vỡ trong
#      im lặng.
#   6. src/utils/** là bề mặt ABI dùng chung cho plugin nhẹ → không được
#      include heavy SDK (VTK/OpenCV/PCL). Code heavy thuộc src/services/*.
#   7. Teardown phải có biên: `waitForFinished()` trần (mặc định Qt chờ 30 s)
#      bị cấm trong src/** — luôn truyền timeout tường minh.
#   8. Plugin cài event filter lên widget KHÔNG thuộc nó (receiver không phải
#      this/m_*) phải removeEventFilter trong CÙNG file — nếu không shell gọi
#      vào code của DLL đã unload.
#   9. Plugin không được dò QApplication::topLevelWidgets() để tìm shell.
#  10. Plugin không được dùng pool nó không sở hữu (QtConcurrent::run,
#      QThreadPool::globalInstance()) — task có thể sống lâu hơn plugin.
#  11. Include path của plugin không được chứa src/core.
#  12. Mỗi lời gọi inference thô phải đi qua một exception guard.
#  13. Plugin không được ghi vào ServiceRegistry: IAppContext::services() trả
#      const ServiceRegistry* (chỉ get/getRequired/has). Write side
#      (MutableServiceRegistry) là PRIVATE của AppCore, composition root sở hữu.
#  14. IAIService không được phơi loader model đồng bộ: plugin kích hoạt load
#      qua IAIModelLoadService (worker AppCore sở hữu), loader thật nằm ở
#      IAIModelAdmin (src/services/ai, PRIVATE).
#
# Bypass-resistance: so sánh tên header là KHÔNG phân biệt hoa/thường (Windows
# filesystem cũng vậy) và include bằng <...> được kiểm tra y như "...".
#
# Chạy: cmake -DSOURCE_DIR=<repo-root> -P tests/CheckArchitectureBoundaries.cmake
# ──────────────────────────────────────────────────────────────────────────────

if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required (repo root)")
endif()

set(VIOLATIONS "")

# ── Normalisation helpers ─────────────────────────────────────────────────────
# Windows filesystems are case-insensitive, so `#include "sceneservice.h"`
# finds SceneService.h. A boundary check that compares case-sensitively is
# therefore bypassable by changing the case of one letter: every name set and
# every extracted basename is lower-cased before it is compared.
function(normalise_name out value)
    string(TOLOWER "${value}" lower)
    set(${out} "${lower}" PARENT_SCOPE)
endfunction()

# ── Name sets derived from the source tree (single source of truth) ───────────
file(GLOB_RECURSE SERVICE_HEADERS "${SOURCE_DIR}/src/services/*.h")
file(GLOB_RECURSE APP_HEADERS     "${SOURCE_DIR}/src/app/*.h")
file(GLOB_RECURSE CORE_HEADERS    "${SOURCE_DIR}/src/core/*.h")
file(GLOB_RECURSE PUBLIC_HEADERS  "${SOURCE_DIR}/src/include/*.h")
file(GLOB_RECURSE UTILS_HEADERS   "${SOURCE_DIR}/src/utils/*.h")

set(APP_BASENAMES "")
foreach(h IN LISTS APP_HEADERS)
    get_filename_component(base "${h}" NAME)
    normalise_name(base "${base}")
    list(APPEND APP_BASENAMES "${base}")
endforeach()

# The PUBLIC surface a plugin may include: src/include (the interfaces, the
# core glue they expose and the value types) plus the shared utility layer.
# This is the SAME boundary the compiler enforces: src/core is PRIVATE on
# AppCore, so a plugin cannot resolve a concrete header at all.
set(PUBLIC_BASENAMES "")
foreach(h IN LISTS PUBLIC_HEADERS)
    get_filename_component(base "${h}" NAME)
    normalise_name(base "${base}")
    list(APPEND PUBLIC_BASENAMES "${base}")
endforeach()

# Forbidden for plugins: every service/app header and every concrete core
# header (src/core).
set(FORBIDDEN_HEADERS "")
foreach(h IN LISTS SERVICE_HEADERS APP_HEADERS CORE_HEADERS)
    get_filename_component(base "${h}" NAME)
    normalise_name(base "${base}")
    list(APPEND FORBIDDEN_HEADERS "${base}")
endforeach()
list(REMOVE_DUPLICATES FORBIDDEN_HEADERS)

set(UTILS_BASENAMES "")
foreach(h IN LISTS UTILS_HEADERS)
    get_filename_component(base "${h}" NAME)
    normalise_name(base "${base}")
    list(APPEND UTILS_BASENAMES "${base}")
endforeach()


file(GLOB_RECURSE PLUGIN_SOURCES
    "${SOURCE_DIR}/src/modules/*.cpp"
    "${SOURCE_DIR}/src/modules/*.h")

# Rule 8 needs .h and .cpp alike (a filter may be installed from a header).
file(GLOB_RECURSE PLUGIN_ALL_SOURCES
    "${SOURCE_DIR}/src/modules/*.cpp"
    "${SOURCE_DIR}/src/modules/*.h")

# ── 1. Plugin → forbidden include ─────────────────────────────────────────────
foreach(source IN LISTS PLUGIN_SOURCES)
    file(READ "${source}" content)
    string(REGEX MATCHALL "#[ \t]*include[ \t]+[\"<][^\">]+[\">]" includes "${content}")
    foreach(inc IN LISTS includes)
        # Angle brackets are NOT skipped: `<SceneService.h>` resolves to the
        # same file as "SceneService.h", and skipping the form was a complete
        # bypass of this rule. A genuine system header (`<vector>`,
        # `<QString>`) is simply not in any of the name sets below, so it
        # falls through harmlessly.
        string(REGEX REPLACE ".*[\"<]([^\">]+)[\">].*" "\\1" path "${inc}")
        get_filename_component(base "${path}" NAME)
        # Case-insensitive on purpose: the filesystem is.
        normalise_name(base "${base}")

        if(base IN_LIST PUBLIC_BASENAMES)
            continue()   # the plugin-facing surface (src/include)
        endif()
        if(base IN_LIST UTILS_BASENAMES)
            continue()   # shared UI utility
        endif()
        if(base IN_LIST FORBIDDEN_HEADERS)
            file(RELATIVE_PATH rel "${SOURCE_DIR}" "${source}")
            list(APPEND VIOLATIONS
                "${rel}: include \"${path}\" — plugin chỉ được include header trong src/include (bề mặt công khai) + utils dùng chung, hoặc header nội bộ của plugin")
        endif()
        # Anything else = plugin-local header — allowed by the principle.
    endforeach()
endforeach()

# ── 2. Service → app-shell include ────────────────────────────────────────────
file(GLOB_RECURSE SERVICE_SOURCES
    "${SOURCE_DIR}/src/services/*.cpp"
    "${SOURCE_DIR}/src/services/*.h")

foreach(source IN LISTS SERVICE_SOURCES)
    file(READ "${source}" content)
    string(REGEX MATCHALL "#[ \t]*include[ \t]+[\"<][^\">]+[\">]" includes "${content}")
    foreach(inc IN LISTS includes)
        string(REGEX REPLACE ".*[\"<]([^\">]+)[\">].*" "\\1" path "${inc}")
        get_filename_component(base "${path}" NAME)
        normalise_name(base "${base}")
        if(base IN_LIST APP_BASENAMES)
            file(RELATIVE_PATH rel "${SOURCE_DIR}" "${source}")
            list(APPEND VIOLATIONS
                "${rel}: service include \"${path}\" — services không được phụ thuộc src/app (lục đục nguồn services → app)")
        endif()
    endforeach()
endforeach()


# ── 3. UserManager::instance() chỉ ở composition root ─────────────────────────
file(GLOB_RECURSE ALL_SOURCES
    "${SOURCE_DIR}/src/*.cpp"
    "${SOURCE_DIR}/src/*.h")

foreach(source IN LISTS ALL_SOURCES)
    get_filename_component(name "${source}" NAME)
    if(name STREQUAL "UserManager.cpp"
       OR name STREQUAL "UserManager.h"
       OR name STREQUAL "AppShell.cpp")
        continue()
    endif()

    file(READ "${source}" content)
    if(content MATCHES "UserManager::instance[ \t]*\\(\\)")
        file(RELATIVE_PATH rel "${SOURCE_DIR}" "${source}")
        list(APPEND VIOLATIONS
            "${rel}: gọi UserManager::instance() ngoài composition root")
    endif()
endforeach()

# ── 3b. AppConfig::bootstrap() chỉ ở bootstrap/composition root ───────────────
foreach(source IN LISTS ALL_SOURCES)
    get_filename_component(name "${source}" NAME)
    if(name STREQUAL "AppConfig.cpp"
       OR name STREQUAL "AppConfig.h"
       OR name STREQUAL "Main.cpp"
       OR name STREQUAL "AppShell.cpp"
       OR name STREQUAL "Logger.cpp")
        continue()
    endif()

    file(READ "${source}" content)
    if(content MATCHES "AppConfig::(bootstrap|instance)[ \t]*\\(\\)")
        file(RELATIVE_PATH rel "${SOURCE_DIR}" "${source}")
        list(APPEND VIOLATIONS
            "${rel}: gọi AppConfig::bootstrap()/instance() ngoài bootstrap — resolve IAppPaths")
    endif()
endforeach()

# ── 4. Plugins must not call process singletons ───────────────────────────────
foreach(source IN LISTS PLUGIN_SOURCES)
    file(READ "${source}" content)
    if(content MATCHES "(AppConfig|LanguageManager)::(instance|bootstrap)[ \t]*\\(\\)")
        file(RELATIVE_PATH rel "${SOURCE_DIR}" "${source}")
        list(APPEND VIOLATIONS
            "${rel}: gọi singleton ::instance()/::bootstrap() — plugin phải resolve service qua ServiceRegistry")
    endif()
endforeach()

# ── 5. Plugin nặng/nhẹ: PCH + third-party link phải khớp với source ───────────
# "Nặng" suy ra từ source (include trực tiếp VTK/OpenCV/PCL) — không có danh
# sách tay để lệch pha. Nặng → SharedPCH; nhẹ → SharedPCHLight và không được
# tham chiếu heavy SDK trong CMakeLists.
set(HEAVY_PLUGIN_NAMES "")
foreach(source IN LISTS PLUGIN_SOURCES)
    file(READ "${source}" content)
    if(content MATCHES "#[ \t]*include[ \t]+[<\"](opencv2/|pcl/|vtk|QVTK)")
        file(RELATIVE_PATH rel "${SOURCE_DIR}/src/modules" "${source}")
        string(REGEX REPLACE "/.*" "" plugin "${rel}")
        list(APPEND HEAVY_PLUGIN_NAMES "${plugin}")
    endif()
endforeach()
list(REMOVE_DUPLICATES HEAVY_PLUGIN_NAMES)

file(GLOB PLUGIN_CMAKELISTS "${SOURCE_DIR}/src/modules/*/CMakeLists.txt")
foreach(cml IN LISTS PLUGIN_CMAKELISTS)
    get_filename_component(plugin_dir "${cml}" DIRECTORY)
    get_filename_component(plugin "${plugin_dir}" NAME)
    file(READ "${cml}" content)
    file(RELATIVE_PATH rel "${SOURCE_DIR}" "${cml}")

    list(FIND HEAVY_PLUGIN_NAMES "${plugin}" heavy_idx)
    if(heavy_idx GREATER -1)
        if(content MATCHES "SharedPCHLight")
            list(APPEND VIOLATIONS
                "${rel}: plugin nặng (source include VTK/OpenCV/PCL) nhưng build với SharedPCHLight — phải REUSE_FROM SharedPCH")
        endif()
    else()
        # Both the CMake variable spelling (VTK_LIBRARIES) and the imported
        # target spelling (VTK::CommonCore, OpenCV::core, PCL::common,
        # Eigen3::Eigen) link the heavy SDK. A find_package() for one of them
        # belongs here too: it is what makes those targets exist.
        if(content MATCHES "(VTK_LIBRARIES|VTK_INCLUDE_DIRS|OpenCV_LIBS|OpenCV_INCLUDE_DIRS|PCL_LIBRARIES|PCL_INCLUDE_DIRS|onnxruntime|SharedPCH[^A-Za-z]|VTK::|OpenCV::|PCL::|Eigen3::|Eigen::|onnxruntime::|find_package[ \t]*\\([ \t]*(VTK|OpenCV|PCL|Eigen3|Eigen|onnxruntime))")
            list(APPEND VIOLATIONS
                "${rel}: plugin nhẹ tham chiếu heavy SDK/SharedPCH — phải dùng SharedPCHLight và không kéo VTK/OpenCV/PCL/ONNX")
        endif()
    endif()

    if(NOT content MATCHES "AppCore")
        list(APPEND VIOLATIONS
            "${rel}: plugin phải link AppCore (ABI chung interface + utils)")
    endif()
endforeach()

# ── 6. src/utils là bề mặt ABI dùng chung → không include heavy SDK ───────────
foreach(h IN LISTS UTILS_HEADERS)
    file(READ "${h}" content)
    if(content MATCHES "#[ \t]*include[ \t]+[<\"](opencv2/|pcl/|vtk|QVTK)")
        file(RELATIVE_PATH rel "${SOURCE_DIR}" "${h}")
        list(APPEND VIOLATIONS
            "${rel}: utils header include heavy SDK — plugin nhẹ include utils sẽ kéo theo; chuyển sang src/services/*")
    endif()
endforeach()

# ── 7. Teardown có biên: không wait()/waitForDone()/waitForFinished() trần ───
# Bản không tham số của Qt:
#   * waitForFinished()/waitForStarted() — mặc định 30 s (treo shutdown);
#   * QThread::wait() / QThreadPool::waitForDone() — mặc định -1 = CHỜ MÃI.
# Guard chặn phát sinh mới. Một ngoại lệ phải tự khai báo trên CÙNG DÒNG bằng
# `// teardown-unbounded: <lý do>` — allow-list theo đường dẫn sẽ âm thầm nuốt
# mọi lời gọi mới trong file đó, còn marker thì buộc người viết giải thích.
#
# NOTE (CMake regex): `\(` mở một NHÓM, không phải dấu ngoặc literal — dùng
# `[(]`/`[)]`. Và tiền tố `[.>]` bắt đúng lời gọi thành viên, nên câu comment
# kiểu "the argument-less wait() blocks" không bị báo nhầm.
file(GLOB_RECURSE ALL_CXX_SOURCES "${SOURCE_DIR}/src/*.cpp" "${SOURCE_DIR}/src/*.h")
foreach(source IN LISTS ALL_CXX_SOURCES)
    file(READ "${source}" _content)
    # `;` trong một dòng sẽ bị coi là dấu phân cách list; escape trước để số
    # dòng báo lỗi khớp với file thật.
    string(REPLACE ";" "\\;" _content "${_content}")
    string(REPLACE "\n" ";" _lines "${_content}")
    set(_line_no 0)
    foreach(_line IN LISTS _lines)
        math(EXPR _line_no "${_line_no} + 1")
        # `wait(` chỉ khớp lời gọi KHÔNG tham số: QWaitCondition::wait(&mutex)
        # và TaskTracker::waitForFinished(ms) không bị bắt.
        if(_line MATCHES "[.>][ \t]*waitForFinished[ \t]*[(][ \t]*[)]" OR
           _line MATCHES "[.>][ \t]*waitForStarted[ \t]*[(][ \t]*[)]" OR
           _line MATCHES "[.>][ \t]*waitForDone[ \t]*[(][ \t]*[)]" OR
           _line MATCHES "[.>][ \t]*wait[ \t]*[(][ \t]*[)]")
            if(NOT _line MATCHES "teardown-unbounded:[ \t]*[^ \t]")
                file(RELATIVE_PATH rel "${SOURCE_DIR}" "${source}")
                list(APPEND VIOLATIONS
                    "${rel}:${_line_no}: wait không biên — truyền timeout, hoặc khai báo ngoại lệ bằng '// teardown-unbounded: <lý do>'")
            endif()
        endif()
    endforeach()
endforeach()

# ── 8. Event filter: a plugin that installs one on a widget it does NOT own
#    must be able to remove it. The filter's vtable lives in the plugin DLL;
#    if the plugin cannot take it off the shell, the shell calls into the
#    plugin's code after the plugin is gone.
#
#    Ownership is decided by the RECEIVER of installEventFilter():
#      this->installEventFilter(...)      — the plugin's own object (self)
#      m_xxx->installEventFilter(...)     — a widget this object created
#    Both die with the plugin, so nothing has to be removed. Any other
#    receiver (a parameter, a shell service getter) is a widget the plugin
#    does not own: the removal must be visible in the SAME file.
#
#    The previous version was directory-scoped, so one removeEventFilter()
#    anywhere in the plugin's folder silenced the rule for every file in it —
#    which is exactly how a dropped filter handle passed this check.
foreach(source IN LISTS PLUGIN_ALL_SOURCES)
    file(READ "${source}" content)

    # Every X->installEventFilter(...) call, and the subset whose receiver is
    # this / a member of this object.
    string(REGEX MATCHALL "[A-Za-z_][A-Za-z0-9_]*[ \t]*->[ \t]*installEventFilter[ \t]*\\([^)]*\\)"
           receiver_calls "${content}")
    string(REGEX MATCHALL "(this|m_[A-Za-z0-9_]+)[ \t]*->[ \t]*installEventFilter[ \t]*\\([^)]*\\)"
           owned_receiver_calls "${content}")
    list(LENGTH receiver_calls receiver_count)
    list(LENGTH owned_receiver_calls owned_count)

    if(receiver_count GREATER owned_count
       AND NOT content MATCHES "removeEventFilter[ \t]*\\(")
        file(RELATIVE_PATH rel "${SOURCE_DIR}" "${source}")
        list(APPEND VIOLATIONS
            "${rel}: cài event filter lên widget KHÔNG thuộc plugin (receiver không phải this/m_*) mà không removeEventFilter trong cùng file — shell sẽ gọi vào code của plugin sau khi plugin đã đi")
    endif()
endforeach()

# ── 9. Plugin không được tự dò cửa sổ shell. IShellService là nguồn duy
#    nhất; quét QApplication::topLevelWidgets() có thể chọn nhầm cửa sổ.
foreach(source IN LISTS PLUGIN_SOURCES)
    file(READ "${source}" content)
    if(content MATCHES "topLevelWidgets[ \t]*\\(")
        file(RELATIVE_PATH rel "${SOURCE_DIR}" "${source}")
        list(APPEND VIOLATIONS
            "${rel}: dò QApplication::topLevelWidgets() để tìm cửa sổ shell — dùng IShellService::dialogParent()")
    endif()
endforeach()

# ── 10. Blocking / asynchronous work trong plugin: phải do owner quản lý ─────
#     QtConcurrent::run và QThreadPool::globalInstance() đều chạy trên pool mà
#     plugin KHÔNG sở hữu: task có thể chạy inline trên GUI thread, và nó vẫn
#     sống sau khi plugin bị unload — lúc đó nó đang thực thi code của DLL đã
#     đi. blocking::run dùng QThread riêng do caller join; việc nền của plugin
#     phải nằm sau một service AppCore sở hữu worker (IAIModelLoadService,
#     IAIPredictionLogService) hoặc một thread chính plugin join trong
#     destructor.
foreach(source IN LISTS PLUGIN_SOURCES)
    file(READ "${source}" content)
    if(content MATCHES "QtConcurrent[ \t]*::[ \t]*run")
        file(RELATIVE_PATH rel "${SOURCE_DIR}" "${source}")
        list(APPEND VIOLATIONS
            "${rel}: QtConcurrent::run cho blocking I/O — pool có thể chạy task inline trên GUI thread; dùng blocking::run")
    endif()
    if(content MATCHES "QThreadPool[ \t]*::[ \t]*globalInstance[ \t]*\\([ \t]*\\)")
        file(RELATIVE_PATH rel "${SOURCE_DIR}" "${source}")
        list(APPEND VIOLATIONS
            "${rel}: QThreadPool::globalInstance() — pool không do plugin sở hữu, task có thể sống sau khi plugin unload; chuyển việc nền sang service AppCore (IAIPredictionLogService/IAIModelLoadService) hoặc thread plugin tự join")
    endif()
endforeach()


# ── 11. Include path của plugin không được chứa src/core ─────────────────────
#     Rule 1 chỉ soi mã nguồn; include path mới là thứ compiler thật sự dùng.
#     Từ khi bề mặt công khai là src/include, việc thêm src/core vào include
#     path của plugin là đưa được header lõi (UserManager.h, SecretVault.h…)
#     vào tầm với của nó — đúng thứ ranh giới này cấm. Không có rule này thì
#     một dòng include path có thể vô hiệu hoá toàn bộ rule 1.
file(GLOB PLUGIN_CMAKE_FILES "${SOURCE_DIR}/src/modules/*/CMakeLists.txt")
foreach(cml IN LISTS PLUGIN_CMAKE_FILES)
    file(READ "${cml}" content)
    # Bỏ comment: một dòng giải thích nhắc tới src/core không phải vi phạm.
    string(REGEX REPLACE "#[^\n]*" "" content "${content}")
    if(content MATCHES "src/core")
        file(RELATIVE_PATH rel "${SOURCE_DIR}" "${cml}")
        list(APPEND VIOLATIONS
            "${rel}: include path của plugin chứa src/core — bề mặt công khai là src/include; header lõi (UserManager.h, SecretVault.h…) là PRIVATE của AppCore")
    endif()
endforeach()


# ── 12. Mỗi lời gọi inference thô phải đi qua MỘT exception guard ────────────
#     AIService::runDetection / runSegmentation / runTracking / detectObjects
#     gọi thẳng vào AIProcessor, nơi ONNX Runtime và OpenCV ném exception. Một
#     exception thoát ra khỏi một Qt slot (GUI thread) hoặc QThread::run()
#     (background thread) sẽ tới std::terminate() và GIẾT CẢ ỨNG DỤNG.
#     Vì vậy mọi lời gọi thô phải bọc trong ai::runInferenceGuarded /
#     ai::runDetectionGuarded.
#
#     Luật: số lời gọi thô phải BẰNG số lời gọi guard — mỗi lời gọi thô có
#     đúng một guard. Thêm một entry point inference mà quên guard ⇒ số lời
#     gọi thô vượt số guard ⇒ vi phạm bị bắt. Đây là câu trả lời cấu trúc cho
#     "guard chỉ được dán vào một thể hiện của lớp lỗi, còn anh em thì không".
set(_ai_service "${SOURCE_DIR}/src/services/ai/AIService.cpp")
if(EXISTS "${_ai_service}")
    file(READ "${_ai_service}" _ai_content)
    string(REGEX MATCHALL "m_processor->(runTracking|runSegmentation|runObjectDetection|detectObjects)[ \t]*\\(" _raw_calls "${_ai_content}")
    string(REGEX MATCHALL "(runInferenceGuarded|runDetectionGuarded)[ \t]*\\(" _guard_calls "${_ai_content}")
    list(LENGTH _raw_calls _raw_count)
    list(LENGTH _guard_calls _guard_count)
    if(NOT _raw_count EQUAL _guard_count)
        list(APPEND VIOLATIONS
            "src/services/ai/AIService.cpp: ${_raw_count} lời gọi inference thô nhưng chỉ ${_guard_count} guard — mỗi lời gọi thô phải có một ai::runInferenceGuarded/runDetectionGuarded (exception ONNX/OpenCV nếu thoát ra sẽ giết app)")
    endif()
endif()


# ── 13. Plugin không được ghi vào ServiceRegistry ────────────────────────────
#     services() trả const ServiceRegistry* nên registerService/unregister/
#     clear không gọi được từ plugin (lỗi biên dịch). Rule này chặn đường
#     vòng const_cast và giữ cho hợp đồng "chỉ composition root được đăng ký"
#     không bị nới ra trong im lặng.
foreach(source IN LISTS PLUGIN_SOURCES)
    file(READ "${source}" content)
    if(content MATCHES "services[ \t]*\\([ \t]*\\)[ \t]*->[ \t]*(registerService|unregister|clear)[ \t]*[<(]")
        file(RELATIVE_PATH rel "${SOURCE_DIR}" "${source}")
        list(APPEND VIOLATIONS
            "${rel}: ghi vào ServiceRegistry từ plugin — IAppContext::services() là read-only; write side (MutableServiceRegistry) thuộc composition root")
    endif()
    if(content MATCHES "const_cast[ \t]*<[ \t]*ServiceRegistry")
        file(RELATIVE_PATH rel "${SOURCE_DIR}" "${source}")
        list(APPEND VIOLATIONS
            "${rel}: const_cast<ServiceRegistry> — plugin không được lấy lại quyền ghi registry")
    endif()
endforeach()


# ── 14. IAIService không được phơi loader model đồng bộ ──────────────────────
#     Loader ONNX là blocking; một slot GUI gọi thẳng sẽ đóng băng cửa sổ.
#     Plugin chỉ được trigger qua IAIModelLoadService và đọc is*Ready().
set(_iai_header "${SOURCE_DIR}/src/include/IAIService.h")
if(EXISTS "${_iai_header}")
    file(READ "${_iai_header}" _iai_content)
    if(_iai_content MATCHES "load(Detection|Segmentation|Tracking)Model")
        list(APPEND VIOLATIONS
            "src/include/IAIService.h: loader model đồng bộ lộ ra plugin — chuyển sang IAIModelAdmin (src/services/ai) và IAIModelLoadService")
    endif()
endif()


if(VIOLATIONS)
    list(JOIN VIOLATIONS "\n  - " report)
    message(FATAL_ERROR "Architecture boundary violations:\n  - ${report}")
endif()

message(STATUS "Architecture boundaries OK")
