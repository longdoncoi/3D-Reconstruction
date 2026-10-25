find_package(Qt6 REQUIRED COMPONENTS Test Core)
set(CMAKE_AUTOMOC ON)

# PCL is looked up first: its config resolves the VTK installation the
# project was built against and therefore fills VTK_INCLUDE_DIRS for the
# targets below (VTK_INCLUDE_DIRS stays empty otherwise).
find_package(PCL REQUIRED COMPONENTS common filters io)
find_package(OpenCV REQUIRED COMPONENTS core)

# Sources compiled into more than one test target. Keeping them in one variable
# is what stops the two lists from drifting: a target that still lists a file
# keeps passing while the other silently stops measuring it.
set(APP_SHELL_SOURCES
    ../src/app/AppShell.cpp
    ../src/app/AppShell.h
)
set(SENDMAIL_HELPER_SOURCES
    ../src/modules/SendMailPlugin/MailMessageFormatter.cpp
    ../src/modules/SendMailPlugin/MailInboxItemFactory.cpp
)
set(VIDEO_TRACKER_SOURCES
    ../src/modules/AIProcessorPlugin/VideoTrackerThread.cpp
)

# Architecture guard — plugin boundaries + UserManager singleton usage.
# Encodes the invariants documented in AGENTS.md; runs in milliseconds.
add_test(NAME TestArchitectureBoundaries
         COMMAND ${CMAKE_COMMAND}
                 "-DSOURCE_DIR=${CMAKE_SOURCE_DIR}"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckArchitectureBoundaries.cmake")

# Interface ABI guard — the generated hash must be derived from the CURRENT
# plugin-facing headers (freshness) and must react to a change of any of them
# (sensitivity). A stale hash would let a plugin built against different
# interfaces load.
add_test(NAME TestInterfaceAbi
         COMMAND ${CMAKE_COMMAND}
                 "-DSOURCE_DIR=${CMAKE_SOURCE_DIR}"
                 "-DINTERFACE_ABI_HEADER=${APP_INTERFACE_ABI_HEADER}"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckInterfaceAbi.cmake")
set_tests_properties(TestInterfaceAbi PROPERTIES
                     REQUIRED_FILES "${APP_INTERFACE_ABI_HEADER}")

# Interface ABI incremental write guard — an unchanged hash must NOT rewrite
# the generated header (its timestamp would rebuild every target), while a
# tampered header must still be repaired.
add_test(NAME TestInterfaceAbiIncremental
         COMMAND ${CMAKE_COMMAND}
                 "-DSOURCE_DIR=${CMAKE_SOURCE_DIR}"
                 "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/interface_abi_incremental"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckInterfaceAbiIncremental.cmake")

# Documentation-consistency guard — every machine-checkable number in the
# architecture docs (ABI surface, PINNED_HEADERS, MIN_MUTANTS, CTest count,
# coverage snapshot) is re-derived from the tree, so prose cannot drift from
# the code it claims to describe.
add_test(NAME TestDocs
         COMMAND ${CMAKE_COMMAND}
                 "-DSOURCE_DIR=${CMAKE_SOURCE_DIR}"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckDocs.cmake")

# Agent-action contract — every action advertised in
# Config/agent_action_manifest.json must be handled by exactly one plugin, and
# every plugin handler must exist in the manifest (no dead handler).
add_test(NAME TestAgentActionsContract
         COMMAND ${CMAKE_COMMAND}
                 "-DSOURCE_DIR=${CMAKE_SOURCE_DIR}"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAgentActions.cmake")

# Deploy guard — the ~4.1 GB runtime deploy must decide with stat() only:
# copy when missing/size-differs/source-newer, never rewrite an up-to-date file.
add_test(NAME TestDeployRuntime
         COMMAND ${CMAKE_COMMAND}
                 "-DSCRIPT=${CMAKE_SOURCE_DIR}/cmake/DeployRuntime.cmake"
                 "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/deploy_runtime_check"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckDeployRuntime.cmake")

# Architecture boundary test (post-ADR-001..004)
qt_add_executable(TestArchitectureBoundary TestArchitectureBoundary.cpp)
target_include_directories(TestArchitectureBoundary PRIVATE
    ../src/include
    ../src
    ../tests/core/mock
)
target_link_libraries(TestArchitectureBoundary PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestArchitectureBoundary COMMAND TestArchitectureBoundary)

# Test ServiceRegistry
qt_add_executable(TestServiceRegistry TestServiceRegistry.cpp)
# MutableServiceRegistry is the composition-root write side and lives in
# src/core (PRIVATE on AppCore) — the test asks for it explicitly.
target_include_directories(TestServiceRegistry PRIVATE ../src/include ../src/core)
target_link_libraries(TestServiceRegistry PRIVATE Qt6::Test Qt6::Core)
add_test(NAME TestServiceRegistry COMMAND TestServiceRegistry)

# Global exception barrier — a slot that throws must not abort the process.
# Its own main(): the subject under test is the QApplication subclass, while
# QTEST_MAIN would construct a plain QApplication.
qt_add_executable(TestGuardedApplication TestGuardedApplication.cpp)
target_include_directories(TestGuardedApplication PRIVATE ../src/app)
target_link_libraries(TestGuardedApplication PRIVATE Qt6::Widgets)
add_test(NAME TestGuardedApplication COMMAND TestGuardedApplication)

# Test PluginDependencyResolver — thứ tự init plugin (deps graph + phase + cycle)
qt_add_executable(TestPluginDependencyResolver TestPluginDependencyResolver.cpp)
target_include_directories(TestPluginDependencyResolver PRIVATE
    ../src/include
    ../src/core
)
target_link_libraries(TestPluginDependencyResolver PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestPluginDependencyResolver COMMAND TestPluginDependencyResolver)

# Test PluginHost — vòng đời load/resolve/init/cleanup (loader giả, không cần DLL thật)
qt_add_executable(TestPluginHost TestPluginHost.cpp)
target_link_libraries(TestPluginHost PRIVATE Qt6::Test Qt6::Core AppCore)
# The concrete kernel is PRIVATE on AppCore (src/include is the public
# surface) — a test of the kernel has to ask for src/core explicitly.
target_include_directories(TestPluginHost PRIVATE ../src/core)
add_test(NAME TestPluginHost COMMAND TestPluginHost)

# Test AIModelLoadService — AppCore-owned worker lifetime/cancellation contract.
qt_add_executable(TestAIModelLoadService TestAIModelLoadService.cpp)
target_include_directories(TestAIModelLoadService PRIVATE
    ../src/include
    ../src/services/ai
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(TestAIModelLoadService PRIVATE
    Qt6::Test Qt6::Core ${OpenCV_LIBS} AppCore)
add_test(NAME TestAIModelLoadService COMMAND TestAIModelLoadService)

# Test AIService — the AI service against the REAL ONNX models (Git LFS).
# The models are the ones the training pipeline publishes; without them the
# test skips and the coverage floor of aiservice.cpp is not met (the C++ CI
# enables `lfs: true`).
qt_add_executable(TestAIService TestAIService.cpp)
target_include_directories(TestAIService PRIVATE
    ../src/include
    ../src/services/ai
    ${OpenCV_INCLUDE_DIRS}
    ${onnxruntime_SOURCE_DIR}/include)
target_compile_definitions(TestAIService PRIVATE
    "AI_MODELS_DIR=\"${CMAKE_SOURCE_DIR}/AIComputerVision/Models\"")
target_link_libraries(TestAIService PRIVATE
    Qt6::Test Qt6::Core ${OpenCV_LIBS} AppCore
    "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib")
add_test(NAME TestAIService COMMAND TestAIService)
# The ONNX Runtime DLL must sit next to the test executable: Windows would
# otherwise load an unrelated onnxruntime.dll found on PATH (the same trap the
# application target guards against).
add_custom_command(TARGET TestAIService POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll"
        "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_shared.dll"
        "$<TARGET_FILE_DIR:TestAIService>"
    COMMENT "Copying ONNX Runtime DLLs next to the test"
)

# Test AIServerProcessController — non-blocking stop() + deferred start(). The
# test executable re-runs itself as a controllable stand-in for the Python
# wrapper (see TestAIServerProcessController.cpp), so it needs no server.
qt_add_executable(TestAIServerProcessController TestAIServerProcessController.cpp)
target_include_directories(TestAIServerProcessController PRIVATE
    ../src/include
    ../src/services/ai)
target_link_libraries(TestAIServerProcessController PRIVATE
    Qt6::Test Qt6::Core AppCore)
add_test(NAME TestAIServerProcessController COMMAND TestAIServerProcessController)

# Smoke test on real plugin DLLs — must run after the plugin targets exist.
qt_add_executable(TestPluginSmoke TestPluginSmoke.cpp)
# The stub context mirrors ISceneService, whose interface speaks VTK types.
# VTK_INCLUDE_DIRS is only filled in by find_package(PCL) further down, so the
# imported targets are used here instead of the raw variable.
target_include_directories(TestPluginSmoke PRIVATE
    ../src/include
    ../src
    ../tests/core
    ${OpenCV_INCLUDE_DIRS}
    ../src/core
)
target_link_libraries(TestPluginSmoke PRIVATE Qt6::Test Qt6::Core Qt6::Widgets ${OpenCV_LIBS} AppCore VTK::CommonCore VTK::GUISupportQt)
target_compile_definitions(TestPluginSmoke PRIVATE
    PLUGIN_SMOKE_DIR="$<TARGET_FILE_DIR:ViewerPlugin>")
add_dependencies(TestPluginSmoke
    ViewerPlugin ReconstructionPlugin AIProcessorPlugin
    AIAssistantPlugin SendMailPlugin UserAuthPlugin)
add_test(NAME TestPluginSmoke COMMAND TestPluginSmoke)
# QTEST_MAIN → QApplication; CI không có desktop → offscreen platform plugin.
set_tests_properties(TestPluginSmoke PROPERTIES
    ENVIRONMENT_MODIFICATION "QT_QPA_PLATFORM=set:offscreen")

# Test SignalBus
qt_add_executable(TestSignalBus TestSignalBus.cpp)
target_link_libraries(TestSignalBus PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestSignalBus COMMAND TestSignalBus)

# Test AppConfig
qt_add_executable(TestAppConfig TestAppConfig.cpp)
target_link_libraries(TestAppConfig PRIVATE Qt6::Test Qt6::Core AppCore)
# The concrete kernel is PRIVATE on AppCore (src/include is the public
# surface) — a test of the kernel has to ask for src/core explicitly.
target_include_directories(TestAppConfig PRIVATE ../src/core)
add_test(NAME TestAppConfig COMMAND TestAppConfig)

# Test LanguageManager
qt_add_executable(TestLanguageManager TestLanguageManager.cpp)
target_link_libraries(TestLanguageManager PRIVATE Qt6::Test Qt6::Core AppCore)
# The concrete kernel is PRIVATE on AppCore (src/include is the public
# surface) — a test of the kernel has to ask for src/core explicitly.
target_include_directories(TestLanguageManager PRIVATE ../src/core)
add_test(NAME TestLanguageManager COMMAND TestLanguageManager)

# Test Mail MIME parser
qt_add_executable(TestMailMimeParser
    TestMailMimeParser.cpp
    ../src/services/mail/MailMimeParser.cpp
)
target_include_directories(TestMailMimeParser PRIVATE
    ../src/include
    ../src/services/mail
)
target_link_libraries(TestMailMimeParser PRIVATE Qt6::Test Qt6::Core)
add_test(NAME TestMailMimeParser COMMAND TestMailMimeParser)
# Subject is a source file → keep it out of the unity TU (see TestCrosshairGeometry).
set_target_properties(TestMailMimeParser PROPERTIES UNITY_BUILD OFF)

# Test Camera Params Parser
find_package(OpenCV REQUIRED COMPONENTS core)
qt_add_executable(TestCameraParamsParser
    TestCameraParamsParser.cpp
    ../src/services/reconstruction/CameraParamsParser.cpp
)
target_include_directories(TestCameraParamsParser PRIVATE
    ../src/include
    ../src/services/reconstruction
)
target_link_libraries(TestCameraParamsParser PRIVATE Qt6::Test Qt6::Core ${OpenCV_LIBS})
add_test(NAME TestCameraParamsParser COMMAND TestCameraParamsParser)
# Subject is a source file → keep it out of the unity TU (see TestCrosshairGeometry).
set_target_properties(TestCameraParamsParser PROPERTIES UNITY_BUILD OFF)

# Test PointCloudFilter
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d)

qt_add_executable(TestPointCloudFilter TestPointCloudFilter.cpp ../src/services/reconstruction/PointCloudFilter.cpp)
target_include_directories(TestPointCloudFilter PRIVATE
    ../src/include
    ../src/services/reconstruction
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(TestPointCloudFilter PRIVATE 
    Qt6::Test Qt6::Core Qt6::Widgets
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
)
add_test(NAME TestPointCloudFilter COMMAND TestPointCloudFilter)
# Subject is a source file → keep it out of the unity TU (see TestCrosshairGeometry).
set_target_properties(TestPointCloudFilter PROPERTIES UNITY_BUILD OFF)

# Source dùng chung cho các test integration của reconstruction pipeline.
set(RECON_PIPELINE_SOURCES
    ../src/services/reconstruction/ReconstructionPipeline.cpp
    ../src/services/reconstruction/ReconstructionPostProcessor.cpp
    ../src/services/reconstruction/ReconstructionService.cpp
    ../src/services/reconstruction/ReconstructionWorker.cpp
    ../src/include/IReconstructionService.h
    ../src/services/reconstruction/SfmStage.cpp
    ../src/services/reconstruction/TrackSfM.cpp
    ../src/services/reconstruction/IncrementalSfM.cpp
    ../src/services/reconstruction/CameraParamsParser.cpp
    ../src/services/reconstruction/FeatureExtractor.cpp
    ../src/services/reconstruction/FeatureMatcher.cpp
    ../src/services/reconstruction/PoseEstimator.cpp
    ../src/services/reconstruction/Triangulator.cpp
    ../src/services/reconstruction/PointCloudFilter.cpp
    ../src/services/reconstruction/FeatureTrack.h
    ../src/services/reconstruction/MultiViewTriangulator.cpp
)

# Test ReconstructionPipeline (nhanh — không chạy hậu kỳ nặng)
qt_add_executable(TestReconstructionPipeline TestReconstructionPipeline.cpp
    ${RECON_PIPELINE_SOURCES}
)
target_include_directories(TestReconstructionPipeline PRIVATE
    ../src/services/reconstruction
    ../src/include
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(TestReconstructionPipeline PRIVATE 
    Qt6::Test Qt6::Core Qt6::Widgets Qt6::Concurrent
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
)
target_compile_definitions(TestReconstructionPipeline PRIVATE APP_LIBRARY)
add_test(NAME TestReconstructionPipeline COMMAND TestReconstructionPipeline)

# Test ReconstructionService — the SERVICE half of the reconstruction
# contract (the boundary the UI talks to): the UI-facing queries and
# mutators must not wait for the worker, and the published result must be
# complete after a run. It is a separate executable because the completed-run
# case needs a full reconstruction, and the heavy post-processing leaves
# several gigabytes behind — a second run in the same process would push it
# over the per-function timeout.
qt_add_executable(TestReconstructionService TestReconstructionService.cpp
    ${RECON_PIPELINE_SOURCES}
)
target_include_directories(TestReconstructionService PRIVATE
    ../src/services/reconstruction
    ../src/include
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(TestReconstructionService PRIVATE
    Qt6::Test Qt6::Core Qt6::Widgets Qt6::Concurrent
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
)
target_compile_definitions(TestReconstructionService PRIVATE APP_LIBRARY)
add_test(NAME TestReconstructionService COMMAND TestReconstructionService)
# The completed-run case needs a full reconstruction (MLS densify to 1M points
# + Poisson depth 10) — ~100 s in Release, and more on a slower machine. It
# gets the same treatment as TestReconstructionFinalize: a generous timeout
# and the "slow" label, so `ctest -LE slow` stays fast for the dev loop.
set_tests_properties(TestReconstructionService PROPERTIES TIMEOUT 900 LABELS slow)

# Test ReconstructionFinalize — hậu kỳ nặng (MLS densify 1M + Poisson depth 10
# + texture mesh). ~100s trong Release nên gắn label "slow": dev chạy
# `ctest -LE slow` khi cần vòng lặp nhanh, CI vẫn chạy mặc định.
qt_add_executable(TestReconstructionFinalize TestReconstructionFinalize.cpp
    ${RECON_PIPELINE_SOURCES}
)
target_include_directories(TestReconstructionFinalize PRIVATE
    ../src/services/reconstruction
    ../src/include
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(TestReconstructionFinalize PRIVATE
    Qt6::Test Qt6::Core Qt6::Widgets Qt6::Concurrent
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
)
target_compile_definitions(TestReconstructionFinalize PRIVATE APP_LIBRARY)
add_test(NAME TestReconstructionFinalize COMMAND TestReconstructionFinalize)
set_tests_properties(TestReconstructionFinalize PROPERTIES TIMEOUT 900 LABELS slow)

# Test ShellService — IShellService contract (menu cache, tab panels, progress dialog)
qt_add_executable(TestShellService TestShellService.cpp)
target_include_directories(TestShellService PRIVATE
    ../src/include
    ../src/services/shell
)
target_link_libraries(TestShellService PRIVATE Qt6::Test Qt6::Core Qt6::Widgets AppCore)
add_test(NAME TestShellService COMMAND TestShellService)
# QTEST_MAIN → QApplication; CI không có desktop → offscreen platform plugin.
set_tests_properties(TestShellService PROPERTIES
    ENVIRONMENT_MODIFICATION "QT_QPA_PLATFORM=set:offscreen")

# Test AppShell — composition-root wiring: registration phases, getter ↔
# registry identity, deferred idempotence, clean teardown + recomposition.
# AppShell lives in the exe target, so the test compiles it directly (same
# pattern as RECON_PIPELINE_SOURCES) and links AppCore + VTK itself.
qt_add_executable(TestAppShell
    TestAppShell.cpp
    ${APP_SHELL_SOURCES}
)
target_include_directories(TestAppShell PRIVATE
    ../src/include
    ../src/app
    ../src/services/scene
    ../src/services/viewer
    ../src/services/settings
    ../src/services/shell
    ../src/services/reconstruction
    ../src/services/ai
    ../src/services/mail
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${onnxruntime_SOURCE_DIR}/include
    ../src/core
)
target_compile_definitions(TestAppShell PRIVATE
    ${VTK_DEFINITIONS}
    ${PCL_DEFINITIONS}
    PCL_SILENCE_MALLOC_WARNING=1
)
target_link_libraries(TestAppShell PRIVATE
    Qt6::Test Qt6::Core Qt6::Widgets
    AppCore
    ${VTK_LIBRARIES}
    ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib
)
add_test(NAME TestAppShell COMMAND TestAppShell)
# QTEST_MAIN → QApplication; CI không có desktop → offscreen platform plugin.
set_tests_properties(TestAppShell PROPERTIES
    ENVIRONMENT_MODIFICATION "QT_QPA_PLATFORM=set:offscreen")

# Test MainWindow — the shell's tab-panel contract (the panels a plugin looks
# up by tab id), the translated labels, the menu bar and the shutdown path.
# MainWindow also lives in the exe target: compile it (and its .ui) into the
# test, exactly like TestAppShell does for AppShell. AUTOUIC is on globally,
# and CMAKE_AUTOUIC_SEARCH_PATHS already contains src/app.
qt_add_executable(TestMainWindow
    TestMainWindow.cpp
    ../src/app/MainWindow.cpp
    ../src/app/MainWindow.h
    ../src/app/MainWindow.ui
    ${APP_SHELL_SOURCES}
)
target_include_directories(TestMainWindow PRIVATE
    ../src/include
    ../src/app
    ../src/core
    ../src/services/scene
    ../src/services/viewer
    ../src/services/settings
    ../src/services/shell
    ../src/services/reconstruction
    ../src/services/ai
    ../src/services/mail
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${onnxruntime_SOURCE_DIR}/include
)
target_compile_definitions(TestMainWindow PRIVATE
    ${VTK_DEFINITIONS}
    ${PCL_DEFINITIONS}
    PCL_SILENCE_MALLOC_WARNING=1
)
target_link_libraries(TestMainWindow PRIVATE
    Qt6::Test Qt6::Core Qt6::Widgets Qt6::OpenGLWidgets
    AppCore
    ${VTK_LIBRARIES}
    ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib
)
add_test(NAME TestMainWindow COMMAND TestMainWindow)
set_tests_properties(TestMainWindow PROPERTIES
    ENVIRONMENT_MODIFICATION "QT_QPA_PLATFORM=set:offscreen")

# Test Agent Actions
qt_add_executable(TestAgentActions TestAgentActions.cpp)
target_include_directories(TestAgentActions PRIVATE
    ../src/include
    ../src/services/ai
)
target_link_libraries(TestAgentActions PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestAgentActions COMMAND TestAgentActions)

# Test ChatSessionManager — session list/current pointer/attachment cleanup/persistence
qt_add_executable(TestChatSessionManager TestChatSessionManager.cpp)
target_include_directories(TestChatSessionManager PRIVATE
    ../src/include
    ../src/services/ai
    ../src/core
)
target_link_libraries(TestChatSessionManager PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestChatSessionManager COMMAND TestChatSessionManager)

# Test AtRestCipher — the AES-256-GCM envelope shared with the Python service:
# byte layout, associated data, key resolution and the fail-closed policy.
qt_add_executable(TestAtRestCipher TestAtRestCipher.cpp)
target_include_directories(TestAtRestCipher PRIVATE
    ../src/include
    ../src/core
)
target_link_libraries(TestAtRestCipher PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestAtRestCipher COMMAND TestAtRestCipher)

# Test AIAssistant — request state machine + injected transport (no sockets,
# no Python child process; the ready transition is driven via the protected seam).
qt_add_executable(TestAIAssistant TestAIAssistant.cpp)
target_include_directories(TestAIAssistant PRIVATE
    ../src/include
    ../src/services/ai
)
target_link_libraries(TestAIAssistant PRIVATE Qt6::Test Qt6::Core Qt6::Network AppCore)
add_test(NAME TestAIAssistant COMMAND TestAIAssistant)

# Test AIRequestQueue — pending/queued bookkeeping without HTTP or a process.
qt_add_executable(TestAIRequestQueue TestAIRequestQueue.cpp)
target_include_directories(TestAIRequestQueue PRIVATE
    ../src/include
    ../src/services/ai
)
target_link_libraries(TestAIRequestQueue PRIVATE Qt6::Test Qt6::Core Qt6::Network AppCore)
add_test(NAME TestAIRequestQueue COMMAND TestAIRequestQueue)

# Test AI Yolo PostProcessor
qt_add_executable(TestAIYoloPostProcessor
    TestAIYoloPostProcessor.cpp
    ../src/services/ai/AIYoloPostProcessor.cpp
)
target_include_directories(TestAIYoloPostProcessor PRIVATE
    ../src/include
    ../src/services/ai
)
target_link_libraries(TestAIYoloPostProcessor PRIVATE Qt6::Test Qt6::Core ${OpenCV_LIBS})
add_test(NAME TestAIYoloPostProcessor COMMAND TestAIYoloPostProcessor)
# Subject is a source file → keep it out of the unity TU (see TestCrosshairGeometry).
set_target_properties(TestAIYoloPostProcessor PROPERTIES UNITY_BUILD OFF)

# Test UserManager crypto (PBKDF2 password hashing + SecretVault)
qt_add_executable(TestUserManagerCrypto TestUserManagerCrypto.cpp)
target_link_libraries(TestUserManagerCrypto PRIVATE Qt6::Test Qt6::Core AppCore)
# The concrete kernel is PRIVATE on AppCore (src/include is the public
# surface) — a test of the kernel has to ask for src/core explicitly.
target_include_directories(TestUserManagerCrypto PRIVATE ../src/core)
if(WIN32)
    target_link_libraries(TestUserManagerCrypto PRIVATE crypt32)
endif()
add_test(NAME TestUserManagerCrypto COMMAND TestUserManagerCrypto)

# Test MailService credential lifecycle (cache invalidation on logout — no network)
qt_add_executable(TestMailCredentialLifecycle TestMailCredentialLifecycle.cpp)
target_include_directories(TestMailCredentialLifecycle PRIVATE
    ../src/include
    ../src/services/mail
    ../src/core
)
target_link_libraries(TestMailCredentialLifecycle PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestMailCredentialLifecycle COMMAND TestMailCredentialLifecycle)

# Test SmtpOutbox (IOutgoingMail adapter — validation branches, no network)
qt_add_executable(TestSmtpOutbox TestSmtpOutbox.cpp)
target_include_directories(TestSmtpOutbox PRIVATE
    ../src/include
    ../src/services/mail
    ../src/core
)
target_link_libraries(TestSmtpOutbox PRIVATE Qt6::Test Qt6::Core AppCore)
if(WIN32)
    target_link_libraries(TestSmtpOutbox PRIVATE crypt32)
endif()
add_test(NAME TestSmtpOutbox COMMAND TestSmtpOutbox)

# Test SettingsService (injected config path — never touches real Config.ini)
qt_add_executable(TestSettingsService TestSettingsService.cpp)
target_include_directories(TestSettingsService PRIVATE
    ../src/include
    ../src/services/settings
)
target_link_libraries(TestSettingsService PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestSettingsService COMMAND TestSettingsService)

# Test CrosshairGeometry (pure DICOM-slice math used by SceneService;
# translation unit has no VTK dependency after ORI_* moved to the pure header)
qt_add_executable(TestCrosshairGeometry
    TestCrosshairGeometry.cpp
    ../src/services/scene/CrosshairGeometry.cpp
)
target_include_directories(TestCrosshairGeometry PRIVATE
    ../src/include
    ../src/services/scene
)
target_link_libraries(TestCrosshairGeometry PRIVATE Qt6::Test Qt6::Core)
add_test(NAME TestCrosshairGeometry COMMAND TestCrosshairGeometry)

# ── Unity build must stay off for targets whose *subject* is a source file ──
# The coverage gate maps executed lines through the PDB.  When the subject is
# compiled into the same unity translation unit as the test, MSVC inlines it
# away, no code is emitted for it, the PDB does not mention the file and the
# exporter reports an empty class — the file silently measures 0 %.  (This is
# exactly what happened to CrosshairGeometry.cpp.)  A separate translation
# unit keeps the subject measurable.
set_target_properties(TestCrosshairGeometry PROPERTIES UNITY_BUILD OFF)

# Test PluginHelpers — the plugin helpers that carry real logic: the mail
# preview renderer, the inbox row factory, the attachment preview factory and
# the prediction logger.  They are compiled into the test (the plugins do not
# export them) and kept out of the unity build so the exporter can map them.
qt_add_executable(TestPluginHelpers
    TestPluginHelpers.cpp
    ${SENDMAIL_HELPER_SOURCES}
    ../src/modules/AIAssistantPlugin/AIAttachmentPreviewFactory.cpp
)
target_include_directories(TestPluginHelpers PRIVATE
    ../src/include
    ../src/utils
    ../src/services/ai
    ../src/modules/SendMailPlugin
    ../src/modules/AIAssistantPlugin
    ../src/modules/AIProcessorPlugin
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(TestPluginHelpers PRIVATE
    Qt6::Test Qt6::Core Qt6::Widgets
    ${OpenCV_LIBS}
    AppCore
)
add_test(NAME TestPluginHelpers COMMAND TestPluginHelpers)
# Subject is a source file → keep it out of the unity TU (see TestCrosshairGeometry).
set_target_properties(TestPluginHelpers PROPERTIES UNITY_BUILD OFF)

# Test UserAuthDialogs — the authentication dialogs of the user-auth plugin
# (≈1 000 lines of GUI that had no test).  They read the user through the
# plugin's statics, so the test initializes the plugin with the stub context
# first and builds the dialogs afterwards.
qt_add_executable(TestUserAuthDialogs
    TestUserAuthDialogs.cpp
    ../src/modules/UserAuthPlugin/UserAuthPlugin.cpp
    ../src/modules/UserAuthPlugin/AboutDialog.cpp
    ../src/modules/UserAuthPlugin/AdminUserManagerDialog.cpp
    ../src/modules/UserAuthPlugin/AvatarCropperDialog.cpp
    ../src/modules/UserAuthPlugin/ChangePasswordDialog.cpp
    ../src/modules/UserAuthPlugin/ForgotPasswordDialog.cpp
    ../src/modules/UserAuthPlugin/LicenseActivationDialog.cpp
    ../src/modules/UserAuthPlugin/LoginDialog.cpp
    ../src/modules/UserAuthPlugin/SettingsDialog.cpp
    ../src/modules/UserAuthPlugin/ThemeSelectionDialog.cpp
    ../src/modules/UserAuthPlugin/OverlayResizer.cpp
)
target_include_directories(TestUserAuthDialogs PRIVATE
    ../src
    ../src/include
    ../src/utils
    ../src/services/scene
    ../src/services/viewer
    ../src/services/settings
    ../src/services/shell
    ../src/modules/UserAuthPlugin
    ../tests/core
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(TestUserAuthDialogs PRIVATE
    Qt6::Test Qt6::Core Qt6::Widgets
    ${VTK_LIBRARIES}
    ${OpenCV_LIBS}
    AppCore
)
add_test(NAME TestUserAuthDialogs COMMAND TestUserAuthDialogs)
# Subject is a source file → keep it out of the unity TU (see TestCrosshairGeometry).
set_target_properties(TestUserAuthDialogs PROPERTIES UNITY_BUILD OFF)

# Test PluginUi — the remaining plugin UI: the four ribbon panels, the two
# mail dialogs, the viewer view model, and the AI plugin's inference dispatch
# (the ONNX call must leave the GUI thread, one at a time).
qt_add_executable(TestPluginUi
    TestPluginUi.cpp
    ../src/modules/SendMailPlugin/SendMailRibbonUI.cpp
    ../src/modules/SendMailPlugin/MailDockWidget.cpp
    ${SENDMAIL_HELPER_SOURCES}
    ../src/modules/SendMailPlugin/MailFilterDialog.cpp
    ../src/modules/SendMailPlugin/MailSettingsDialog.cpp
    ../src/modules/AIAssistantPlugin/AIAssistantRibbonUI.cpp
    ../src/modules/AIAssistantPlugin/ChatMessageRenderer.cpp
    ../src/modules/AIProcessorPlugin/AIProcessorRibbonUI.cpp
    ../src/modules/AIProcessorPlugin/AITrainDockWidget.cpp
    ../src/modules/AIProcessorPlugin/AIProcessorPlugin.cpp
    ${VIDEO_TRACKER_SOURCES}
    ../src/modules/ViewerPlugin/ViewerRibbonUI.cpp
    ../src/modules/ViewerPlugin/ViewerViewModel.cpp
    ../src/modules/ViewerPlugin/ViewerListUI.cpp
)
target_include_directories(TestPluginUi PRIVATE
    ../src
    ../src/include
    ../src/utils
    ../src/services/scene
    ../src/services/viewer
    ../src/services/settings
    ../src/modules/SendMailPlugin
    ../src/modules/AIAssistantPlugin
    ../src/modules/AIProcessorPlugin
    ../src/modules/ViewerPlugin
    ../tests/core
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(TestPluginUi PRIVATE
    Qt6::Test Qt6::Core Qt6::Widgets
    ${VTK_LIBRARIES}
    ${OpenCV_LIBS}
    AppCore
)
# The ribbon panels are built on AbstractRibbonPanel, whose QPointer<QGroupBox>
# member needs the complete type — the light PCH provides it (the plugins use
# the same PCH, so the panel sources compile identically here).
target_precompile_headers(TestPluginUi REUSE_FROM SharedPCHLight)
target_link_libraries(TestPluginUi PRIVATE SharedPCHLight)
add_test(NAME TestPluginUi COMMAND TestPluginUi)
# Subject is a source file → keep it out of the unity TU (see TestCrosshairGeometry).
set_target_properties(TestPluginUi PROPERTIES UNITY_BUILD OFF)

# Test SmtpMailer — the SMTP client: input validation and failure reporting.
# The blocking call runs on a worker thread (the mail stack asserts on the GUI
# thread in debug builds).
qt_add_executable(TestSmtpMailer TestSmtpMailer.cpp)
target_include_directories(TestSmtpMailer PRIVATE
    ../src/services/mail
    ../src/include
)
target_link_libraries(TestSmtpMailer PRIVATE
    Qt6::Test Qt6::Core Qt6::Concurrent
    AppCore
)
add_test(NAME TestSmtpMailer COMMAND TestSmtpMailer)

# Test GuardBranches — the remaining decision branches of the guard layer:
# the plugin host's loader failures, the dependency resolver's wiring
# diagnostics and the per-user settings fallback.
qt_add_executable(TestGuardBranches TestGuardBranches.cpp)
target_link_libraries(TestGuardBranches PRIVATE Qt6::Test Qt6::Core AppCore)
# The concrete kernel is PRIVATE on AppCore (src/include is the public
# surface) — a test of the kernel has to ask for src/core explicitly.
target_include_directories(TestGuardBranches PRIVATE ../src/core)
add_test(NAME TestGuardBranches COMMAND TestGuardBranches)

# Test SecretProtection — the at-rest confidentiality contract of the user
# INI: the vault fails closed, a secret key never reaches the plaintext
# namespace, the legacy plaintext secrets are sealed on load and the licence
# inventory survives a blob the application cannot decrypt.
qt_add_executable(TestSecretProtection TestSecretProtection.cpp)
target_link_libraries(TestSecretProtection PRIVATE Qt6::Test Qt6::Core AppCore)
# The concrete kernel is PRIVATE on AppCore (src/include is the public
# surface) — a test of the kernel has to ask for src/core explicitly.
target_include_directories(TestSecretProtection PRIVATE ../src/core)
add_test(NAME TestSecretProtection COMMAND TestSecretProtection)

# Test PluginWidgets — the last two plugin translation units without a test:
# the sealed-attachment viewer and the video tracking thread.
qt_add_executable(TestPluginWidgets
    TestPluginWidgets.cpp
    ../src/modules/AIAssistantPlugin/ChatImageViewer.cpp
    ${VIDEO_TRACKER_SOURCES}
)
target_include_directories(TestPluginWidgets PRIVATE
    ../src
    ../src/include
    ../src/utils
    ../src/modules/AIAssistantPlugin
    ../src/modules/AIProcessorPlugin
    ../tests/core
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ../src/core
)
target_link_libraries(TestPluginWidgets PRIVATE
    Qt6::Test Qt6::Core Qt6::Widgets
    ${VTK_LIBRARIES}
    ${OpenCV_LIBS}
    AppCore
)
add_test(NAME TestPluginWidgets COMMAND TestPluginWidgets)
# Subject is a source file → keep it out of the unity TU (see TestCrosshairGeometry).
set_target_properties(TestPluginWidgets PROPERTIES UNITY_BUILD OFF)

# Test UtilsLayer — the shared utility layer (src/utils/**) that every plugin
# links against: file/thumbnail handling, markdown rendering, icon factory,
# theming, logging, dialogs and the ribbon scaffolding.  These helpers were
# the least measured part of src/; this target pins their contract.
qt_add_executable(TestUtilsLayer TestUtilsLayer.cpp)
target_link_libraries(TestUtilsLayer PRIVATE Qt6::Test Qt6::Core Qt6::Widgets AppCore)
add_test(NAME TestUtilsLayer COMMAND TestUtilsLayer)

# Test CrosshairManager — the MPR crosshair overlay: three views, the
# reslice pipeline, the hit-tested anchor and the drag state machine.
# CrosshairManager is not exported by AppCore, so the translation unit is
# compiled into the test (and kept out of the unity build so the exporter
# can map it).
qt_add_executable(TestCrosshairManager
    TestCrosshairManager.cpp
    ../src/services/scene/CrosshairManager.cpp
    ../src/services/scene/CrosshairGeometry.cpp
)
target_include_directories(TestCrosshairManager PRIVATE
    ../src/include
    ../src/services/scene
    ${VTK_INCLUDE_DIRS}
)
target_link_libraries(TestCrosshairManager PRIVATE
    Qt6::Test Qt6::Core
    ${VTK_LIBRARIES}
)
add_test(NAME TestCrosshairManager COMMAND TestCrosshairManager)
set_target_properties(TestCrosshairManager PROPERTIES UNITY_BUILD OFF)

# Test ViewerLoaders — the file → VTK actor loaders of the 2-D/3-D viewer.
# Both classes are exported by AppCore; the test only needs their headers.
qt_add_executable(TestViewerLoaders TestViewerLoaders.cpp)
target_include_directories(TestViewerLoaders PRIVATE
    ../src/include
    ../src/services/viewer
    ${OpenCV_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
)
target_link_libraries(TestViewerLoaders PRIVATE
    Qt6::Test Qt6::Core
    ${OpenCV_LIBS}
    ${VTK_LIBRARIES}
    AppCore
)
add_test(NAME TestViewerLoaders COMMAND TestViewerLoaders)

# Test SceneService — 3-D scene manager, VTK actor lifecycle, point clouds, view settings.
qt_add_executable(TestSceneService TestSceneService.cpp)
target_include_directories(TestSceneService PRIVATE
    ../src/include
    ../src/services/scene
    ${OpenCV_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
)
target_link_libraries(TestSceneService PRIVATE
    Qt6::Test Qt6::Core
    ${OpenCV_LIBS}
    ${VTK_LIBRARIES}
    AppCore
)
add_test(NAME TestSceneService COMMAND TestSceneService)

# ── UTF-8 source literals in the tests ───────────────────────────────────────
# The test sources contain Vietnamese literals (mail previews, user-facing
# strings).  Without /utf-8 MSVC decodes them with the system code page, so
# the expected strings end up mangled and the assertions compare garbage.
# The application targets already compile with /utf-8 (root CMakeLists.txt);
# the tests did not.
if(MSVC)
    get_property(_test_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_test_target IN LISTS _test_targets)
        target_compile_options(${_test_target} PRIVATE /utf-8)
    endforeach()
endif()

# ── Test environment for every registered test ───────────────────────────────
# Two properties, applied uniformly instead of per test:
#
#  * QT_ASSUME_STDERR_HAS_CONSOLE — QtTest writes its report to stderr; Qt on
#    Windows silently drops stderr when no console is attached (ctest/CI),
#    which made `ctest --output-on-failure` log an empty output block for
#    every failure.
#  * QT_QPA_PLATFORM=offscreen — there is no desktop in CI, and a test that
#    happens to construct a widget must not depend on the machine it runs on.
#    The old build set this on five hand-picked tests; a new GUI test was free
#    to forget it and pass locally while failing in CI (or vice versa).
get_property(ALL_TESTS DIRECTORY PROPERTY TESTS)
if(ALL_TESTS)
    foreach(test_name IN LISTS ALL_TESTS)
        set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT_MODIFICATION
            "QT_ASSUME_STDERR_HAS_CONSOLE=set:1")
        set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT_MODIFICATION
            "QT_QPA_PLATFORM=set:offscreen")
    endforeach()
    list(LENGTH ALL_TESTS ALL_TESTS_COUNT)
    message(STATUS "QtTest environment (stderr + offscreen) applied to ${ALL_TESTS_COUNT} tests")
endif()
