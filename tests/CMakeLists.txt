find_package(Qt6 REQUIRED COMPONENTS Test Core)
set(CMAKE_AUTOMOC ON)

# PCL is looked up first: its config resolves the VTK installation the
# project was built against and therefore fills VTK_INCLUDE_DIRS for the
# targets below (VTK_INCLUDE_DIRS stays empty otherwise).
find_package(PCL REQUIRED COMPONENTS common filters io)
find_package(OpenCV REQUIRED COMPONENTS core)

# Architecture guard — plugin boundaries + UserManager singleton usage.
# Encodes the invariants documented in AGENTS.md; runs in milliseconds.
add_test(NAME TestArchitectureBoundaries
         COMMAND ${CMAKE_COMMAND}
                 "-DSOURCE_DIR=${CMAKE_SOURCE_DIR}"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckArchitectureBoundaries.cmake")

# Architecture boundary test (post-ADR-001..004)
qt_add_executable(TestArchitectureBoundary TestArchitectureBoundary.cpp)
target_include_directories(TestArchitectureBoundary PRIVATE ../src ../tests/core/mock)
target_link_libraries(TestArchitectureBoundary PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestArchitectureBoundary COMMAND TestArchitectureBoundary)

# Test ServiceRegistry
qt_add_executable(TestServiceRegistry TestServiceRegistry.cpp)
target_include_directories(TestServiceRegistry PRIVATE ../src/core)
target_link_libraries(TestServiceRegistry PRIVATE Qt6::Test Qt6::Core)
add_test(NAME TestServiceRegistry COMMAND TestServiceRegistry)

# Test PluginDependencyResolver — thứ tự init plugin (deps graph + phase + cycle)
qt_add_executable(TestPluginDependencyResolver TestPluginDependencyResolver.cpp)
target_include_directories(TestPluginDependencyResolver PRIVATE ../src/core)
target_link_libraries(TestPluginDependencyResolver PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestPluginDependencyResolver COMMAND TestPluginDependencyResolver)

# Test PluginHost — vòng đời load/resolve/init/cleanup (loader giả, không cần DLL thật)
qt_add_executable(TestPluginHost TestPluginHost.cpp)
target_link_libraries(TestPluginHost PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestPluginHost COMMAND TestPluginHost)

# Test AIModelLoadService — AppCore-owned worker lifetime/cancellation contract.
qt_add_executable(TestAIModelLoadService TestAIModelLoadService.cpp)
target_include_directories(TestAIModelLoadService PRIVATE
    ../src/core
    ../src/services/ai
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(TestAIModelLoadService PRIVATE
    Qt6::Test Qt6::Core ${OpenCV_LIBS} AppCore)
add_test(NAME TestAIModelLoadService COMMAND TestAIModelLoadService)

# Smoke test on real plugin DLLs — must run after the plugin targets exist.
qt_add_executable(TestPluginSmoke TestPluginSmoke.cpp)
# The stub context mirrors ISceneService, whose interface speaks VTK types.
# VTK_INCLUDE_DIRS is only filled in by find_package(PCL) further down, so the
# imported targets are used here instead of the raw variable.
target_include_directories(TestPluginSmoke PRIVATE ../src ../tests/core ${OpenCV_INCLUDE_DIRS})
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
add_test(NAME TestAppConfig COMMAND TestAppConfig)

# Test LanguageManager
qt_add_executable(TestLanguageManager TestLanguageManager.cpp)
target_link_libraries(TestLanguageManager PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestLanguageManager COMMAND TestLanguageManager)

# Test Mail MIME parser
qt_add_executable(TestMailMimeParser
    TestMailMimeParser.cpp
    ../src/services/mail/MailMimeParser.cpp
)
target_include_directories(TestMailMimeParser PRIVATE ../src/core ../src/services/mail)
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
target_include_directories(TestCameraParamsParser PRIVATE ../src/services/reconstruction)
target_link_libraries(TestCameraParamsParser PRIVATE Qt6::Test Qt6::Core ${OpenCV_LIBS})
add_test(NAME TestCameraParamsParser COMMAND TestCameraParamsParser)
# Subject is a source file → keep it out of the unity TU (see TestCrosshairGeometry).
set_target_properties(TestCameraParamsParser PROPERTIES UNITY_BUILD OFF)

# Test PointCloudFilter
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d)

qt_add_executable(TestPointCloudFilter TestPointCloudFilter.cpp ../src/services/reconstruction/PointCloudFilter.cpp)
target_include_directories(TestPointCloudFilter PRIVATE 
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
    ../src/core/IReconstructionService.h
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
    ../src/core
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

# Test ReconstructionFinalize — hậu kỳ nặng (MLS densify 1M + Poisson depth 10
# + texture mesh). ~100s trong Release nên gắn label "slow": dev chạy
# `ctest -LE slow` khi cần vòng lặp nhanh, CI vẫn chạy mặc định.
qt_add_executable(TestReconstructionFinalize TestReconstructionFinalize.cpp
    ${RECON_PIPELINE_SOURCES}
)
target_include_directories(TestReconstructionFinalize PRIVATE
    ../src/services/reconstruction
    ../src/core
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
target_include_directories(TestShellService PRIVATE ../src/services/shell)
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
    ../src/app/AppShell.cpp
    ../src/app/AppShell.h
)
target_include_directories(TestAppShell PRIVATE
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

# Test Agent Actions
qt_add_executable(TestAgentActions TestAgentActions.cpp)
target_include_directories(TestAgentActions PRIVATE ../src/services/ai)
target_link_libraries(TestAgentActions PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestAgentActions COMMAND TestAgentActions)

# Test ChatSessionManager — session list/current pointer/attachment cleanup/persistence
qt_add_executable(TestChatSessionManager TestChatSessionManager.cpp)
target_include_directories(TestChatSessionManager PRIVATE ../src/services/ai)
target_link_libraries(TestChatSessionManager PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestChatSessionManager COMMAND TestChatSessionManager)

# Test AtRestCipher — the AES-256-GCM envelope shared with the Python service:
# byte layout, associated data, key resolution and the fail-closed policy.
qt_add_executable(TestAtRestCipher TestAtRestCipher.cpp)
target_include_directories(TestAtRestCipher PRIVATE ../src/core)
target_link_libraries(TestAtRestCipher PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestAtRestCipher COMMAND TestAtRestCipher)

# Test AIAssistant — request state machine + injected transport (no sockets,
# no Python child process; the ready transition is driven via the protected seam).
qt_add_executable(TestAIAssistant TestAIAssistant.cpp)
target_include_directories(TestAIAssistant PRIVATE ../src/services/ai)
target_link_libraries(TestAIAssistant PRIVATE Qt6::Test Qt6::Core Qt6::Network AppCore)
add_test(NAME TestAIAssistant COMMAND TestAIAssistant)

# Test AIRequestQueue — pending/queued bookkeeping without HTTP or a process.
qt_add_executable(TestAIRequestQueue TestAIRequestQueue.cpp)
target_include_directories(TestAIRequestQueue PRIVATE ../src/services/ai)
target_link_libraries(TestAIRequestQueue PRIVATE Qt6::Test Qt6::Core Qt6::Network AppCore)
add_test(NAME TestAIRequestQueue COMMAND TestAIRequestQueue)

# Test AI Yolo PostProcessor
qt_add_executable(TestAIYoloPostProcessor
    TestAIYoloPostProcessor.cpp
    ../src/services/ai/AIYoloPostProcessor.cpp
)
target_include_directories(TestAIYoloPostProcessor PRIVATE
    ../src/services/ai
)
target_link_libraries(TestAIYoloPostProcessor PRIVATE Qt6::Test Qt6::Core ${OpenCV_LIBS})
add_test(NAME TestAIYoloPostProcessor COMMAND TestAIYoloPostProcessor)
# Subject is a source file → keep it out of the unity TU (see TestCrosshairGeometry).
set_target_properties(TestAIYoloPostProcessor PROPERTIES UNITY_BUILD OFF)

# Test UserManager crypto (PBKDF2 password hashing + SecretVault)
qt_add_executable(TestUserManagerCrypto TestUserManagerCrypto.cpp)
target_link_libraries(TestUserManagerCrypto PRIVATE Qt6::Test Qt6::Core AppCore)
if(WIN32)
    target_link_libraries(TestUserManagerCrypto PRIVATE crypt32)
endif()
add_test(NAME TestUserManagerCrypto COMMAND TestUserManagerCrypto)

# Test MailService credential lifecycle (cache invalidation on logout — no network)
qt_add_executable(TestMailCredentialLifecycle TestMailCredentialLifecycle.cpp)
target_include_directories(TestMailCredentialLifecycle PRIVATE
    ../src/core
    ../src/services/mail)
target_link_libraries(TestMailCredentialLifecycle PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestMailCredentialLifecycle COMMAND TestMailCredentialLifecycle)

# Test SmtpOutbox (IOutgoingMail adapter — validation branches, no network)
qt_add_executable(TestSmtpOutbox TestSmtpOutbox.cpp)
target_include_directories(TestSmtpOutbox PRIVATE ../src/services/mail)
target_link_libraries(TestSmtpOutbox PRIVATE Qt6::Test Qt6::Core AppCore)
if(WIN32)
    target_link_libraries(TestSmtpOutbox PRIVATE crypt32)
endif()
add_test(NAME TestSmtpOutbox COMMAND TestSmtpOutbox)

# Test SettingsService (injected config path — never touches real Config.ini)
qt_add_executable(TestSettingsService TestSettingsService.cpp)
target_include_directories(TestSettingsService PRIVATE ../src/services/settings)
target_link_libraries(TestSettingsService PRIVATE Qt6::Test Qt6::Core AppCore)
add_test(NAME TestSettingsService COMMAND TestSettingsService)

# Test CrosshairGeometry (pure DICOM-slice math used by SceneService;
# translation unit has no VTK dependency after ORI_* moved to the pure header)
qt_add_executable(TestCrosshairGeometry
    TestCrosshairGeometry.cpp
    ../src/services/scene/CrosshairGeometry.cpp
)
target_include_directories(TestCrosshairGeometry PRIVATE
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
    ../src/modules/SendMailPlugin/MailMessageFormatter.cpp
    ../src/modules/SendMailPlugin/MailInboxItemFactory.cpp
    ../src/modules/AIAssistantPlugin/AIAttachmentPreviewFactory.cpp
    ../src/modules/AIProcessorPlugin/AIPredictionLogger.cpp
)
target_include_directories(TestPluginHelpers PRIVATE
    ../src/core
    ../src/utils
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
# QTEST_MAIN → QApplication; CI không có desktop → offscreen platform plugin.
set_tests_properties(TestPluginHelpers PROPERTIES
    ENVIRONMENT_MODIFICATION "QT_QPA_PLATFORM=set:offscreen")
# Subject is a source file → keep it out of the unity TU (see TestCrosshairGeometry).
set_target_properties(TestPluginHelpers PROPERTIES UNITY_BUILD OFF)

# Test UtilsLayer — the shared utility layer (src/utils/**) that every plugin
# links against: file/thumbnail handling, markdown rendering, icon factory,
# theming, logging, dialogs and the ribbon scaffolding.  These helpers were
# the least measured part of src/; this target pins their contract.
qt_add_executable(TestUtilsLayer TestUtilsLayer.cpp)
target_link_libraries(TestUtilsLayer PRIVATE Qt6::Test Qt6::Core Qt6::Widgets AppCore)
add_test(NAME TestUtilsLayer COMMAND TestUtilsLayer)
# QTEST_MAIN → QApplication; CI không có desktop → offscreen platform plugin.
set_tests_properties(TestUtilsLayer PROPERTIES
    ENVIRONMENT_MODIFICATION "QT_QPA_PLATFORM=set:offscreen")

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

# ── Test reporting on Windows ────────────────────────────────────────────────
# QtTest writes its report to stderr; Qt on Windows silently drops stderr when
# no console is attached (ctest/CI), which made `ctest --output-on-failure`
# log an empty output block for every failure. Force stderr for all tests that
# were registered in this directory.
get_property(ALL_TESTS DIRECTORY PROPERTY TESTS)
if(ALL_TESTS)
    foreach(test_name IN LISTS ALL_TESTS)
        set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT_MODIFICATION
            "QT_ASSUME_STDERR_HAS_CONSOLE=set:1")
    endforeach()
    list(LENGTH ALL_TESTS ALL_TESTS_COUNT)
    message(STATUS "QtTest stderr reporting enabled for ${ALL_TESTS_COUNT} tests")
endif()
