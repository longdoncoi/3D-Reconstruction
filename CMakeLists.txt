cmake_minimum_required(VERSION 3.19)
project(3D-Reconstruction LANGUAGES CXX)

# Fix future timestamps that cause constant rebuilds in MSBuild
if(WIN32)
    message(STATUS "🔍 Checking for files with future timestamps in source directory...")
    execute_process(
        COMMAND powershell -Command "
            $futureFiles = Get-ChildItem -Recurse -File | Where-Object { $_.LastWriteTime -gt (Get-Date) };
            if ($futureFiles) {
                Write-Host '⚠️ Found' $futureFiles.Count 'files with future timestamps in source directory. Fixing...';
                $futureFiles | ForEach-Object { 
                    Write-Host '  Fixing:' $_.FullName;
                    $_.LastWriteTime = (Get-Date) 
                };
            } else {
                Write-Host '✅ No files with future timestamps found in source directory.';
            }
        "
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

# ─── 1. Qt6 ───────────────────────────────────────────────────────────────────
find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets OpenGL Network Concurrent OpenGLWidgets Svg SvgWidgets)
qt_standard_project_setup()

# ─── 2. VTK ───────────────────────────────────────────────────────────────────
set(VTK_MODULE_ENABLE_VTK_RenderingVulkan NO)
set(VTK_DIR "C:/Program Files/VTK/lib/cmake/vtk-9.6" CACHE PATH "Path to VTK")
find_package(VTK REQUIRED COMPONENTS
    CommonCore
    RenderingOpenGL2
    GUISupportQt
    ViewsQt
    IOGeometry
    IOImport
    FiltersCore
    IOImage
    IOXML
    RenderingCore
    ImagingCore
    RenderingVolume
    RenderingVolumeOpenGL2
)

# ─── 3. OpenCV ────────────────────────────────────────────────────────────────
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)

# ─── 4. PCL ───────────────────────────────────────────────────────────────────
set(PCL_DIR "C:/Program Files/PCL 1.15.1/cmake" CACHE PATH "Path to PCL CMake files")
find_package(PCL REQUIRED COMPONENTS common filters io)

# ─── 5. ONNX Runtime ─────────────────────────────────────────────────────────
include(FetchContent)
set(ONNXRUNTIME_VERSION "1.20.1")
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}.zip")
# set(ONNXRUNTIME_URL "F:/onnxruntime-win-x64-gpu-1.20.1.zip")
FetchContent_Declare(onnxruntime URL ${ONNXRUNTIME_URL})
FetchContent_GetProperties(onnxruntime)
if(NOT onnxruntime_POPULATED)
  FetchContent_Populate(onnxruntime)
endif()

# ─── 5b. TensorRT ────────────────────────────────────────────────────────────
# onnxruntime_providers_tensorrt.dll loads TRT DLLs dynamically at runtime.
# They MUST be present in the executable directory or on PATH.
set(TENSORRT_ROOT "C:/TensorRT-10.16.1.11" CACHE PATH "Path to TensorRT installation")
set(TENSORRT_BIN_DIR "${TENSORRT_ROOT}/bin")

# ─── 6. AUTOUIC search paths ─────────────────────────────────────────────────
set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/app)

# ─── 7. Executable ────────────────────────────────────────────────────────────
qt_add_executable(3D-Reconstruction WIN32 MACOSX_BUNDLE

    # ── app/ (UI Shell) ──────────────────────────────────────────────────────
    src/app/Main.cpp
    src/app/MainWindow.h
    src/app/MainWindow.cpp
    src/app/MainWindow.ui
    src/app/AppShell.h                          # NEW: IAppContext implementation
    src/app/AppShell.cpp
    src/app/AppConstants.h
    src/app/StyleManager.h
    src/app/StyleManager.cpp

    # ── core/ (Interfaces + Kernel) ──────────────────────────────────────────
    src/core/Global.h
    src/core/AppConfig.h
    src/core/AppConfig.cpp
    src/core/SignalBus.h
    src/core/SignalBus.cpp
    src/core/LanguageManager.h
    src/core/LanguageManager.cpp
    src/core/UserManager.h
    src/core/UserManager.cpp
    src/core/SmtpMailer.h
    src/core/SmtpMailer.cpp
    src/core/ServiceRegistry.h                  # NEW: Type-safe service locator
    src/core/IReconstructionService.h           # NEW: Reconstruction interface
    src/core/IAIService.h                       # NEW: AI service interface
    src/core/IMailService.h
    src/core/IAppContext.h                      # Updated: added services()
    src/core/IAIAssistantService.h
    src/core/ISceneService.h
    src/core/ISettingsService.h
    src/core/IViewerService.h
    src/core/IPlugin.h
    src/core/AIMode.h

    # ── services/scene/ ──────────────────────────────────────────────────────
    src/services/scene/SceneService.h
    src/services/scene/SceneService.cpp
    src/services/scene/CrosshairManager.h
    src/services/scene/CrosshairManager.cpp

    # ── services/viewer/ ─────────────────────────────────────────────────────
    src/services/viewer/ViewerService.h
    src/services/viewer/ViewerService.cpp
    src/services/viewer/Image2DLoader.h
    src/services/viewer/Image2DLoader.cpp
    src/services/viewer/Model3DLoader.h
    src/services/viewer/Model3DLoader.cpp
    src/services/viewer/DicomLoader.h
    src/services/viewer/DicomLoader.cpp
    src/services/viewer/ChatImageViewer.h
    src/services/viewer/ChatImageViewer.cpp

    # ── services/settings/ ───────────────────────────────────────────────────
    src/services/settings/SettingsService.h
    src/services/settings/SettingsService.cpp

    # ── services/reconstruction/ ─────────────────────────────────────────────
    src/services/reconstruction/ReconstructionPipeline.h
    src/services/reconstruction/ReconstructionPipeline.cpp
    src/services/reconstruction/ReconstructionConfig.h
    src/services/reconstruction/ReconstructThread.h
    src/services/reconstruction/ReconstructThread.cpp
    src/services/reconstruction/ReconstructionService.h
    src/services/reconstruction/ReconstructionService.cpp
    src/services/reconstruction/FeatureExtractor.h        # Phase 4
    src/services/reconstruction/FeatureExtractor.cpp
    src/services/reconstruction/FeatureMatcher.h
    src/services/reconstruction/FeatureMatcher.cpp
    src/services/reconstruction/PoseEstimator.h
    src/services/reconstruction/PoseEstimator.cpp
    src/services/reconstruction/Triangulator.h
    src/services/reconstruction/Triangulator.cpp
    src/services/reconstruction/PointCloudFilter.h
    src/services/reconstruction/PointCloudFilter.cpp

    # ── services/ai/ ─────────────────────────────────────────────────────────
    src/services/ai/AIProcessor.h
    src/services/ai/AIProcessor.cpp
    src/services/ai/AIAssistant.h
    src/services/ai/AIAssistant.cpp
    src/services/ai/AIService.h                 # NEW
    src/services/ai/AIService.cpp               # NEW
    src/services/mail/MailService.h
    src/services/mail/MailService.cpp

    # ── utils/ ───────────────────────────────────────────────────────────────
    src/utils/PanStyle.h
    src/utils/PanStyle.cpp
    src/utils/Logger.h
    src/utils/Logger.cpp
    src/utils/CustomProgressDialog.h
    src/utils/CustomProgressDialog.cpp
    src/utils/ModernDialog.h
    src/utils/ModernDialog.cpp
    src/utils/ModernMessageBox.h
    src/utils/HtmlUtilities.h
    src/utils/HtmlUtilities.cpp
    src/utils/FileUtilities.h
    src/utils/FileUtilities.cpp
    src/utils/IconFactory.h            # Phase 5
    src/utils/IconFactory.cpp
)

# ─── 8. Include directories ───────────────────────────────────────────────────
target_include_directories(3D-Reconstruction PRIVATE
    # App shell
    ${CMAKE_CURRENT_SOURCE_DIR}/src/app

    # Core layer (interfaces + kernel)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core

    # Services layer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/scene
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/viewer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/settings
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/reconstruction
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/ai
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/mail

    # Utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils

    # Third-party
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${onnxruntime_SOURCE_DIR}/include
)

# ─── 9. Compile definitions & options ─────────────────────────────────────────
target_compile_definitions(3D-Reconstruction PRIVATE
    ${VTK_DEFINITIONS}
    ${PCL_DEFINITIONS}
    PCL_SILENCE_MALLOC_WARNING=1
    APP_LIBRARY
)
if(MSVC)
    target_compile_options(3D-Reconstruction PRIVATE /utf-8 /FS)
    add_compile_options(/FS)
endif()

# ─── 10. Link libraries ───────────────────────────────────────────────────────
target_link_libraries(3D-Reconstruction
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        Qt6::OpenGL
        Qt6::Network
        Qt6::Concurrent
        Qt6::OpenGLWidgets
        ${VTK_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib
)

set_target_properties(3D-Reconstruction PROPERTIES ENABLE_EXPORTS ON)

# ─── 11. Post-build: Copy DLLs ───────────────────────────────────────────────
# Copy ONNX Runtime DLLs
add_custom_command(TARGET 3D-Reconstruction POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll"
        "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_shared.dll"
        "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_cuda.dll"
        "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_tensorrt.dll"
        "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_shared.lib"
        $<TARGET_FILE_DIR:3D-Reconstruction>
)

# Copy TensorRT DLLs — required by onnxruntime_providers_tensorrt.dll at runtime.
# Without these, the TRT provider crashes with access violation (0xc0000005).
# Only the core runtime DLLs are needed; builder_resource DLLs are loaded lazily.
add_custom_command(TARGET 3D-Reconstruction POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TENSORRT_BIN_DIR}/nvinfer_10.dll"
        "${TENSORRT_BIN_DIR}/nvinfer_dispatch_10.dll"
        "${TENSORRT_BIN_DIR}/nvinfer_lean_10.dll"
        "${TENSORRT_BIN_DIR}/nvinfer_plugin_10.dll"
        "${TENSORRT_BIN_DIR}/nvinfer_vc_plugin_10.dll"
        "${TENSORRT_BIN_DIR}/nvonnxparser_10.dll"
        $<TARGET_FILE_DIR:3D-Reconstruction>
    COMMENT "Copying TensorRT runtime DLLs to output directory"
)

# Copy TensorRT builder-resource DLLs (needed when TRT builds/caches engines)
add_custom_command(TARGET 3D-Reconstruction POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TENSORRT_BIN_DIR}/nvinfer_builder_resource_ptx_10.dll"
        "${TENSORRT_BIN_DIR}/nvinfer_builder_resource_sm75_10.dll"
        "${TENSORRT_BIN_DIR}/nvinfer_builder_resource_sm80_10.dll"
        "${TENSORRT_BIN_DIR}/nvinfer_builder_resource_sm86_10.dll"
        "${TENSORRT_BIN_DIR}/nvinfer_builder_resource_sm89_10.dll"
        "${TENSORRT_BIN_DIR}/nvinfer_builder_resource_sm90_10.dll"
        "${TENSORRT_BIN_DIR}/nvinfer_builder_resource_sm100_10.dll"
        "${TENSORRT_BIN_DIR}/nvinfer_builder_resource_sm120_10.dll"
        $<TARGET_FILE_DIR:3D-Reconstruction>
    COMMENT "Copying TensorRT builder-resource DLLs to output directory"
)

# Copy missing PCL, VTK 9.4, and OpenNI2 DLLs required at runtime
add_custom_command(TARGET 3D-Reconstruction POST_BUILD
    COMMAND powershell -Command "$destDir = '$<TARGET_FILE_DIR:3D-Reconstruction>'; Get-ChildItem -Path 'C:\\Program Files\\PCL 1.15.1\\3rdParty\\VTK\\bin' -Filter vtk*.dll | ForEach-Object { $target = Join-Path $destDir $_.Name; if (-not (Test-Path $target) -or (Get-Item $target).Length -ne $_.Length) { Copy-Item $_.FullName -Destination $destDir -Force } }"
    COMMAND powershell -Command "$destDir = '$<TARGET_FILE_DIR:3D-Reconstruction>'; Get-ChildItem -Path 'C:\\Program Files\\PCL 1.15.1\\bin' | Where-Object { $_.Name -like 'pcl*.dll' -or $_.Name -like 'boost*.dll' -or $_.Name -like 'flann*.dll' -or $_.Name -like 'qhull*.dll' } | ForEach-Object { $target = Join-Path $destDir $_.Name; if (-not (Test-Path $target) -or (Get-Item $target).Length -ne $_.Length) { Copy-Item $_.FullName -Destination $destDir -Force } }"
    COMMAND powershell -Command "$destDir = '$<TARGET_FILE_DIR:3D-Reconstruction>'; Get-ChildItem -Path 'C:\\Program Files\\OpenNI2\\Tools' -Filter *.dll | ForEach-Object { $target = Join-Path $destDir $_.Name; if (-not (Test-Path $target) -or (Get-Item $target).Length -ne $_.Length) { Copy-Item $_.FullName -Destination $destDir -Force } }"
    COMMENT "Copying PCL, VTK, and OpenNI dependencies safely (avoiding Qt/vcpkg DLL Hell)"
)

# ─── Deploy Qt Plugins Manually (Bypassing windeployqt) ────────────────────
# vcpkg often does not build windeployqt.exe. Falling back to the system's windeployqt
# causes ABI mismatches (mixing Qt 6.5 and Qt 6.9 DLLs).
# vcpkg AppLocal already deploys Qt DLLs like Qt6Core.dll. We just manually copy the plugins!
file(GLOB_RECURSE QWINDOWS_DLL_PATH "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/x64-windows/*/qwindows.dll")
if(QWINDOWS_DLL_PATH)
    list(GET QWINDOWS_DLL_PATH 0 QWINDOWS_DLL_FIRST)
    get_filename_component(PLATFORMS_DIR "${QWINDOWS_DLL_FIRST}" DIRECTORY)
    get_filename_component(PLUGINS_ROOT "${PLATFORMS_DIR}" DIRECTORY)
    
    add_custom_command(TARGET 3D-Reconstruction POST_BUILD
        COMMAND powershell -Command "if (-not (Test-Path '$<TARGET_FILE_DIR:3D-Reconstruction>\\platforms')) { Copy-Item -Path '${PLATFORMS_DIR}' -Destination '$<TARGET_FILE_DIR:3D-Reconstruction>' -Recurse -Force }"
        COMMAND powershell -Command "if (-not (Test-Path '$<TARGET_FILE_DIR:3D-Reconstruction>\\imageformats')) { Copy-Item -Path '${PLUGINS_ROOT}/imageformats' -Destination '$<TARGET_FILE_DIR:3D-Reconstruction>' -Recurse -Force }"
        COMMAND powershell -Command "if (-not (Test-Path '$<TARGET_FILE_DIR:3D-Reconstruction>\\iconengines')) { Copy-Item -Path '${PLUGINS_ROOT}/iconengines' -Destination '$<TARGET_FILE_DIR:3D-Reconstruction>' -Recurse -Force }"
        COMMAND powershell -Command "if (-not (Test-Path '$<TARGET_FILE_DIR:3D-Reconstruction>\\styles')) { Copy-Item -Path '${PLUGINS_ROOT}/styles' -Destination '$<TARGET_FILE_DIR:3D-Reconstruction>' -Recurse -Force }"
        COMMAND powershell -Command "if (-not (Test-Path '$<TARGET_FILE_DIR:3D-Reconstruction>\\tls')) { Copy-Item -Path '${PLUGINS_ROOT}/tls' -Destination '$<TARGET_FILE_DIR:3D-Reconstruction>' -Recurse -Force }"
        COMMENT "Manually copying Qt plugins from vcpkg (avoiding windeployqt ABI mismatch)"
    )
else()
    message(WARNING "qwindows.dll NOT FOUND in vcpkg_installed! GUI might not start.")
endif()


# Copy translations folder
add_custom_command(TARGET 3D-Reconstruction POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/translations"
        "$<TARGET_FILE_DIR:3D-Reconstruction>/translations"
    COMMENT "Copying translations to output directory"
)

# ─── 12. Prevent DLL Hell by wiping stale DLLs at Configure time ────────────
# Because the user previously copied conflicting DLLs (freetype.dll, etc.) into Release,
# we must forcefully delete them so windeployqt and vcpkg can deploy the correct ones.
file(GLOB STALE_DLLS "${CMAKE_CURRENT_BINARY_DIR}/Release/*.dll")
if(STALE_DLLS)
    file(REMOVE ${STALE_DLLS})
    message(STATUS "🧹 Wiped stale DLLs from Release directory to prevent DLL Hell conflicts.")
endif()

# ─── 13. Plugins ─────────────────────────────────────────────────────────────
add_subdirectory(src/modules)

# ─── 13. Testing ─────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)
