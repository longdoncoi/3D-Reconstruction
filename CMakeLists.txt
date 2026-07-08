cmake_minimum_required(VERSION 3.19)
project(3D-Reconstruction LANGUAGES CXX)

# ─── Build Performance Options ────────────────────────────────────────────────
option(CMAKE_UNITY_BUILD "Enable Unity (Jumbo) build for faster compilation" ON)
option(CLEAN_STALE_DLLS "Remove stale DLLs from Release directory at configure time" OFF)
if(CMAKE_UNITY_BUILD)
    set(CMAKE_UNITY_BUILD_BATCH_SIZE 8)
endif()

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
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets OpenGL Network Concurrent OpenGLWidgets Svg SvgWidgets)
qt_standard_project_setup()

# ─── 2. VTK ───────────────────────────────────────────────────────────────────
set(VTK_MODULE_ENABLE_VTK_RenderingVulkan NO)
# Only set a hardcoded VTK_DIR for local builds; vcpkg handles discovery in CI.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(VTK_DIR "C:/Program Files/VTK/lib/cmake/vtk-9.6" CACHE PATH "Path to VTK")
endif()
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
    RenderingAnnotation
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
qt_add_executable(3D-Reconstruction WIN32 MACOSX_BUNDLE)

# ─── Source subdirectories ───────────────────────────────────────────────────
add_subdirectory(src/app)
add_subdirectory(src/core)
add_subdirectory(src/services)
add_subdirectory(src/utils)

# ─── 7b. Precompiled Headers (PCH) — biggest build speed win ─────────────────
target_precompile_headers(3D-Reconstruction PRIVATE src/pch.h)

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
    target_compile_options(3D-Reconstruction PRIVATE /utf-8 /FS /MP)
    add_compile_options(/FS /MP)
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
# These DLL copy commands reference local paths that don't exist on CI runners.
# Skip them when building via vcpkg toolchain (CI); vcpkg AppLocal handles DLL deployment.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
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
    # Consolidated into a single PowerShell invocation to avoid repeated process startup overhead
    add_custom_command(TARGET 3D-Reconstruction POST_BUILD
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass -Command "$destDir = '$<TARGET_FILE_DIR:3D-Reconstruction>'; $sources = @(@{ Path='C:/Program Files/PCL 1.15.1/3rdParty/VTK/bin'; Filter='vtk*.dll'; Where=$null }, @{ Path='C:/Program Files/PCL 1.15.1/bin'; Filter=$null; Where={ $_.Name -like 'pcl*.dll' -or $_.Name -like 'boost*.dll' -or $_.Name -like 'flann*.dll' -or $_.Name -like 'qhull*.dll' } }, @{ Path='C:/Program Files/OpenNI2/Tools'; Filter='*.dll'; Where=$null }); foreach ($s in $sources) { if (Test-Path $s.Path) { $items = if ($s.Filter) { Get-ChildItem -Path $s.Path -Filter $s.Filter } else { Get-ChildItem -Path $s.Path }; if ($s.Where) { $items = $items | Where-Object $s.Where }; $items | ForEach-Object { $target = Join-Path $destDir $_.Name; if (-not (Test-Path $target) -or (Get-Item $target).Length -ne $_.Length) { Copy-Item $_.FullName -Destination $destDir -Force } } } }"
        COMMENT "Copying PCL, VTK, and OpenNI dependencies (single PS invocation)"
    )
else()
    message(STATUS "⏭️ Skipping local DLL copy commands (CI/vcpkg toolchain build detected)")
endif()

# ─── Deploy Qt Plugins Manually (Bypassing windeployqt) ────────────────────
# vcpkg often does not build windeployqt.exe. Falling back to the system's windeployqt
# causes ABI mismatches (mixing Qt 6.5 and Qt 6.9 DLLs).
# vcpkg AppLocal already deploys Qt DLLs like Qt6Core.dll. We just manually copy the plugins!
file(GLOB_RECURSE QWINDOWS_DLL_PATH "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/x64-windows/*/qwindows.dll")
if(QWINDOWS_DLL_PATH)
    list(GET QWINDOWS_DLL_PATH 0 QWINDOWS_DLL_FIRST)
    get_filename_component(PLATFORMS_DIR "${QWINDOWS_DLL_FIRST}" DIRECTORY)
    get_filename_component(PLUGINS_ROOT "${PLATFORMS_DIR}" DIRECTORY)
    
    # Consolidated into single PowerShell invocation (was 5 separate invocations)
    add_custom_command(TARGET 3D-Reconstruction POST_BUILD
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass -Command "$dest = '$<TARGET_FILE_DIR:3D-Reconstruction>'; @('${PLATFORMS_DIR}','${PLUGINS_ROOT}/imageformats','${PLUGINS_ROOT}/iconengines','${PLUGINS_ROOT}/styles','${PLUGINS_ROOT}/tls') | ForEach-Object { $dirName = Split-Path $_ -Leaf; $targetDir = Join-Path $dest $dirName; if (-not (Test-Path $targetDir) -and (Test-Path $_)) { Copy-Item -Path $_ -Destination $dest -Recurse -Force } }"
        COMMENT "Copying Qt plugins from vcpkg (single PS invocation)"
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
# Only runs when -DCLEAN_STALE_DLLS=ON is passed (e.g., after DLL conflicts).
# Disabled by default to avoid unnecessary re-copies on every configure.
if(CLEAN_STALE_DLLS)
    file(GLOB STALE_DLLS "${CMAKE_CURRENT_BINARY_DIR}/Release/*.dll")
    if(STALE_DLLS)
        file(REMOVE ${STALE_DLLS})
        message(STATUS "🧹 Wiped stale DLLs from Release directory to prevent DLL Hell conflicts.")
    endif()
endif()

# ─── 13. Plugins ─────────────────────────────────────────────────────────────
add_subdirectory(src/modules)

# ─── 13. Testing ─────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)
