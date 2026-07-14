cmake_minimum_required(VERSION 3.19)

# ─── vcpkg toolchain selection ─────────────────────────────────────────────
# Two distinct scenarios share this file:
#
#  - Local dev machine: no toolchain is supplied on the command line (Qt
#    Creator only sets a stale, project-local one). Fall back to a single
#    shared, prebuilt vcpkg tree on F: so `vcpkg install` never re-runs on
#    every configure.
#
#  - CI (GitHub Actions, Build-App-ci.yml): the workflow already runs
#    `lukka/run-vcpkg` and passes -DCMAKE_TOOLCHAIN_FILE, -DVCPKG_INSTALLED_DIR
#    (pointed at its GitHub Actions cache, D:/v) explicitly on the `cmake -B
#    build` command line. It also relies on manifest-mode auto-install during
#    configure, since it never calls `vcpkg install` itself. None of the
#    "shared tree" settings below may apply in this case, or CI loses its
#    cache directory and silently stops installing vtk/pcl/opencv4/openssl.
#
# A caller-supplied CMAKE_TOOLCHAIN_FILE (i.e. anything other than the stale
# Qt Creator one) is the signal that we're in the CI/external case.
if(DEFINED CMAKE_TOOLCHAIN_FILE AND NOT CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg-dependencies")
    set(_external_vcpkg_toolchain TRUE)
    message(STATUS "Using externally supplied vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
else()
    set(_external_vcpkg_toolchain FALSE)
endif()

if(NOT _external_vcpkg_toolchain)
    # Qt Creator's package-manager helper is disabled in
    # QtCreatorPackageManager.cmake so it cannot replace this toolchain with a
    # project-local copy and trigger a second manifest installation.
    set(PROJECT_VCPKG_ROOT "F:/PROJECTS/QT/vcpkg" CACHE PATH
        "Shared vcpkg root used by this project")
    set(_project_vcpkg_toolchain
        "${PROJECT_VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

    if(NOT EXISTS "${_project_vcpkg_toolchain}")
        message(FATAL_ERROR
            "Shared vcpkg toolchain was not found: ${_project_vcpkg_toolchain}")
    endif()

    set(CMAKE_TOOLCHAIN_FILE "${_project_vcpkg_toolchain}" CACHE FILEPATH
        "vcpkg toolchain file" FORCE)

    set(VCPKG_INSTALLED_DIR "${PROJECT_VCPKG_ROOT}/installed" CACHE PATH
        "Directory containing the prebuilt vcpkg packages" FORCE)
    # The shared installed tree is prepared separately; do not invoke `vcpkg
    # install` during every CMake configure.
    set(VCPKG_MANIFEST_INSTALL OFF CACHE BOOL
        "Install vcpkg manifest dependencies during CMake configure" FORCE)
endif()

# Needed by both scenarios (plugin deployment further below also relies on
# it). Only set a default if the toolchain/CI command line hasn't already
# defined one — no FORCE, so CI's own triplet detection is never overridden.
if(NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg target triplet")
endif()

# vtk-config.cmake uses this package for its code generators.  A previous
# interrupted local configure may leave the cache value as NOTFOUND, so
# provide the known location in the shared installed tree explicitly. Only
# relevant for the local shared-tree scenario; CI's manifest install resolves
# this on its own.
if(NOT _external_vcpkg_toolchain)
    set(VTKCompileTools_DIR
        "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/vtk-compile-tools"
        CACHE PATH "VTK compile-tools package directory" FORCE)
endif()

project(3D-Reconstruction LANGUAGES CXX)

# ─── Build Performance Options ────────────────────────────────────────────────
option(CMAKE_UNITY_BUILD "Enable Unity (Jumbo) build for faster compilation" OFF)
option(CLEAN_STALE_DLLS "Remove stale DLLs from Release directory at configure time" OFF)
if(CMAKE_UNITY_BUILD)
    # Large unity batches exhaust cl.exe's heap with the Qt/PCL/VTK/OpenCV
    # headers used here.  A batch of one is still opt-in but remains safe.
    set(CMAKE_UNITY_BUILD_BATCH_SIZE 1)
endif()

# Fix future timestamps that cause constant rebuilds in MSBuild.  Restrict the
# scan to project sources: walking build/, .git/, or vcpkg/ is slow and those
# generated trees can legitimately change while CMake is configuring.
if(WIN32)
    message(STATUS "🔍 Checking for files with future timestamps in project sources...")
    execute_process(
        COMMAND powershell -Command "
            $sourceDirectories = @('src', 'tests', 'translations') | Where-Object { Test-Path $_ };
            $futureFiles = @(
                $sourceDirectories | ForEach-Object {
                    Get-ChildItem -Path $_ -Recurse -File -ErrorAction SilentlyContinue
                }
            ) | Where-Object { $_.LastWriteTime -gt (Get-Date) };
            if ($futureFiles) {
                Write-Host '⚠️ Found' $futureFiles.Count 'files with future timestamps in project sources. Fixing...';
                $futureFiles | ForEach-Object { 
                    Write-Host '  Fixing:' $_.FullName;
                    $_.LastWriteTime = (Get-Date) 
                };
            } else {
                Write-Host '✅ No files with future timestamps found in project sources.';
            }
        "
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

# ─── 1. Qt6 ───────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# vcpkg also provides a Qt package, but this application must use the complete
# Qt kit selected in Qt Creator (including Svg and SvgWidgets). Derive its
# CMake package directory from qmake so changing kits remains supported.
if(QT_QMAKE_EXECUTABLE)
    get_filename_component(_qt_bin_dir "${QT_QMAKE_EXECUTABLE}" DIRECTORY)
    get_filename_component(_qt_kit_root "${_qt_bin_dir}" DIRECTORY)
    set(_qt6_kit_config_dir "${_qt_kit_root}/lib/cmake/Qt6")
    if(EXISTS "${_qt6_kit_config_dir}/Qt6Config.cmake")
        # Qt6Config.cmake finds each component as a separate package (for
        # example Qt6CoreTools). Point every available Qt6 package to the
        # selected kit so vcpkg's partial Qt installation cannot be mixed in.
        file(GLOB _qt6_package_dirs LIST_DIRECTORIES true
             "${_qt_kit_root}/lib/cmake/Qt6*")
        foreach(_qt6_package_dir IN LISTS _qt6_package_dirs)
            get_filename_component(_qt6_package_name
                "${_qt6_package_dir}" NAME)
            if(EXISTS "${_qt6_package_dir}/${_qt6_package_name}Config.cmake")
                set(${_qt6_package_name}_DIR "${_qt6_package_dir}" CACHE PATH
                    "${_qt6_package_name} directory selected by the active Qt kit" FORCE)
            endif()
        endforeach()
    endif()
endif()
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
    # /MP creates additional cl.exe processes inside each MSBuild job.  With
    # Qt, PCL, VTK and OpenCV headers this exhausts the compiler heap and can
    # also race PDB creation.  Let the build tool control parallelism instead.
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
# ONNX Runtime is fetched and linked directly by this project rather than by
# vcpkg. It must therefore be copied even when the vcpkg toolchain is active;
# otherwise Windows can load an unrelated onnxruntime.dll found on PATH.
add_custom_command(TARGET 3D-Reconstruction POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll"
        "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_shared.dll"
        "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_cuda.dll"
        "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_tensorrt.dll"
        $<TARGET_FILE_DIR:3D-Reconstruction>
    COMMENT "Copying ONNX Runtime DLLs"
)

# The remaining DLL copy commands reference local paths that don't exist on
# CI runners. vcpkg AppLocal deploys their vcpkg dependencies.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
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
# vcpkg AppLocal deploys its Qt runtime DLLs beside the executable.  The
# platform plugins must come from that exact Qt runtime AND from the matching
# build configuration: Debug/Release plugin DLLs are NOT ABI compatible with
# each other (different CRT / Qt build). vcpkg keeps them in two separate
# trees:
#   installed/<triplet>/Qt6/plugins           -> Release plugins (qwindows.dll)
#   installed/<triplet>/debug/Qt6/plugins     -> Debug plugins   (qwindowsd.dll)
#
# The previous version of this script always copied the Release tree, even
# for Debug builds. Loading a Release qwindows.dll into a Debug (Qt6Cored.dll)
# process fails to initialize the platform plugin, which is exactly the
# "This application failed to start because no Qt platform plugin could be
# initialized" / ASSERT "theme" crash seen in Debug builds.
#
# Fix: resolve BOTH trees at configure time, then let MSBuild pick the right
# one per configuration using a generator expression, so the copy always
# matches the configuration actually being built.
set(_qt_plugin_root_release "")
set(_qt_plugin_root_debug "")

set(_vcpkg_qt_plugin_root_release
    "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/Qt6/plugins")
set(_vcpkg_qt_plugin_root_debug
    "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/Qt6/plugins")

if(EXISTS "${_vcpkg_qt_plugin_root_release}/platforms/qwindows.dll")
    set(_qt_plugin_root_release "${_vcpkg_qt_plugin_root_release}")
endif()
if(EXISTS "${_vcpkg_qt_plugin_root_debug}/platforms/qwindowsd.dll")
    set(_qt_plugin_root_debug "${_vcpkg_qt_plugin_root_debug}")
endif()

# Fallback for configurations that do not deploy Qt through vcpkg: the Qt
# Creator kit ships a single set of plugins matching its own build type, so
# use the same root for both configurations (this preserves the previous
# behaviour when vcpkg's Qt6 plugin tree is not present).
if((NOT _qt_plugin_root_release OR NOT _qt_plugin_root_debug) AND QT_QMAKE_EXECUTABLE)
    get_filename_component(_qt_plugin_bin_dir "${QT_QMAKE_EXECUTABLE}" DIRECTORY)
    get_filename_component(_qt_plugin_kit_root "${_qt_plugin_bin_dir}" DIRECTORY)
    if(NOT _qt_plugin_root_release)
        set(_qt_plugin_root_release "${_qt_plugin_kit_root}/plugins")
    endif()
    if(NOT _qt_plugin_root_debug)
        set(_qt_plugin_root_debug "${_qt_plugin_kit_root}/plugins")
    endif()
endif()

# Fallback for CI (e.g. jurplel/install-qt-action / aqtinstall): neither
# vcpkg's Qt6 plugin tree nor QT_QMAKE_EXECUTABLE is set.  Derive the
# plugin directory from the Qt6::Core imported target location instead.
if((NOT _qt_plugin_root_release OR NOT _qt_plugin_root_debug) AND TARGET Qt6::Core)
    get_target_property(_qt6_core_location Qt6::Core LOCATION)
    if(_qt6_core_location)
        get_filename_component(_qt6_bin_dir "${_qt6_core_location}" DIRECTORY)
        get_filename_component(_qt6_prefix "${_qt6_bin_dir}" DIRECTORY)
        set(_qt6_aqt_plugin_dir "${_qt6_prefix}/plugins")
        if(EXISTS "${_qt6_aqt_plugin_dir}/platforms")
            if(NOT _qt_plugin_root_release)
                set(_qt_plugin_root_release "${_qt6_aqt_plugin_dir}")
            endif()
            if(NOT _qt_plugin_root_debug)
                set(_qt_plugin_root_debug "${_qt6_aqt_plugin_dir}")
            endif()
            message(STATUS "Found Qt plugins via Qt6::Core target: ${_qt6_aqt_plugin_dir}")
        endif()
    endif()
endif()

if(_qt_plugin_root_release AND _qt_plugin_root_debug)
    foreach(_qt_plugin_dir IN ITEMS platforms imageformats iconengines styles tls)
        if(EXISTS "${_qt_plugin_root_release}/${_qt_plugin_dir}" OR
           EXISTS "${_qt_plugin_root_debug}/${_qt_plugin_dir}")
            add_custom_command(TARGET 3D-Reconstruction POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "$<IF:$<CONFIG:Debug>,${_qt_plugin_root_debug},${_qt_plugin_root_release}>/${_qt_plugin_dir}"
                    "$<TARGET_FILE_DIR:3D-Reconstruction>/${_qt_plugin_dir}"
                COMMENT "Copying Qt ${_qt_plugin_dir} plugins ($<CONFIG> build)"
            )
        endif()
    endforeach()
else()
    message(WARNING
        "Qt platform plugins were not found (release root: '${_qt_plugin_root_release}', debug root: '${_qt_plugin_root_debug}').")
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