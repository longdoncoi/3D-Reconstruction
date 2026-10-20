cmake_minimum_required(VERSION 3.19)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

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
# A caller-supplied toolchain other than the shared local toolchain (or Qt
# Creator's stale project-local helper) signals the CI/external case. The
# explicit local comparison is important on the second configure, when the
# local toolchain is already present in CMakeCache.txt.
if(NOT DEFINED PROJECT_VCPKG_ROOT)
    if(DEFINED ENV{PROJECT_VCPKG_ROOT})
        file(TO_CMAKE_PATH "$ENV{PROJECT_VCPKG_ROOT}" _project_vcpkg_root_default)
    elseif(DEFINED ENV{VCPKG_ROOT})
        file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" _project_vcpkg_root_default)
    elseif(EXISTS "F:/PROJECTS/QT/vcpkg")
        set(_project_vcpkg_root_default "F:/PROJECTS/QT/vcpkg")
    else()
        set(_project_vcpkg_root_default "")
    endif()
    set(PROJECT_VCPKG_ROOT "${_project_vcpkg_root_default}" CACHE PATH
        "Shared vcpkg root used by this project")
endif()
set(_project_vcpkg_toolchain
    "${PROJECT_VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
file(TO_CMAKE_PATH "${CMAKE_TOOLCHAIN_FILE}" _supplied_vcpkg_toolchain)
file(TO_CMAKE_PATH "${_project_vcpkg_toolchain}" _local_vcpkg_toolchain)
if(DEFINED CMAKE_TOOLCHAIN_FILE
   AND NOT _supplied_vcpkg_toolchain MATCHES "vcpkg-dependencies"
   AND NOT _supplied_vcpkg_toolchain STREQUAL _local_vcpkg_toolchain)
    set(_external_vcpkg_toolchain TRUE)
    message(STATUS "Using externally supplied vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
elseif(DEFINED ENV{CI} OR DEFINED ENV{GITHUB_ACTIONS})
    set(_external_vcpkg_toolchain TRUE)
    message(STATUS "CI environment detected. Treating toolchain as external.")
else()
    set(_external_vcpkg_toolchain FALSE)
endif()

if(NOT _external_vcpkg_toolchain)
    # Qt Creator's package-manager helper is disabled in
    # QtCreatorPackageManager.cmake so it cannot replace this toolchain with a
    # project-local copy and trigger a second manifest installation.
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

    # AppLocal starts a PowerShell dependency scan after every executable and
    # DLL. With the application, plugins, and tests this dominated even no-op
    # local builds. Runtime deployment remains opt-in for dependency refreshes.
    option(DEV_AUTO_DEPLOY_VCPKG_RUNTIME
        "Run vcpkg AppLocal after every local binary target" OFF)
    set(VCPKG_APPLOCAL_DEPS ${DEV_AUTO_DEPLOY_VCPKG_RUNTIME} CACHE BOOL
        "Automatically copy vcpkg runtime dependencies after each target"
        FORCE)
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

# ─── Dependency package locations ────────────────────────────────────────────
# Local defaults only when the conventional install path exists. Callers can
# override with -D<PACKAGE>_DIR=... or the matching environment variable, so a
# fresh machine never has to edit this file.
function(set_cache_dir var default)
    if(NOT "${${var}}")
        if(DEFINED ENV{${var}} AND NOT "$ENV{${var}}" STREQUAL "")
            file(TO_CMAKE_PATH "$ENV{${var}}" _env_dir)
            set(${var} "${_env_dir}" CACHE PATH "" FORCE)
        elseif(EXISTS "${default}")
            set(${var} "${default}" CACHE PATH "" FORCE)
        endif()
    endif()
endfunction()

set_cache_dir(VTK_DIR "C:/Program Files/VTK/lib/cmake/vtk-9.6")
set_cache_dir(OpenCV_DIR "C:/opencv/build")
set_cache_dir(PCL_DIR "C:/Program Files/PCL 1.15.1/cmake")
set_cache_dir(Eigen3_DIR "C:/Program Files/PCL 1.15.1/3rdParty/Eigen3/share/eigen3/cmake")

project(3D-Reconstruction LANGUAGES CXX)

# ─── Build Performance Options ────────────────────────────────────────────────
option(ENABLE_UNITY_BUILD "Enable Unity (Jumbo) build for faster compilation" ON)
option(BUILD_TESTING "Build unit-test executables as part of the default build" OFF)
option(CLEAN_STALE_DLLS "Remove stale DLLs from Release directory at configure time" OFF)
set(CMAKE_UNITY_BUILD ${ENABLE_UNITY_BUILD})
if(ENABLE_UNITY_BUILD)
    # PCH keeps the heavy Qt/PCL/VTK/OpenCV headers out of each source parse.
    # Eight retains enough independent compiler work for /MP without creating
    # translation units so large that MSVC loses parallelism.
    set(CMAKE_UNITY_BUILD_BATCH_SIZE 8)
endif()
set(CMAKE_AUTOGEN_PARALLEL 8)

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
# qt_standard_project_setup() only turns on AUTOMOC/AUTOUIC. Without AUTORCC the
# .qrc files listed in target_sources() are ignored by the generator (they ended
# up as plain "None" items), so ":/app_icon.png" and the language flags in
# ":/flags/..." never existed at runtime and the icons stayed blank.
set(CMAKE_AUTORCC ON)

# ─── 2. VTK ───────────────────────────────────────────────────────────────────
set(VTK_MODULE_ENABLE_VTK_RenderingVulkan NO)
# Prefer caller-provided package locations. Local defaults are applied only
# when the conventional install path exists, so fresh machines can rely on
# CMAKE_PREFIX_PATH, vcpkg, or explicit -D*_DIR values without editing this file.
if(NOT VTK_DIR AND EXISTS "C:/Program Files/VTK/lib/cmake/vtk-9.6")
    set(VTK_DIR "C:/Program Files/VTK/lib/cmake/vtk-9.6" CACHE PATH "Path to VTK" FORCE)
endif()
find_package(VTK REQUIRED COMPONENTS
    CommonCore
    RenderingOpenGL2
    GUISupportQt
    ViewsQt
    IOGeometry
    IOImport
    IOPLY
    FiltersCore
    IOImage
    IOXML
    RenderingCore
    ImagingCore
    RenderingVolume
    RenderingVolumeOpenGL2
    RenderingAnnotation
)

# ─── 5. Paths & Outputs ────────────────────────────────────────────────────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins)

# ─── 5b. Debug symbols in Release ──────────────────────────────────────────────
# The coverage gate (tests/coverage_gate.py) maps executed addresses to lines
# through the PDB, and crash reports from the field are only useful with
# symbols too. /DEBUG alone turns off /OPT:REF and /OPT:ICF, so keep the
# Release optimization behaviour explicit — codegen stays identical to a
# normal Release link.
if(MSVC)
    # The compiler side too: without /Zi the Release PDB only carries the
    # debug info merged from the CRT/STL libraries, so no project line can be
    # mapped. The /Zi guard keeps repeated configures idempotent.
    foreach(_config IN ITEMS RELEASE RELWITHDEBINFO)
        if(NOT "${CMAKE_CXX_FLAGS_${_config}}" MATCHES "/Zi")
            set(CMAKE_CXX_FLAGS_${_config} "${CMAKE_CXX_FLAGS_${_config}} /Zi"
                CACHE STRING "C++ flags (Release ships with symbols)" FORCE)
        endif()
    endforeach()

    # The Visual Studio generator reads these cache entries when it emits
    # <GenerateDebugInformation>; a plain normal-variable append would be
    # ignored. The /DEBUG guard keeps repeated configures idempotent.
    foreach(_config IN ITEMS RELEASE RELWITHDEBINFO)
        foreach(_var IN ITEMS
                CMAKE_EXE_LINKER_FLAGS_${_config}
                CMAKE_SHARED_LINKER_FLAGS_${_config})
            if(NOT "${${_var}}" MATCHES "/DEBUG")
                set(${_var} "${${_var}} /DEBUG /OPT:REF /OPT:ICF"
                    CACHE STRING "Linker flags (Release ships with symbols)" FORCE)
            endif()
        endforeach()
    endforeach()
endif()

# ─── 5b. Build Optimizations ───────────────────────────────────────────────────
# ─── 3. OpenCV ────────────────────────────────────────────────────────────────
if(NOT OpenCV_DIR AND EXISTS "C:/opencv/build")
    set(OpenCV_DIR "C:/opencv/build" CACHE PATH "Path to OpenCV CMake files" FORCE)
endif()
find_package(OpenCV REQUIRED)

# ─── 4. PCL ───────────────────────────────────────────────────────────────────
# Suppress MSVC-unsupported developer warning emitted by PCL's FindPCL.cmake
set(PCL_DISABLE_MSVC_DEPRECATION_WARNINGS ON)
if(NOT PCL_DIR AND EXISTS "C:/Program Files/PCL 1.15.1/cmake")
    set(PCL_DIR "C:/Program Files/PCL 1.15.1/cmake" CACHE PATH "Path to PCL CMake files" FORCE)
endif()
find_package(PCL REQUIRED COMPONENTS common filters io surface features search)

# ─── 4b. Eigen (dùng cho MultiViewTriangulator) ──────────────────────────────
if(NOT Eigen3_DIR AND EXISTS "C:/Program Files/PCL 1.15.1/3rdParty/Eigen3/share/eigen3/cmake")
    set(Eigen3_DIR "C:/Program Files/PCL 1.15.1/3rdParty/Eigen3/share/eigen3/cmake" CACHE PATH "Path to Eigen3 CMake files" FORCE)
endif()
find_package(Eigen3 CONFIG REQUIRED)

# ─── 5. ONNX Runtime ─────────────────────────────────────────────────────────
include(FetchContent)
set(ONNXRUNTIME_VERSION "1.20.1")
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}.zip")
# set(ONNXRUNTIME_URL "F:/onnxruntime-win-x64-gpu-1.20.1.zip")
FetchContent_Declare(
    onnxruntime
    URL ${ONNXRUNTIME_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime)

# ─── 5b. TensorRT ────────────────────────────────────────────────────────────
# onnxruntime_providers_tensorrt.dll loads TRT DLLs dynamically at runtime.
# They MUST be present in the executable directory or on PATH.
if(NOT DEFINED TENSORRT_ROOT)
    if(DEFINED ENV{TENSORRT_ROOT})
        file(TO_CMAKE_PATH "$ENV{TENSORRT_ROOT}" _tensorrt_root_default)
    elseif(EXISTS "C:/TensorRT-10.16.1.11")
        set(_tensorrt_root_default "C:/TensorRT-10.16.1.11")
    else()
        set(_tensorrt_root_default "")
    endif()
    set(TENSORRT_ROOT "${_tensorrt_root_default}" CACHE PATH "Path to TensorRT installation")
endif()
set(TENSORRT_BIN_DIR "${TENSORRT_ROOT}/bin")

# ─── 6. AUTOUIC search paths ─────────────────────────────────────────────────
set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/app)

# ─── 7b. Shared Precompiled Header (PCH) Target ────────────────────────────────
# We create a clean STATIC library to build the PCH. This avoids baking in macros 
# like APP_LIBRARY or _EXPORTS, so plugins can reuse it safely without link errors.
add_library(SharedPCH STATIC src/pch_dummy.cpp)
target_include_directories(SharedPCH PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/scene
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/viewer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/settings
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/shell
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/reconstruction
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/ai
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/mail
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${onnxruntime_SOURCE_DIR}/include
)
target_compile_definitions(SharedPCH PUBLIC
    ${VTK_DEFINITIONS}
    ${PCL_DEFINITIONS}
    PCL_SILENCE_MALLOC_WARNING=1
)
if(MSVC)
    # /W4: first-party warnings; /external:anglebrackets + /external:W0 keeps
    # third-party SDK headers (VTK/OpenCV/PCL/Qt) out of the warning stream.
    # /WX: first-party warnings are build failures; the codebase is clean at /W4
    # and the flag documents the gate for future code.
    target_compile_options(SharedPCH PUBLIC /utf-8 /MP /FS /W4 /WX /external:anglebrackets /external:W0)
endif()
target_precompile_headers(SharedPCH PUBLIC src/pch.h)
# Heavy PCH — consumers self-select: only the exe/AppCore and the plugins whose
# sources genuinely use VTK/OpenCV/PCL link this target (the architecture guard
# enforces it), so the heavy libs may stay PUBLIC here: the PCL/VTK/OpenCV
# package definitions (BOOST_*, __AVX__, _OPENMP, …) must match the PCH binary
# exactly or MSVC fires C4651 on every unity TU. Light plugins link
# SharedPCHLight instead and never see any of this.
target_link_libraries(SharedPCH
    PUBLIC
        Qt6::Core
        Qt6::Widgets
        Qt6::OpenGL
        Qt6::Network
        Qt6::Concurrent
        Qt6::OpenGLWidgets
        ${VTK_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        Eigen3::Eigen
)

# ─── 7b'. SharedPCHLight — PCH nhẹ cho plugin không dùng VTK/OpenCV/PCL ────────
# pch_light.h is Qt/STL only: a light plugin compiles without a single heavy
# SDK include path. The Qt module set mirrors AppCore's PUBLIC interface so
# MSVC's precompiled-header definition parity (C4651) holds for light
# consumers. Which plugins must use this target vs SharedPCH is encoded in
# tests/CheckArchitectureBoundaries.cmake (derived from their sources).
add_library(SharedPCHLight STATIC src/pch_light_dummy.cpp)
if(MSVC)
    target_compile_options(SharedPCHLight PUBLIC /utf-8 /MP /FS /W4 /WX /external:anglebrackets /external:W0)
endif()
target_precompile_headers(SharedPCHLight PUBLIC src/pch_light.h)
target_link_libraries(SharedPCHLight
    PUBLIC
        Qt6::Core
        Qt6::Widgets
        Qt6::OpenGL
        Qt6::Network
        Qt6::Concurrent
        Qt6::OpenGLWidgets
)

# ─── 7c. Executable ────────────────────────────────────────────────────────────
# ─── 7a. AppCore — foundation shared library ───────────────────────────────────
# core (interfaces + kernel: UserManager, SignalBus, AppConfig, …) and utils
# (StyleManager, IconFactory, AbstractRibbonPanel, …) form the stable ABI that
# the exe, every plugin and every test import. Building them as their own
# shared library removes the exe↔plugin circular link (plugins used to link
# the exe just for APP_EXPORT symbols) and gives one canonical set of Qt
# link dependencies for the whole product.
add_library(AppCore SHARED
    # core
    src/core/Global.h
    src/core/AppConfig.h
    src/core/AppConfig.cpp
    src/services/ai/AgentActionManifest.h
    src/services/ai/AgentActionManifest.cpp
    src/core/SecretVault.h
    src/core/SecretVault.cpp
    src/core/AtRestCipher.h
    src/core/AtRestCipher.cpp
    src/core/SignalBus.h
    src/core/SignalBus.cpp
    src/core/LanguageManager.h
    src/core/LanguageManager.cpp
    src/core/CryptoUtils.h
    src/core/CryptoUtils.cpp
    src/core/LicenseKeyLedger.h
    src/core/LicenseKeyLedger.cpp
    src/core/OtpStore.h
    src/core/OtpStore.cpp
    src/core/UserAccountStore.h
    src/core/UserAccountStore.cpp
    src/core/UserSettingsStore.h
    src/core/UserSettingsStore.cpp
    src/core/UserManager.h
    src/core/UserManager.cpp
    src/core/IOutgoingMail.h
    src/core/ServiceRegistry.h
    src/core/IReconstructionService.h
    src/core/IAIService.h
    src/core/IAIModelLoadService.h
    src/core/IMailService.h
    src/core/IAppContext.h
    src/core/IAIAssistantService.h
    src/core/IUserManager.h
    src/core/IUserPreferences.h
    src/core/ILanguageService.h
    src/core/ISceneService.h
    src/core/ISettingsService.h
    src/core/IViewerService.h
    src/core/IPlugin.h
    src/core/IShellService.h
    src/core/IProgressDialog.h
    src/core/IAppPaths.h
    src/core/AIMode.h
    src/core/AppConstants.h
    src/core/PluginDependencyResolver.h
    src/core/PluginDependencyResolver.cpp
    src/core/PluginHost.h
    src/core/PluginHost.cpp
    # shell
    src/services/shell/ShellService.h
    src/services/shell/ShellService.cpp
    # utils
    src/utils/StyleManager.h
    src/utils/StyleManager.cpp
    src/utils/AbstractRibbonPanel.h
    src/utils/AbstractRibbonPanel.cpp
    src/utils/Logger.h
    src/utils/Logger.cpp
    src/utils/CustomProgressDialog.h
    src/utils/CustomProgressDialog.cpp
    src/utils/ModernDialog.h
    src/utils/ModernDialog.cpp
    src/utils/ModernMessageBox.h
    src/utils/ModernMessageBox.cpp
    src/utils/HtmlUtilities.h
    src/utils/HtmlUtilities.cpp
    src/utils/FileUtilities.h
    src/utils/FileUtilities.cpp
    src/utils/IconFactory.h
    src/utils/IconFactory.cpp
)
target_precompile_headers(AppCore REUSE_FROM SharedPCH)
# PUBLIC surface = the plugin-facing ABI only: interfaces (core) + shared UI
# utilities. Service implementation headers are NOT part of that surface.
target_include_directories(AppCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
)
# Implementation-only include paths for AppCore's own sources (SceneService.h,
# ShellService.h, …). PRIVATE on purpose: plugins/tests must use the I*.h
# interfaces; the build graph now enforces the boundary that
# tests/CheckArchitectureBoundaries.cmake double-checks. The heavy SDK include
# paths are PRIVATE for the same reason: the plugin-facing interfaces use
# forward declarations / ABI-neutral DTOs, so a light plugin must not need
# VTK/OpenCV/PCL/ONNX headers to include them.
target_include_directories(AppCore PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/scene
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/viewer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/settings
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/shell
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/reconstruction
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/ai
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/mail
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${onnxruntime_SOURCE_DIR}/include
)
# APP_LIBRARY private to AppCore → APP_EXPORT = dllexport here, dllimport in
# every consumer (exe / plugins / tests) that links this target.
target_compile_definitions(AppCore PRIVATE APP_LIBRARY)
# PCH flag parity: the reused PCH (REUSE_FROM SharedPCH) is compiled with
# SharedPCH's definitions, but AppCore does not link that target — so it must
# repeat them on its own TUs. Otherwise MSVC fires C4651 ("specified for
# precompiled header but not for current compile") on every unity source.
# PRIVATE: consumers that parse heavy headers get these from SharedPCH
# themselves; light plugins never need them.
target_compile_definitions(AppCore PRIVATE
    ${VTK_DEFINITIONS}
    ${PCL_DEFINITIONS}
    PCL_SILENCE_MALLOC_WARNING=1
)
if(MSVC)
    target_compile_options(AppCore PRIVATE /utf-8 /MP /FS /W4 /WX /external:anglebrackets /external:W0)
endif()
# PUBLIC = Qt only (the ABI is made of Qt types). The heavy SDKs are PRIVATE:
# a plugin that does not itself use VTK/OpenCV/PCL/ONNX must not link (nor
# load) them. Targets that do use the heavy SDK declare it themselves — the
# exe, ViewerPlugin, AIProcessorPlugin — and the rule is enforced in
# tests/CheckArchitectureBoundaries.cmake.
target_link_libraries(AppCore
    PUBLIC
        Qt6::Core
        Qt6::Widgets
        Qt6::OpenGL
        Qt6::Network
        Qt6::Concurrent
        Qt6::OpenGLWidgets
    PRIVATE
        ${VTK_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        Eigen3::Eigen
        ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib
)
if(WIN32)
    # crypt32: the DPAPI vault (SecretVault).
    # bcrypt: the AES-256-GCM envelope of the at-rest cipher (AtRestCipher).
    target_link_libraries(AppCore PRIVATE crypt32 bcrypt)
endif()

qt_add_executable(3D-Reconstruction WIN32 MACOSX_BUNDLE)
target_precompile_headers(3D-Reconstruction REUSE_FROM SharedPCH)
# ─── Source subdirectories ───────────────────────────────────────────────────
add_subdirectory(src/app)
add_subdirectory(src/services)
# ─── 8. Include directories ───────────────────────────────────────────────────
# PRIVATE on purpose: the exe (composition root) is the only target allowed to
# reach every layer directly — src/app plus the concrete service headers it
# wires together. Plugins import core/utils transitively from AppCore and can
# no longer resolve service headers at all (they are PRIVATE on AppCore).
target_include_directories(3D-Reconstruction PRIVATE
    # App shell
    ${CMAKE_CURRENT_SOURCE_DIR}/src/app

    # Concrete service headers (composition-root wiring only)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/scene
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/viewer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/settings
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/shell
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/reconstruction
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/ai
    ${CMAKE_CURRENT_SOURCE_DIR}/src/services/mail

    # Third-party (core/utils arrive transitively through AppCore)
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${onnxruntime_SOURCE_DIR}/include
)

# ─── 9. Compile definitions & options ─────────────────────────────────────────
target_compile_definitions(3D-Reconstruction PUBLIC
    ${VTK_DEFINITIONS}
    ${PCL_DEFINITIONS}
    PCL_SILENCE_MALLOC_WARNING=1
)
# (APP_LIBRARY intentionally NOT defined here: core/utils classes are imported
# from AppCore.dll instead of being dllexport'ed from this executable.)
if(MSVC)
    # /MP enables multi-core compilation. Since heavy PCL/VTK headers are now in PCH,
    # cl.exe uses much less memory per file, avoiding the previous heap exhaustion issues.
    target_compile_options(3D-Reconstruction PRIVATE /utf-8 /MP /FS /W4 /WX /external:anglebrackets /external:W0)
endif()

# ─── 10. Link libraries ───────────────────────────────────────────────────────
target_link_libraries(3D-Reconstruction
    PRIVATE
        AppCore
        Qt6::Core
        Qt6::Widgets
        Qt6::OpenGL
        Qt6::Network
        Qt6::Concurrent
        Qt6::OpenGLWidgets
        ${VTK_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        Eigen3::Eigen
        ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib
)

set_target_properties(3D-Reconstruction PROPERTIES WIN32_EXECUTABLE ON)
# The application target is compiled without the unity build on purpose: the
# coverage gate maps executed lines through the PDB, and a source file that
# shares a unity translation unit with its neighbours can be inlined away —
# no code is emitted, the PDB does not mention the file and the exporter
# reports an empty class.  That makes whole translation units invisible to
# the measurement (the GUI shell's Main.cpp / MainWindow.cpp were).  One
# translation unit per file keeps every TU visible; the PCH still absorbs the
# cost of the heavy VTK/PCL headers.
set_target_properties(3D-Reconstruction PROPERTIES UNITY_BUILD OFF)

# ─── 11. Post-build: Copy DLLs ───────────────────────────────────────────────
# AppCore.dll must sit next to the executable (the exe and every plugin import
# its symbols; plugins resolve it from the executable directory).
add_custom_command(TARGET AppCore POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:AppCore>
        $<TARGET_FILE_DIR:3D-Reconstruction>
    COMMENT "Copying AppCore.dll next to the executable"
)
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

# The remaining DLL copy commands reference optional local SDK installs that
# don't exist on CI runners. vcpkg AppLocal deploys vcpkg dependencies.
option(ENABLE_LOCAL_RUNTIME_DLL_COPY
    "Copy runtime DLLs from locally installed TensorRT/PCL/OpenNI folders" ON)
if(ENABLE_LOCAL_RUNTIME_DLL_COPY AND NOT _external_vcpkg_toolchain)
    if(NOT DEFINED PCL_VTK_RUNTIME_BIN_DIR AND EXISTS "C:/Program Files/PCL 1.15.1/3rdParty/VTK/bin")
        set(PCL_VTK_RUNTIME_BIN_DIR "C:/Program Files/PCL 1.15.1/3rdParty/VTK/bin"
            CACHE PATH "Directory containing VTK runtime DLLs bundled with PCL")
    endif()
    if(NOT DEFINED PCL_RUNTIME_BIN_DIR AND EXISTS "C:/Program Files/PCL 1.15.1/bin")
        set(PCL_RUNTIME_BIN_DIR "C:/Program Files/PCL 1.15.1/bin"
            CACHE PATH "Directory containing PCL runtime DLLs")
    endif()
    if(NOT DEFINED OPENNI2_TOOLS_DIR AND EXISTS "C:/Program Files/OpenNI2/Tools")
        set(OPENNI2_TOOLS_DIR "C:/Program Files/OpenNI2/Tools"
            CACHE PATH "Directory containing OpenNI2 runtime DLLs")
    endif()
    # Copy TensorRT DLLs — required by onnxruntime_providers_tensorrt.dll at runtime.
    # Without these, the TRT provider crashes with access violation (0xc0000005).
    # Only the core runtime DLLs are needed; builder_resource DLLs are loaded lazily.
    if(EXISTS "${TENSORRT_BIN_DIR}")
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

    endif()

    # Copy missing PCL, VTK 9.4, and OpenNI2 DLLs required at runtime
    # Consolidated into a single PowerShell invocation to avoid repeated process startup overhead
    if(EXISTS "${PCL_VTK_RUNTIME_BIN_DIR}" OR EXISTS "${PCL_RUNTIME_BIN_DIR}" OR EXISTS "${OPENNI2_TOOLS_DIR}")
        add_custom_command(TARGET 3D-Reconstruction POST_BUILD
            COMMAND powershell -NoProfile -ExecutionPolicy Bypass -Command "$destDir = '$<TARGET_FILE_DIR:3D-Reconstruction>'; $sources = @(@{ Path='${PCL_VTK_RUNTIME_BIN_DIR}'; Filter='vtk*.dll'; Where=$null }, @{ Path='${PCL_RUNTIME_BIN_DIR}'; Filter=$null; Where={ $_.Name -like 'pcl*.dll' -or $_.Name -like 'boost*.dll' -or $_.Name -like 'flann*.dll' -or $_.Name -like 'qhull*.dll' } }, @{ Path='${OPENNI2_TOOLS_DIR}'; Filter='*.dll'; Where=$null }); foreach ($s in $sources) { if ($s.Path -and (Test-Path $s.Path)) { $items = if ($s.Filter) { Get-ChildItem -Path $s.Path -Filter $s.Filter } else { Get-ChildItem -Path $s.Path }; if ($s.Where) { $items = $items | Where-Object $s.Where }; $items | ForEach-Object { $target = Join-Path $destDir $_.Name; if (-not (Test-Path $target) -or (Get-Item $target).Length -ne $_.Length) { Copy-Item $_.FullName -Destination $destDir -Force } } } }"
            COMMENT "Copying PCL, VTK, and OpenNI dependencies (single PS invocation)"
        )
    endif()
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

# CI release-only triplet (VCPKG_BUILD_TYPE=release) does not produce a debug
# Qt plugin tree. Reuse the release root so the "platform plugins not found"
# warning is not emitted when the debug tree is legitimately absent.
if(_external_vcpkg_toolchain AND _qt_plugin_root_release AND NOT _qt_plugin_root_debug)
    set(_qt_plugin_root_debug "${_qt_plugin_root_release}")
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

# ─── 13. Manifest for cross-language contract (Phase 2) ──────────────────────
# src/core/AppManifest.json is an authored, tracked contract file: it is read by
# AIComputerVision/tests/test_contract.py and shipped with the app. It is
# deliberately NOT generated into the source tree — the former template carried
# no @VARIABLE@ substitutions, so configure_file() only rewrote an identical
# file on every configure (dirtying the working tree for no benefit).

# ─── 13. Plugins ─────────────────────────────────────────────────────────────
add_subdirectory(src/modules)

# ─── 13. Testing ─────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# ─── 14. Install rules ───────────────────────────────────────────────────────
# Portable install tree: exe + AppCore in bin, plugins in bin/plugins and the
# shipped resources next to them. Callers may override CMAKE_INSTALL_PREFIX.
include(GNUInstallDirs)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH
    "Default install prefix for this project")

install(TARGETS 3D-Reconstruction AppCore
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(TARGETS
        AIAssistantPlugin
        AIProcessorPlugin
        ReconstructionPlugin
        SendMailPlugin
        UserAuthPlugin
        ViewerPlugin
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/plugins
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/plugins)

install(DIRECTORY translations
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
