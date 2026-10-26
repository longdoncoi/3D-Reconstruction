# ── Guard: the plugin-facing interface ABI hash must be honest ───────────────
#
# Two properties are checked:
#
#   1. FRESHNESS   — the generated header must equal a fresh computation over
#                    the current files. A stale generated header would let a
#                    plugin built against different interfaces load.
#   2. SENSITIVITY — changing the content of an interface header must change
#                    the hash. A hash that ignores the content would defeat
#                    the whole mechanism.
#
# Usage:
#   cmake -DSOURCE_DIR=<repo> -DINTERFACE_ABI_HEADER=<generated header>
#         -P tests/CheckInterfaceAbi.cmake

if(NOT DEFINED SOURCE_DIR OR NOT DEFINED INTERFACE_ABI_HEADER
   OR NOT DEFINED APP_SDK_FINGERPRINT)
    message(FATAL_ERROR
        "CheckInterfaceAbi.cmake needs SOURCE_DIR, INTERFACE_ABI_HEADER and "
        "APP_SDK_FINGERPRINT")
endif()

set(APP_REPO_ROOT "${SOURCE_DIR}")
include("${SOURCE_DIR}/cmake/PluginInterface.cmake")
include("${SOURCE_DIR}/cmake/InterfaceAbi.cmake")

collect_interface_surface(_surface APP_INTERFACE_ROOTS APP_INTERFACE_SEARCH_DIRS)
if(NOT _surface)
    message(FATAL_ERROR "the plugin-facing interface surface is empty — nothing to hash")
endif()
compute_interface_abi_hash(_expected ${_surface})

if(NOT EXISTS "${INTERFACE_ABI_HEADER}")
    message(FATAL_ERROR
        "the generated interface ABI header is missing: ${INTERFACE_ABI_HEADER}")
endif()
file(READ "${INTERFACE_ABI_HEADER}" _generated)
string(REGEX MATCH "APP_INTERFACE_ABI_HASH (0x[0-9a-f]+)ULL" _match "${_generated}")
if(NOT _match)
    message(FATAL_ERROR
        "the generated interface ABI header has no APP_INTERFACE_ABI_HASH: "
        "${INTERFACE_ABI_HEADER}")
endif()
set(_actual "${CMAKE_MATCH_1}")
if(NOT _actual STREQUAL _expected)
    message(FATAL_ERROR
        "the interface ABI hash is STALE: generated ${_actual}, "
        "the current interface surface hashes to ${_expected}")
endif()

# The SDK half of the fingerprint: the VTK/OpenCV/ONNX versions are part of
# the ABI (their types cross the plugin boundary), so the generated value must
# match the one this configure resolved.
string(REGEX MATCH "APP_TOOLCHAIN_SDK_FINGERPRINT (0x[0-9a-f]+)ULL" _sdk_match "${_generated}")
if(NOT _sdk_match)
    message(FATAL_ERROR
        "the generated interface ABI header has no "
        "APP_TOOLCHAIN_SDK_FINGERPRINT: ${INTERFACE_ABI_HEADER}")
endif()
if(NOT CMAKE_MATCH_1 STREQUAL APP_SDK_FINGERPRINT)
    message(FATAL_ERROR
        "the SDK fingerprint is STALE: generated ${CMAKE_MATCH_1}, this "
        "configure resolved ${APP_SDK_FINGERPRINT}")
endif()

# Sensitivity of the SDK half: the generator must actually EMIT the value it
# is given. A header that always writes 0 would leave a VTK/OpenCV bump
# invisible to the host's ABI gate.
set(_probe "${CMAKE_CURRENT_BINARY_DIR}/interface_abi_sdk_probe/InterfaceAbiHash.h")
file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/interface_abi_sdk_probe")
execute_process(
    COMMAND "${CMAKE_COMMAND}"
            "-DAPP_INTERFACE_ROOTS=${APP_INTERFACE_ROOTS}"
            "-DAPP_INTERFACE_SEARCH_DIRS=${APP_INTERFACE_SEARCH_DIRS}"
            "-DAPP_SDK_FINGERPRINT=0x1"
            "-DOUTPUT=${_probe}"
            -P "${SOURCE_DIR}/cmake/GenerateInterfaceAbi.cmake"
    RESULT_VARIABLE _probe_rc
    OUTPUT_VARIABLE _probe_out
    ERROR_VARIABLE _probe_out)
if(NOT _probe_rc EQUAL 0)
    message(FATAL_ERROR "the ABI generator failed for the SDK probe:\n${_probe_out}")
endif()
file(READ "${_probe}" _probe_header)
if(NOT _probe_header MATCHES "APP_TOOLCHAIN_SDK_FINGERPRINT 0x1ULL")
    message(FATAL_ERROR
        "the ABI generator ignores the SDK fingerprint it is given — a SDK "
        "bump would not change the value the host compares")
endif()

# ── Sensitivity: perturb one interface header, the hash must change ─────────
set(_scratch "${CMAKE_CURRENT_BINARY_DIR}/interface_abi_sensitivity")
file(REMOVE_RECURSE "${_scratch}")
file(MAKE_DIRECTORY "${_scratch}")

# Copy the surface, preserving the file names (the hash uses the name).
foreach(_file IN LISTS _surface)
    get_filename_component(_name "${_file}" NAME)
    configure_file("${_file}" "${_scratch}/${_name}" COPYONLY)
endforeach()
file(GLOB _perturbed_files "${_scratch}/*")
list(SORT _perturbed_files)
if(NOT _perturbed_files)
    message(FATAL_ERROR "could not copy the interface surface for the sensitivity check")
endif()
list(GET _perturbed_files 0 _victim)
file(READ "${_victim}" _victim_content)
file(WRITE "${_victim}" "${_victim_content}\n// perturbed by CheckInterfaceAbi\n")
compute_interface_abi_hash(_perturbed ${_perturbed_files})
if(_perturbed STREQUAL _expected)
    message(FATAL_ERROR
        "the interface ABI hash does NOT change when an interface header "
        "changes — the hash ignores the content")
endif()

list(LENGTH _surface _surface_size)

# ── The surface is the #include graph, not only the quoted spelling ─────────
# A surface header can be included as <PointCloudData.h> (src/include is on
# the include path, so the compiler accepts it). If the closure ignored that
# spelling, a one-character edit would drop a DTO out of the hash and the
# plugin would be built against a different struct with an unchanged
# fingerprint.
set(_ab_dir "${CMAKE_CURRENT_BINARY_DIR}/interface_abi_angle")
file(REMOVE_RECURSE "${_ab_dir}")
file(MAKE_DIRECTORY "${_ab_dir}")
file(WRITE "${_ab_dir}/Root.h" "#pragma once\n#include <Child.h>\n")
file(WRITE "${_ab_dir}/Child.h" "#pragma once\nstruct Child { int x; };\n")
set(_ab_roots "${_ab_dir}/Root.h")
set(_ab_dirs "${_ab_dir}")
collect_interface_surface(_ab_surface _ab_roots _ab_dirs)
list(FIND _ab_surface "${_ab_dir}/Child.h" _ab_index)
if(_ab_index EQUAL -1)
    message(FATAL_ERROR
        "an angle-bracket include of a surface header is not part of the "
        "surface — the closure only follows the quoted spelling")
endif()

message(STATUS "Interface ABI OK: ${_expected} over ${_surface_size} file(s)")
