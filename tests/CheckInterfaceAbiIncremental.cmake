# ── Guard: an unchanged ABI hash must not rewrite the generated header ───────
#
# IPlugin.h includes InterfaceAbiHash.h, so its TIMESTAMP is a build input of
# every target (AppCore, all plugins, the executable and their AUTOMOC runs).
# Rewriting the header with the same value therefore recompiles the whole
# project (measured: ~2.5 minutes) even though no interface changed — a touch,
# a git checkout or a formatter-only edit used to do exactly that.
#
# This test pins both directions of the decision:
#
#   1. unchanged surface  → the generator must NOT touch the file
#   2. tampered output    → the generator must repair it
#
# (1) is measured on the file's mtime; the one-second sleep makes the
# filesystem timestamp distinguishable.
#
# Usage:
#   cmake -DSOURCE_DIR=<repo> -DWORK_DIR=<scratch dir>
#         -P tests/CheckInterfaceAbiIncremental.cmake

if(NOT DEFINED SOURCE_DIR OR NOT DEFINED WORK_DIR
   OR NOT DEFINED APP_SDK_FINGERPRINT)
    message(FATAL_ERROR
        "CheckInterfaceAbiIncremental.cmake needs SOURCE_DIR, WORK_DIR and "
        "APP_SDK_FINGERPRINT")
endif()

set(APP_REPO_ROOT "${SOURCE_DIR}")
include("${SOURCE_DIR}/cmake/PluginInterface.cmake")

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
set(_header "${WORK_DIR}/InterfaceAbiHash.h")
set(_generator "${SOURCE_DIR}/cmake/GenerateInterfaceAbi.cmake")

function(run_generator)
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
                "-DAPP_INTERFACE_ROOTS=${APP_INTERFACE_ROOTS}"
                "-DAPP_INTERFACE_SEARCH_DIRS=${APP_INTERFACE_SEARCH_DIRS}"
                "-DAPP_SDK_FINGERPRINT=${APP_SDK_FINGERPRINT}"
                "-DOUTPUT=${_header}"
                -P "${_generator}"
        RESULT_VARIABLE _rc
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE _out)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "GenerateInterfaceAbi.cmake failed:\n${_out}")
    endif()
endfunction()

# ── Generate once, then re-run over the same (unchanged) surface ────────────
run_generator()
if(NOT EXISTS "${_header}")
    message(FATAL_ERROR "the generator did not create ${_header}")
endif()
file(READ "${_header}" _expected)

execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 1)
file(TIMESTAMP "${_header}" _before "%s")

run_generator()
file(TIMESTAMP "${_header}" _after "%s")
if(NOT _before STREQUAL _after)
    message(FATAL_ERROR
        "the generator rewrote an UNCHANGED header (${_before} -> ${_after}); "
        "every target would rebuild for nothing")
endif()

# ── Non-vacuous: a tampered header must be repaired ─────────────────────────
file(APPEND "${_header}" "\n// tampered by CheckInterfaceAbiIncremental\n")
execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 1)
file(TIMESTAMP "${_header}" _before "%s")

run_generator()
file(TIMESTAMP "${_header}" _after "%s")
file(READ "${_header}" _repaired)
if(NOT _repaired STREQUAL _expected)
    message(FATAL_ERROR
        "the generator did not repair a tampered header:\n${_repaired}")
endif()
if(_before STREQUAL _after)
    message(FATAL_ERROR
        "the generator repaired the content but left the timestamp untouched")
endif()

message(STATUS "Interface ABI incremental write OK")
