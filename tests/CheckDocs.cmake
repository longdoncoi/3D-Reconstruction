# ──────────────────────────────────────────────────────────────────────────────
# Documentation-consistency guard.
#
# The architecture docs are part of the contract: a number that nobody can
# reproduce is a claim, not evidence. This gate re-derives every machine-
# checkable number in Docs/ARCHITECTURE_SRC.md and tests/README.md from the
# tree and fails when the prose has drifted:
#
#   * the plugin-facing ABI surface count (the SAME closure definition the
#     hash uses — cmake/PluginInterface.cmake + cmake/InterfaceAbi.cmake);
#   * the PINNED_HEADERS list vs tests/mutation_check.py;
#   * MIN_MUTANTS vs the number of mutants actually defined (and the §1 /
#     README statements of that number);
#   * the architecture-guard rule count vs the rule headers in
#     tests/CheckArchitectureBoundaries.cmake;
#   * the CTest count vs the number of add_test() registrations, and the
#     executable/CMake-script split (a script test passes `-P`);
#   * the coverage snapshot: coverage_floors.json ↔ tests/README.md table ↔
#     the "tổng" row in Docs/ARCHITECTURE_SRC.md (file count, covered/total
#     lines and percentage).
#
# A docs change that updates a number without the tree (or the reverse) is
# exactly the drift this gate exists to catch.
#
# Chạy: cmake -DSOURCE_DIR=<repo-root> -P tests/CheckDocs.cmake
# ──────────────────────────────────────────────────────────────────────────────

if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required (repo root)")
endif()

set(_problems "")

# NOTE: a CMake function call does not concatenate adjacent quoted strings —
# `_problem("a" "b")` passes two arguments and only the first would be used.
# Every call below therefore builds ONE string (or uses ${ARGN}).
function(_problem)
    string(JOIN " " _message ${ARGN})
    set(_problems "${_problems}\n  - ${_message}" PARENT_SCOPE)
endfunction()

# ── Inputs ────────────────────────────────────────────────────────────────────
set(_arch "${SOURCE_DIR}/Docs/ARCHITECTURE_SRC.md")
set(_readme "${SOURCE_DIR}/tests/README.md")
set(_mutation "${SOURCE_DIR}/tests/mutation_check.py")
set(_floors "${SOURCE_DIR}/tests/coverage_floors.json")
set(_tests_cmake "${SOURCE_DIR}/tests/CMakeLists.txt")

foreach(_file IN ITEMS "${_arch}" "${_readme}" "${_mutation}" "${_floors}" "${_tests_cmake}")
    if(NOT EXISTS "${_file}")
        message(FATAL_ERROR "CheckDocs.cmake: missing input ${_file}")
    endif()
endforeach()

file(READ "${_arch}" _arch_text)
file(READ "${_readme}" _readme_text)
file(READ "${_mutation}" _mutation_text)
file(READ "${_floors}" _floors_text)
file(READ "${_tests_cmake}" _tests_cmake_text)

# Flatten the Markdown so a claim that wraps over several lines is still one
# regex match. CRLF is normalised first: the checkout may use either ending.
string(REPLACE "\r" "" _arch_flat "${_arch_text}")
string(REPLACE "\n" " " _arch_flat "${_arch_flat}")

# ── 1. ABI surface count ──────────────────────────────────────────────────────
# Reuse the ONE definition of the surface (the hash generator uses the same
# two modules), so the gate cannot drift from the thing it describes.
set(APP_REPO_ROOT "${SOURCE_DIR}")
include("${SOURCE_DIR}/cmake/PluginInterface.cmake")
include("${SOURCE_DIR}/cmake/InterfaceAbi.cmake")
collect_interface_surface(_surface APP_INTERFACE_ROOTS APP_INTERFACE_SEARCH_DIRS)
list(LENGTH _surface _surface_count)

if(NOT _arch_flat MATCHES "hash ABI \\(giờ gồm ([0-9]+) file")
    _problem("Docs/ARCHITECTURE_SRC.md no longer states the ABI surface count" 
             "(\"hash ABI (giờ gồm N file)\") — the docs gate cannot verify it")
else()
    if(NOT CMAKE_MATCH_1 EQUAL _surface_count)
        _problem("Docs/ARCHITECTURE_SRC.md says the ABI surface has ${CMAKE_MATCH_1}"
                 "file(s); the closure in cmake/PluginInterface.cmake has ${_surface_count}")
    endif()
endif()

# ── 2. PINNED_HEADERS: docs ↔ tests/mutation_check.py ─────────────────────────
if(NOT _mutation_text MATCHES "PINNED_HEADERS: tuple\\[str, \\.\\.\\.\\] = \\(([^)]*)\\)")
    _problem("tests/mutation_check.py: cannot parse PINNED_HEADERS")
else()
    set(_pinned_block "${CMAKE_MATCH_1}")
    string(REGEX MATCHALL "\"[^\"]+\"" _pinned_quoted "${_pinned_block}")
    set(_pinned "")
    foreach(_quoted IN LISTS _pinned_quoted)
        string(REGEX REPLACE "\"([^\"]+)\"" "\\1" _path "${_quoted}")
        list(APPEND _pinned "${_path}")
    endforeach()
    list(SORT _pinned)

    if(NOT _arch_flat MATCHES "Danh sách hiện tại:(.*)Gate 4 tự kiểm")
        _problem("Docs/ARCHITECTURE_SRC.md: cannot find the PINNED_HEADERS list"
                 "(\"Danh sách hiện tại:\" … \"Gate 4 tự kiểm\")")
    else()
        set(_doc_list "${CMAKE_MATCH_1}")
        string(REGEX MATCHALL "`[^`]+`" _doc_quoted "${_doc_list}")
        set(_doc_pinned "")
        foreach(_quoted IN LISTS _doc_quoted)
            string(REGEX REPLACE "`([^`]+)`" "\\1" _path "${_quoted}")
            list(APPEND _doc_pinned "${_path}")
        endforeach()
        list(SORT _doc_pinned)
        if(NOT "${_pinned}" STREQUAL "${_doc_pinned}")
            _problem("PINNED_HEADERS drifted between tests/mutation_check.py and"
                     "Docs/ARCHITECTURE_SRC.md; mutation_check.py: ${_pinned};"
                     "ARCHITECTURE_SRC.md: ${_doc_pinned}")
        endif()
    endif()
endif()

# ── 3. MIN_MUTANTS vs the mutants actually defined ────────────────────────────
if(NOT _mutation_text MATCHES "MIN_MUTANTS = ([0-9]+)")
    _problem("tests/mutation_check.py: cannot parse MIN_MUTANTS")
else()
    set(_min_mutants "${CMAKE_MATCH_1}")
    string(REGEX MATCHALL "\n    Mutant\\(" _mutant_calls "${_mutation_text}")
    list(LENGTH _mutant_calls _mutant_count)
    if(NOT _min_mutants EQUAL _mutant_count)
        _problem("tests/mutation_check.py defines ${_mutant_count} mutant(s) but"
                 "MIN_MUTANTS is ${_min_mutants} — update the floor in the same commit")
    endif()
    if(NOT _arch_flat MATCHES "\"${_min_mutants}/${_min_mutants} killed")
        _problem("Docs/ARCHITECTURE_SRC.md does not state the current mutation"
                 "result (\"${_min_mutants}/${_min_mutants} killed\")")
    endif()
    if(NOT _arch_flat MATCHES "\\(([0-9]+) mutant\\)")
        _problem("Docs/ARCHITECTURE_SRC.md §1 no longer states the mutant count"
                 "(\"(N mutant)\") — the gate cannot verify it")
    elseif(NOT CMAKE_MATCH_1 EQUAL _min_mutants)
        _problem("Docs/ARCHITECTURE_SRC.md §1 says ${CMAKE_MATCH_1} mutant(s);"
                 "MIN_MUTANTS is ${_min_mutants}")
    endif()
    if(NOT _readme_text MATCHES "\\*\\*${_min_mutants}/${_min_mutants} killed")
        _problem("tests/README.md does not state the current mutation reference"
                 "result (\"${_min_mutants}/${_min_mutants} killed\")")
    endif()
endif()

# ── 3b. Architecture-guard rule count ────────────────────────────────────────
# The rule number is the contract: the docs name how many groups of guards
# run, and a new rule that the prose does not mention is a guard nobody
# reviews. Rule "3b" is a sub-case of rule 3, so only distinct NUMBERS count.
set(_guard "${SOURCE_DIR}/tests/CheckArchitectureBoundaries.cmake")
if(NOT EXISTS "${_guard}")
    _problem("tests/CheckArchitectureBoundaries.cmake is missing")
else()
    file(READ "${_guard}" _guard_text)
    string(REGEX MATCHALL "\n# ── [0-9]+[a-z]?\\." _rule_headers "${_guard_text}")
    set(_rules "")
    foreach(_header IN LISTS _rule_headers)
        if(_header MATCHES "([0-9]+)[a-z]?\\.")
            list(APPEND _rules "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _rules)
    list(LENGTH _rules _rule_count)
    if(NOT _arch_flat MATCHES "([0-9]+) nhóm luật")
        _problem("Docs/ARCHITECTURE_SRC.md no longer states the architecture-guard"
                 "rule count (\"N nhóm luật\")")
    elseif(NOT CMAKE_MATCH_1 EQUAL _rule_count)
        _problem("Docs/ARCHITECTURE_SRC.md says the architecture guard has"
                 "${CMAKE_MATCH_1} rule group(s); tests/CheckArchitectureBoundaries.cmake"
                 "defines ${_rule_count}")
    endif()
endif()

# ── 4. CTest count vs add_test() registrations ────────────────────────────────
string(REGEX MATCHALL "\nadd_test\\(" _add_test_calls "${_tests_cmake_text}")
list(LENGTH _add_test_calls _test_count)
if(NOT _arch_flat MATCHES "\\| ${_test_count}/${_test_count} \\|")
    _problem("Docs/ARCHITECTURE_SRC.md does not state the current CTest count"
             "(${_test_count}/${_test_count}); tests/CMakeLists.txt registers"
             "${_test_count} test(s)")
endif()

# The suite is two kinds of test: an executable and a CMake script (-P). The
# docs state the split, so a script test that silently became an executable
# (or the reverse) is visible in the prose instead of hiding behind the total.
string(REGEX MATCHALL "-P \"\\$\\{CMAKE_CURRENT_SOURCE_DIR\\}/" _script_calls
       "${_tests_cmake_text}")
list(LENGTH _script_calls _script_count)
math(EXPR _exe_count "${_test_count} - ${_script_count}")
if(NOT _arch_flat MATCHES
       "\\(${_test_count} CTest entry: ${_exe_count} executable \\+ ${_script_count} CMake-script test\\)")
    _problem("Docs/ARCHITECTURE_SRC.md does not state the current CTest split"
             "(\"${_test_count} CTest entry: ${_exe_count} executable +"
             "${_script_count} CMake-script test\")")
endif()
if(NOT _readme_text MATCHES
       "${_test_count} CTest entries \\(${_exe_count} test executables \\+ ${_script_count} CMake-script tests\\)")
    _problem("tests/README.md does not state the current CTest split"
             "(\"${_test_count} CTest entries (${_exe_count} test executables +"
             "${_script_count} CMake-script tests)\")")
endif()

# ── 5. Coverage snapshot: JSON ↔ README table ↔ the "tổng" row ────────────────
string(JSON _floor_count LENGTH "${_floors_text}")
if(NOT _arch_flat MATCHES "${_floor_count} file\\(s\\) at or above their floors")
    _problem("Docs/ARCHITECTURE_SRC.md does not state the current measured-file"
             "count (\"${_floor_count} file(s) at or above their floors\")")
endif()

# Sum the per-file rows of the README's measured reference. The row shape is:
#   | `src/.../file.cpp` | 64.1 % (25/39) | 64 |
set(_readme_covered 0)
set(_readme_total 0)
set(_readme_rows 0)
file(STRINGS "${_readme}" _readme_lines)
foreach(_line IN LISTS _readme_lines)
    if(_line MATCHES "\\| `src/[^`]+\\.cpp` \\| [0-9.]+ % \\(([0-9]+)/([0-9]+)\\) \\| [0-9]+ \\|")
        math(EXPR _readme_covered "${_readme_covered} + ${CMAKE_MATCH_1}")
        math(EXPR _readme_total "${_readme_total} + ${CMAKE_MATCH_2}")
        math(EXPR _readme_rows "${_readme_rows} + 1")
    endif()
endforeach()

if(NOT _readme_rows EQUAL _floor_count)
    _problem("tests/README.md lists ${_readme_rows} measured file(s);"
             "coverage_floors.json has ${_floor_count} — the reference table"
             "and the floors must describe the same set")
endif()

if(NOT _arch_flat MATCHES "\\*\\*([0-9]+)\\*\\* \\| \\*\\*([0-9 ]+) / ([0-9 ]+)\\*\\* \\| \\*\\*([0-9,]+) %\\*\\*")
    _problem("Docs/ARCHITECTURE_SRC.md: cannot parse the coverage \"tổng\" row")
else()
    set(_doc_rows "${CMAKE_MATCH_1}")
    set(_doc_covered "${CMAKE_MATCH_2}")
    set(_doc_total "${CMAKE_MATCH_3}")
    set(_doc_percent "${CMAKE_MATCH_4}")
    string(REPLACE " " "" _doc_covered "${_doc_covered}")
    string(REPLACE " " "" _doc_total "${_doc_total}")

    if(NOT _doc_rows EQUAL _floor_count)
        _problem("Docs/ARCHITECTURE_SRC.md total row says ${_doc_rows} file(s);"
                 "coverage_floors.json has ${_floor_count}")
    endif()
    if(NOT _doc_covered EQUAL _readme_covered OR NOT _doc_total EQUAL _readme_total)
        _problem("Docs/ARCHITECTURE_SRC.md total row is ${_doc_covered}/${_doc_total};"
                 "the README reference table sums to ${_readme_covered}/${_readme_total}")
    endif()

    # One decimal, rounded half up, with the Vietnamese decimal comma.
    if(_readme_total GREATER 0)
        math(EXPR _pct10 "((1000 * ${_readme_covered}) + (${_readme_total} / 2)) / ${_readme_total}")
        math(EXPR _pct_whole "${_pct10} / 10")
        math(EXPR _pct_frac "${_pct10} % 10")
        set(_expected_percent "${_pct_whole},${_pct_frac}")
        if(NOT _doc_percent STREQUAL _expected_percent)
            _problem("Docs/ARCHITECTURE_SRC.md total percentage is ${_doc_percent} %;"
                     "the README rows compute to ${_expected_percent} %")
        endif()
    endif()
endif()

# ── Report ────────────────────────────────────────────────────────────────────
if(_problems)
    message(FATAL_ERROR "Documentation drift:${_problems}\n\n"
        "A number in the docs is part of the contract. Fix the tree or the "
        "prose in the SAME commit; do not delete the claim from the gate.")
endif()

message(STATUS "Documentation consistent: ABI surface ${_surface_count} file(s), "
               "${_test_count} CTest(s), ${_floor_count} coverage floor(s), "
               "${_mutant_count} mutant(s), ${_readme_covered}/${_readme_total} "
               "measured line(s)")
