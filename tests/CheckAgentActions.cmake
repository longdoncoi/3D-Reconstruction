# ──────────────────────────────────────────────────────────────────────────────
# Agent-action contract guard: Config/agent_action_manifest.json ↔ plugins.
#
# The manifest is what the AI service advertises to the model; the plugins are
# what actually performs the action. Nothing in the build links the two, so an
# action could be advertised and silently do nothing (or a handler could
# outlive its manifest entry and become dead code). This gate makes the
# contract explicit and bidirectional:
#
#   1. every action id in the manifest is handled by EXACTLY ONE plugin
#      (`action == "<id>"` in src/modules/**);
#   2. every handler literal names an action that exists in the manifest;
#   3. the manifest's `sample_data` block is non-empty and every fixture it
#      provides is a relative path (the AI service fills path/folder from it;
#      an absolute developer path would leak into the shipped contract).
#
# Chạy: cmake -DSOURCE_DIR=<repo-root> -P tests/CheckAgentActions.cmake
# ──────────────────────────────────────────────────────────────────────────────

if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required (repo root)")
endif()

set(_manifest_path "${SOURCE_DIR}/Config/agent_action_manifest.json")
if(NOT EXISTS "${_manifest_path}")
    message(FATAL_ERROR "CheckAgentActions.cmake: missing ${_manifest_path}")
endif()

file(READ "${_manifest_path}" _manifest)
string(JSON _action_count LENGTH "${_manifest}" actions)
if(_action_count EQUAL 0)
    message(FATAL_ERROR "the agent-action manifest advertises no actions")
endif()

# ── Manifest ids ──────────────────────────────────────────────────────────────
set(_manifest_ids "")
math(EXPR _last "${_action_count} - 1")
foreach(_index RANGE 0 ${_last})
    string(JSON _id GET "${_manifest}" actions ${_index} id)
    if(_id STREQUAL "")
        message(FATAL_ERROR "action #${_index} has no id")
    endif()
    list(APPEND _manifest_ids "${_id}")
endforeach()

# ── Handler literals in the plugins ───────────────────────────────────────────
file(GLOB_RECURSE _plugin_sources "${SOURCE_DIR}/src/modules/*.cpp")
set(_handled_ids "")
set(_handled_plugins "")
foreach(_source IN LISTS _plugin_sources)
    file(READ "${_source}" _content)
    string(REGEX MATCHALL "action[ \t]*==[ \t]*\"[^\"]+\"" _matches "${_content}")
    if(NOT _matches)
        continue()
    endif()
    get_filename_component(_plugin_dir "${_source}" DIRECTORY)
    get_filename_component(_plugin "${_plugin_dir}" NAME)
    foreach(_match IN LISTS _matches)
        string(REGEX REPLACE ".*\"([^\"]+)\".*" "\\1" _id "${_match}")
        list(APPEND _handled_ids "${_id}")
        list(APPEND _handled_plugins "${_plugin}")
    endforeach()
endforeach()

# ── 1 + 2: bidirectional coverage, exactly one PLUGIN per action ──────────────
# A plugin may compare the same id more than once (branch selection, deciding
# who owns the completion signal); what must be unique is the OWNER plugin.
# Two plugins claiming one action would both emit a completion.
set(_problems "")
foreach(_id IN LISTS _manifest_ids)
    list(FIND _handled_ids "${_id}" _first)
    if(_first EQUAL -1)
        list(APPEND _problems
            "manifest action \"${_id}\" has NO plugin handler (no `action == \"${_id}\"` in src/modules/**)")
    else()
        set(_owners "")
        set(_i 0)
        foreach(_handled IN LISTS _handled_ids)
            if(_handled STREQUAL _id)
                list(GET _handled_plugins ${_i} _owner)
                list(APPEND _owners "${_owner}")
            endif()
            math(EXPR _i "${_i} + 1")
        endforeach()
        list(REMOVE_DUPLICATES _owners)
        list(LENGTH _owners _owner_count)
        if(_owner_count GREATER 1)
            list(JOIN _owners ", " _owner_list)
            list(APPEND _problems
                "manifest action \"${_id}\" is handled by ${_owner_count} plugins (${_owner_list}), exactly one owner is required")
        endif()
    endif()
endforeach()

foreach(_id IN LISTS _handled_ids)
    if(NOT _id IN_LIST _manifest_ids)
        list(APPEND _problems
            "plugin handler \"${_id}\" is not advertised in Config/agent_action_manifest.json (dead handler)")
    endif()
endforeach()
list(REMOVE_DUPLICATES _problems)

# ── 3: the demo fixtures are relative, non-empty paths ────────────────────────
string(JSON _sample_type TYPE "${_manifest}" sample_data)
if(NOT _sample_type STREQUAL "OBJECT")
    list(APPEND _problems "the manifest has no \"sample_data\" object — the AI service cannot fill path/folder parameters")
else()
    string(JSON _sample_count LENGTH "${_manifest}" sample_data)
    if(_sample_count EQUAL 0)
        list(APPEND _problems "\"sample_data\" is empty — demo actions would be dispatched without inputs")
    endif()
    math(EXPR _sample_last "${_sample_count} - 1")
    foreach(_index RANGE 0 ${_sample_last})
        string(JSON _key MEMBER "${_manifest}" sample_data ${_index})
        string(JSON _value GET "${_manifest}" sample_data "${_key}")
        if(_value STREQUAL "")
            list(APPEND _problems "sample_data[\"${_key}\"] is empty")
        elseif(_value MATCHES "^[A-Za-z]:[/\\\\]" OR _value MATCHES "^/")
            list(APPEND _problems "sample_data[\"${_key}\"] = \"${_value}\" is an absolute path; the manifest is shipped, so fixtures must be relative")
        endif()
    endforeach()
endif()

if(_problems)
    list(JOIN _problems "\n  - " _report)
    message(FATAL_ERROR "Agent-action contract violations:\n  - ${_report}")
endif()

message(STATUS "Agent actions OK: ${_action_count} action(s), each handled by exactly one plugin")
