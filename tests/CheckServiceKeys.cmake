# ─────────────────────────────────────────────────────────────────────────────
# Service-key contract guard.
#
# The registry identifies a service by the ONE string in its `ServiceKey`.
# Two different interfaces that declare the same string collide: the second
# registerService() is refused (the first entry wins), and get<B>() would cast
# the A* stored under that key to B* — undefined behaviour. The registry cannot
# detect this safely from a void* (RTTI/mangled names are not stable across
# DLLs), so uniqueness is a contract this gate enforces by scanning the
# interface headers.
#
# The key must also follow the namespaced shape
#   com.3dreconstruction.service.<Interface>
# so a typo cannot create a second, silently empty service identity.
# ─────────────────────────────────────────────────────────────────────────────
if(NOT SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required")
endif()

file(GLOB _headers "${SOURCE_DIR}/src/include/I*.h")
list(SORT _headers)

set(_seen_keys "")
set(_seen_files "")
set(_declarations 0)
set(_problems "")

foreach(_header IN LISTS _headers)
    file(READ "${_header}" _text)
    # The declaration may wrap: ServiceKey = newline "com.…".
    string(REGEX MATCHALL
           "ServiceKey[ \t\r\n]*=[ \t\r\n]*\"([^\"]+)\""
           _decls "${_text}")
    get_filename_component(_name "${_header}" NAME)
    foreach(_decl IN LISTS _decls)
        string(REGEX REPLACE ".*\"([^\"]+)\".*" "\\1" _key "${_decl}")
        math(EXPR _declarations "${_declarations} + 1")
        list(FIND _seen_keys "${_key}" _index)
        if(NOT _index EQUAL -1)
            list(GET _seen_files ${_index} _other)
            string(APPEND _problems
                "\n  - ${_name} declares the same ServiceKey as ${_other}: ${_key}")
        else()
            list(APPEND _seen_keys "${_key}")
            list(APPEND _seen_files "${_name}")
        endif()
        if(NOT _key MATCHES "^com\\.3dreconstruction\\.service\\.I[A-Za-z0-9_]+$")
            string(APPEND _problems
                "\n  - ${_name}: ServiceKey does not follow"
                " com.3dreconstruction.service.<Interface>: ${_key}")
        endif()
    endforeach()
endforeach()

if(_declarations EQUAL 0)
    message(FATAL_ERROR
        "No ServiceKey declaration found in src/include/I*.h — the scan is broken")
endif()

if(_problems)
    message(FATAL_ERROR
        "Service-key contract broken:${_problems}\n\n"
        "Two interfaces sharing a key make registerService() refuse the second"
        " one, and get<T>() reinterpret the first service as the wrong type."
        " Fix the key; do not remove this gate.")
endif()

message(STATUS
    "Service keys OK: ${_declarations} declaration(s), all unique and namespaced")
