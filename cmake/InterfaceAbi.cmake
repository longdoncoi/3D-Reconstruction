# ── Plugin-facing interface ABI hash ──────────────────────────────────────────
#
# The plugin contract is not a version number somebody has to remember to
# bump: it is the CONTENT of the headers a plugin is compiled against. These
# helpers compute a stable hash of that surface so the host can reject a
# plugin built against different interfaces instead of loading it against a
# stale ABI.
#
# "The plugin-facing surface" is the transitive closure of the #include graph
# of the roots defined in cmake/PluginInterface.cmake. The closure matters: a
# plugin sees IReconstructionService.h AND the DTO it includes
# (PointCloudData.h) — hashing only the I*.h files would miss a change to that
# DTO.

# Resolves the #include graph of one header to files that exist.
#
# Both spellings are part of the surface: a surface header can be included as
# "PointCloudData.h" OR <PointCloudData.h> (src/include is on the include
# path, so both compile). Resolving only the quoted form would let a
# one-character edit drop a DTO out of the hash — the plugin would then be
# built against a different struct with an unchanged fingerprint.
#
# An angle-bracket include is only resolved when it names a file INSIDE the
# surface search dirs; Qt/STL/system headers (<QString>, <vector>, <windows.h>)
# live elsewhere and are therefore not part of the surface.
#
# NOTE on the CMake calling convention: inside a function, ARGV0 is the NAME of
# the output variable, so the header is ARGV1 and ARGN still contains it.
function(resolve_interface_includes OUT_VAR)
    set(_header "${ARGV1}")
    set(_search_dirs ${ARGN})
    list(REMOVE_AT _search_dirs 0)   # drop the header itself
    file(READ "${_header}" _content)
    string(REGEX MATCHALL "#[ \t]*include[ \t]*[\"<]([^\">]+)[\">]" _matches "${_content}")
    set(_resolved "")
    foreach(_match IN LISTS _matches)
        string(REGEX REPLACE "^[^\"]*[\"<]([^\">]+)[\">].*$" "\\1" _include "${_match}")
        foreach(_dir IN LISTS _search_dirs)
            if(EXISTS "${_dir}/${_include}")
                list(APPEND _resolved "${_dir}/${_include}")
                break()
            endif()
        endforeach()
    endforeach()
    set(${OUT_VAR} ${_resolved} PARENT_SCOPE)
endfunction()

# The transitive closure of the quoted #include graph of the given roots.
# The roots and the search directories are passed as the NAMES of variables
# holding them, so the two lists stay distinguishable.
#
# The membership test is case-INSENSITIVE: on Windows "aimode.h" and
# "AIMode.h" are the same file, and a surface that counted it twice would
# hash differently depending on which spelling an include used.
function(collect_interface_surface OUT_VAR ROOTS_VAR SEARCH_DIRS_VAR)
    set(_roots ${${ROOTS_VAR}})
    set(_search_dirs ${${SEARCH_DIRS_VAR}})
    set(_surface ${_roots})
    set(_seen "")
    foreach(_root IN LISTS _roots)
        string(TOLOWER "${_root}" _key)
        list(APPEND _seen "${_key}")
    endforeach()
    set(_queue ${_roots})
    while(_queue)
        list(POP_FRONT _queue _head)
        resolve_interface_includes(_children "${_head}" ${_search_dirs})
        foreach(_child IN LISTS _children)
            string(TOLOWER "${_child}" _key)
            if(NOT _key IN_LIST _seen)
                list(APPEND _seen "${_key}")
                list(APPEND _surface "${_child}")
                list(APPEND _queue "${_child}")
            endif()
        endforeach()
    endwhile()
    list(SORT _surface)
    set(${OUT_VAR} ${_surface} PARENT_SCOPE)
endfunction()

# The stable hash of a header set: file NAME + file CONTENT of every file.
# Only the name is used (not the absolute path) so the value does not depend
# on where the repository is checked out.
function(compute_interface_abi_hash OUT_VAR)
    set(_input "")
    foreach(_file IN LISTS ARGN)
        get_filename_component(_name "${_file}" NAME)
        file(SHA256 "${_file}" _content_hash)
        string(APPEND _input "${_name}:${_content_hash}\n")
    endforeach()
    string(SHA256 _hash "${_input}")
    string(SUBSTRING "${_hash}" 0 16 _hash)
    set(${OUT_VAR} "0x${_hash}" PARENT_SCOPE)
endfunction()

# The SDK fingerprint: the versions of the libraries whose TYPES cross the
# plugin boundary (vtkSmartPointer/vtkRenderer in ISceneService, cv::Mat in
# IAIService, the ONNX Runtime ABI behind both). Bumping one of them changes
# struct layouts and inline implementations WITHOUT changing Qt or the
# compiler, so the toolchain fingerprint alone would accept a plugin built
# against a different SDK. Same 64-bit truncation as the interface hash.
function(compute_sdk_fingerprint OUT_VAR)
    string(JOIN "|" _versions ${ARGN})
    string(SHA256 _hash "${_versions}")
    string(SUBSTRING "${_hash}" 0 16 _hash)
    set(${OUT_VAR} "0x${_hash}" PARENT_SCOPE)
endfunction()
