"""One owner for "resolve a program name to the one binary that may run".

The platform resolves a program in three places — the command sandbox, the
git tools and the LSP client — and each of them used to do it its own way.
The sandbox resolved it correctly (ADR 0034): it walks the ``PATH`` entries
itself and never looks at the current directory. The other two did not:

* ``shutil.which`` **prepends the current directory** to the search path on
  Windows, so a binary planted next to the process wins over the operator's
  own tool — and the value it returns is then *relative*, not the absolute
  hit the gate promised. The allow-list was therefore not enforced, and the
  refusal for "a name PATH does not carry" never fired, because ``which``
  had already found the planted file;
* one caller threw the verified absolute hit away with ``resolved or "git"``,
  which silently reverted to the operating system's search order — the very
  order the gate exists to remove.

This module owns the rule so that no caller can re-implement it. It is a
kernel leaf — the standard library only — and the static guard
(``tests/integration/test_executable_resolution_guard.py``) fails the build
if ``shutil.which`` reappears anywhere in the package.
"""
from __future__ import annotations

import os

#: Whether this platform resolves a program through ``PATHEXT``. It is a
#: module constant rather than an inline ``os.name`` check so the branch can
#: be exercised on either platform by the tests: the coverage gate runs on
#: both Linux and Windows, and a branch only one of them can reach would sit
#: at 99 % forever on the other.
IS_WINDOWS = os.name == "nt"


def executable_candidates(candidate: str) -> list[str]:
    """The spellings a bare program name may have on this platform.

    Windows resolves a program through ``PATHEXT`` (``.COM;.EXE;.BAT;…``);
    POSIX has exactly one spelling. The extension is part of the *program*
    the allow-list names, so it is resolved here rather than left to the
    operating system.
    """
    if not IS_WINDOWS:
        return [candidate]
    declared = os.environ.get("PATHEXT", "")
    extensions = [
        item if item.startswith(".") else f".{item}"
        for item in declared.split(os.pathsep) if item
    ]
    if not extensions:
        extensions = [".com", ".exe", ".bat", ".cmd"]
    return [candidate] + [candidate + extension.lower() for extension in extensions]


def resolve_executable(name: str, path: str) -> str | None:
    """Resolve an allow-listed bare name to exactly one absolute PATH hit.

    The scan walks the ``PATH`` entries itself, in order, and **never looks
    at the current directory**. The result is absolute, so the operating
    system cannot re-order the search when the child starts: the program
    that runs is the one this function validated.

    A name that carries a path (``x/git``), an absolute name, or an empty
    name is refused outright — the allow-list constrains *which* program
    runs, and only the deployment's own ``PATH`` may satisfy it.
    """
    if not name or os.path.isabs(name) or os.path.dirname(name):
        return None
    for entry in (path or "").split(os.pathsep):
        if not entry:
            continue
        directory = entry.strip('"')
        for candidate in executable_candidates(os.path.join(directory, name)):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return os.path.abspath(candidate)
    return None


def resolve_tool_binary(name: str) -> str | None:
    """Resolve a program against the PATH this process inherits.

    The convenience form for the callers that do not carry their own PATH.
    """
    return resolve_executable(name, os.environ.get("PATH", ""))


__all__ = [
    "IS_WINDOWS",
    "executable_candidates",
    "resolve_executable",
    "resolve_tool_binary",
]
