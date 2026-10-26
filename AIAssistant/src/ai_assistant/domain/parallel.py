"""Bounded fan-out policy for one agent step (ADR 0044).

A workflow used to run its plan steps strictly one at a time. The tool node
now fans out a *group* of independent read-only steps, and that behaviour
needs one owner for its limits: the width a deployment gets by default, the
hard ceiling it may not exceed, and the clamp that maps a raw environment
value onto both. Keeping the numbers here (stdlib-only, like
``domain/context_budget``) means the settings parser, the dependency bundle and
the tool node cannot disagree about what "parallel" is allowed to mean.
"""
from __future__ import annotations

#: Width used when ``AGENT_PARALLEL_MAX`` is unset.
PARALLEL_MAX_DEFAULT = 4
#: Ceiling an operator cannot raise: each branch is an OS thread in tool I/O.
PARALLEL_MAX_HARD_CAP = 8


def clamp_parallel_max(raw: int) -> int:
    """Clamp *raw* into ``[1, PARALLEL_MAX_HARD_CAP]``.

    Zero and negatives mean "no fan-out" in the setting's spirit, but a batch
    with no width is not a batch: one is the floor. The ceiling is a resource
    bound, not a preference — a deployment cannot opt into more branches than
    a desktop host is expected to carry.
    """
    return max(1, min(PARALLEL_MAX_HARD_CAP, raw))


__all__ = ["PARALLEL_MAX_DEFAULT", "PARALLEL_MAX_HARD_CAP", "clamp_parallel_max"]
