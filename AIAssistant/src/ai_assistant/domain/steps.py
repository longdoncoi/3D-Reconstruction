"""The observable step-log vocabulary (ADR 0019).

``Step`` used to live in ``orchestration.state``, which forced every layer
outside orchestration to spell the same shape as ``dict[str, Any]``: the
application contract may not import orchestration, so ``AgentRunResult`` and
the persistence/HTTP boundaries kept a blanket ``Any`` even after the
in-process write side became typed (finding 19). What a step *is* — the
discriminator plus the payload keys the UI and the approval log may observe —
is domain language, so the vocabulary lives here: stdlib-only, framework-free,
and importable inward by every layer. ``orchestration.state`` re-exports it so
existing call sites keep one import path.

The type layer only checks what this process constructs. Data persisted by a
previous run crosses the trust boundary as plain JSON, so
:func:`validate_steps` is the runtime twin of the ``Step`` TypedDict: it
re-derives the discriminator check from the ``StepType`` literal itself, so a
vocabulary change can never drift between the compile-time and the runtime
gate.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypedDict, cast, get_args

#: Closed set of step discriminators. Read sites must compare through
#: :func:`has_step_type` so a typo (``"tool_cal"``) becomes a mypy error
#: instead of a routing condition that silently never matches.
StepType = Literal[
    "approval_granted",
    "cancelled",
    "coding_incomplete",
    "context_compacted",
    "delegation",
    "error",
    "final_answer",
    "memory_recall",
    "pending_approval",
    "plan",
    "plan_incomplete",
    "plan_reflection",
    "reflection",
    "step_answer",
    "thinking",
    "tool_call",
    "tool_result",
    "validation_error",
    "verification",
]


class _StepTyped(TypedDict):
    #: Discriminator every entry carries; routers and guards read it.
    type: StepType


class Step(_StepTyped, total=False):
    """One entry of the observable step log (finding 19, write side).

    ``type`` is required and closed; payload keys are optional because most
    steps carry only a subset of them. Typing ``AgentState.steps`` as
    ``list[Step]`` makes mypy check every construction (unknown ``type`` or
    unknown payload key = error) and every read (typed instead of ``Any``).
    TypedDicts are dicts at runtime, so this is a compile-time-only change.
    """

    content: str
    tool: str
    params: dict[str, Any]
    result: Any
    iteration: int
    reason: str
    error: str
    plan_step_index: int
    plan_attempt: int
    remaining: list[str]
    missing: list[str]
    steps: list[str]
    spec: dict[str, Any] | None
    tool_hints: list[dict[str, Any]]
    agent: str
    scope_id: str
    action_id: str
    request_id: str
    idempotency_key: str
    remote_endpoint: str | None
    #: Payload of a ``memory_recall`` step: how many long-term records were
    #: injected into the system prompt of this run (ADR 0043).
    count: int
    #: Parallel-batch labels (ADR 0044): the group a ``tool_call``/
    #: ``tool_result`` pair belongs to and the branch that produced it. Both
    #: are additive — a sequential call keeps them absent, so the client and
    #: the checkpointer see the same shape they always did.
    batch_id: str
    branch_index: int


def has_step_type(step: Mapping[str, Any], expected: StepType) -> bool:
    """True when *step* carries the discriminator *expected*."""
    return step.get("type") == expected


def validate_steps(raw: object) -> list[Step]:
    """Runtime gate for a step list crossing a trust boundary (ADR 0019).

    Rehydration (``PendingActionStore.load``) calls this before persisted
    entries may reach resume, because the compiler cannot see data written by
    a previous process. The contract:

    * *raw* must be a list — anything else raises :class:`ValueError`;
    * every entry must be a dict whose ``type`` is in :data:`StepType`,
      re-checked against ``get_args(StepType)`` so the vocabulary has exactly
      one owner;
    * payload keys beyond the vocabulary are **kept**: the step log is display
      state, and dropping keys a newer writer produced would corrupt it after
      a downgrade. The discriminator, by contrast, is fail-closed — an unknown
      ``type`` would reach a client switch that has no arm for it.

    Callers drop the *containing entry* on :class:`ValueError` and log it;
    the validator itself never logs and never swallows.
    """
    if not isinstance(raw, list):
        # ValueError (not TypeError): the rehydration caller skips bad payloads
        # by catching ValueError alone (a2a_client convention).
        msg = f"steps must be a list, got {type(raw).__name__}"
        raise ValueError(msg)  # noqa: TRY004
    validated: list[Step] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            msg = f"step[{index}] must be a dict, got {type(entry).__name__}"
            raise ValueError(msg)  # noqa: TRY004
        kind = entry.get("type")
        if kind not in get_args(StepType):
            msg = f"step[{index}] has type {kind!r} outside the step vocabulary"
            raise ValueError(msg)
        # The two checks above are this cast's proof: mypy cannot narrow
        # ``dict[Any, Any]`` to a TypedDict, which is exactly the gap the
        # runtime gate exists to close.
        validated.append(cast(Step, entry))
    return validated
