from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum


class DataClassification(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    REGULATED = "regulated"


class ExecutionOrigin(StrEnum):
    LOCAL = "local"
    REMOTE_A2A = "remote_a2a"


@dataclass(frozen=True, slots=True)
class Principal:
    subject: str
    scopes: frozenset[str] = frozenset()
    classification: DataClassification = DataClassification.INTERNAL

    def permits(self, required_scope: str | None) -> bool:
        return required_scope is None or required_scope in self.scopes


# The active principal for the current execution context. Transports that execute
# on behalf of a caller with reduced privileges (for example an A2A capability)
# bind their principal here so the shared tool gateway enforces those scopes even
# when the orchestration engine was written for the local desktop caller. The
# default is ``None``: callers fall back to their own configured identity.
_current_principal: ContextVar["Principal | None"] = ContextVar("ai_assistant_principal", default=None)


def current_principal() -> "Principal | None":
    """Return the principal bound to the current execution context, if any."""
    return _current_principal.get()


@contextmanager
def bind_principal(principal: "Principal") -> "Iterator[Principal]":
    """Bind ``principal`` as the active identity for the enclosed block."""
    token = _current_principal.set(principal)
    try:
        yield principal
    finally:
        _current_principal.reset(token)

