from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum


class DataClassification(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    REGULATED = "regulated"


#: The classification lattice. ADR 0032 moved it here: the enum is the
#: domain's, so the question "is PUBLIC ⊆ RESTRICTED?" is a question about
#: the domain's own model — the policy engine used to sit one layer *above*
#: the model it constrains, and could not answer it without reaching in.
CLASSIFICATION_ORDER: Mapping[DataClassification, int] = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.RESTRICTED: 2,
    DataClassification.REGULATED: 3,
}


def classification_at_least(
    value: DataClassification, minimum: DataClassification,
) -> bool:
    """Whether ``value`` is at least as protective as ``minimum``.

    The comparison the tool gateway needs: a principal may only use a tool
    whose maximum classification is at least the principal's own.
    """
    return CLASSIFICATION_ORDER[value] >= CLASSIFICATION_ORDER[minimum]


#: The scope lattice of the least-privilege model. ADR 0032 moved it here
#: for the same reason as the classification order: the scopes are a fact
#: about the domain's principals, not about the transport that resolves them.
#: A capability that is not listed falls back to the read-only set.
DEFAULT_TRUSTED_SCOPES = frozenset({"project.read", "project.write", "project.execute"})
READ_ONLY_SCOPES = frozenset({"project.read"})
DESKTOP_ACTION_SCOPE = "desktop.action"


def scopes_for_capability(
    capability: str,
    capability_scopes: Mapping[str, frozenset[str]] | None = None,
) -> frozenset[str]:
    """The scopes a capability may act under.

    ``capability_scopes`` is the deployment's capability→scope map; an
    unknown capability gets the read-only set, so a capability nobody
    described cannot act.
    """
    if capability_scopes is None:
        return DEFAULT_TRUSTED_SCOPES
    return capability_scopes.get(capability, READ_ONLY_SCOPES)


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

