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

    The comparison the tool gateway needs: a principal's clearance must be at
    least the tool's maximum classification (its data ceiling). The comparison
    used to run the other way round (``tool >= principal``), which denied the
    most trusted callers and let the least trusted ones through.
    """
    return CLASSIFICATION_ORDER[value] >= CLASSIFICATION_ORDER[minimum]


#: The scope lattice of the least-privilege model. ADR 0032 moved it here
#: for the same reason as the classification order: the scopes are a fact
#: about the domain's principals, not about the transport that resolves them.
#: A capability that is not listed — or a host that supplies no map at all —
#: falls back to the read-only set. There is deliberately no "trusted default".
READ_ONLY_SCOPES = frozenset({"project.read"})
DESKTOP_ACTION_SCOPE = "desktop.action"


def scopes_for_capability(
    capability: str,
    capability_scopes: Mapping[str, frozenset[str]] | None = None,
) -> frozenset[str]:
    """The scopes a capability may act under.

    ``capability_scopes`` is the deployment's capability→scope map; an
    unknown capability, or a host that forgot to inject a map, gets the
    read-only set. A missing map must fail closed: the old fallback granted
    ``project.write``/``project.execute`` to every embedded caller, so an
    unconfigured host silently ran with the desktop's trusted scopes.
    """
    if capability_scopes is None:
        return READ_ONLY_SCOPES
    return capability_scopes.get(capability, READ_ONLY_SCOPES)


#: The scopes that let a principal change the world rather than only read it.
#: A principal that holds one of them may drive a tool that touches
#: restricted data; a read-only principal may not.
_ELEVATED_SCOPES = frozenset({"project.write", "project.execute", "desktop.action"})


def clearance_for_scopes(scopes: frozenset[str]) -> DataClassification:
    """The data clearance a principal's granted scopes imply.

    A principal's clearance is a fact about **what the deployment granted**,
    never about what the caller declared about its own data: a remote caller
    that labels its payload ``regulated`` must not thereby buy the right to
    drive a tool that rewrites the project. The scope map is the deployment's
    own statement of trust, so the clearance follows it.

    * read-only scopes → ``INTERNAL`` (the project corpus);
    * an elevated scope (write / execute / desktop action) → ``REGULATED``.
    """
    if scopes & _ELEVATED_SCOPES:
        return DataClassification.REGULATED
    return DataClassification.INTERNAL


class ExecutionOrigin(StrEnum):
    LOCAL = "local"
    REMOTE_A2A = "remote_a2a"


@dataclass(frozen=True, slots=True)
class Principal:
    subject: str
    scopes: frozenset[str] = frozenset()
    #: The principal's **clearance**: the highest data classification it may
    #: work with. It is derived from the granted scopes
    #: (:func:`clearance_for_scopes`), never from the caller's own claim
    #: about its payload — a task that declares itself ``regulated`` does not
    #: thereby become trusted to drive a restricted tool.
    classification: DataClassification = DataClassification.INTERNAL

    def permits(self, required_scope: str | None) -> bool:
        return required_scope is None or required_scope in self.scopes


def principal_for(subject: str, scopes: frozenset[str] | set[str]) -> Principal:
    """A principal whose clearance follows the scopes it was granted."""
    granted = frozenset(scopes)
    return Principal(subject, granted, clearance_for_scopes(granted))


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

