"""Single-use approval grants backing the tool gateway (ADR 0009).

A grant is issued only by a path that represents a user decision
(``ApprovalService.approve`` or an explicit ``resume.approved`` continuation)
and is consumed by the gateway on the first execution it unlocks.  It is bound
to the fingerprint of the approved invocation, so one approval can never open a
*different* tool or different parameters, and it expires on its own.
"""
from __future__ import annotations

import threading
from time import monotonic

#: Grants live as long as a HITL round trip reasonably can. A resumed run that
#: stalls for ten minutes falls back to asking the user again (fail closed).
DEFAULT_GRANT_TTL_SECONDS = 600.0


class InMemoryApprovalGrantStore:
    """Thread-safe, process-local store of single-use approval grants."""

    def __init__(self, ttl_seconds: float = DEFAULT_GRANT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._grants: dict[str, tuple[str, float, int]] = {}
        self._lock = threading.Lock()

    def issue(self, token: str, fingerprint: str) -> None:
        """Create (or refresh) a grant for one invocation fingerprint."""
        if not token:
            return
        with self._lock:
            self._grants[token] = (fingerprint, monotonic() + self._ttl, 1)

    def covers(self, token: str, fingerprint: str) -> bool:
        """True when ``token`` unlocks ``fingerprint`` — and consume it.

        Consumption makes the grant single use: replaying an approved
        invocation, or reusing the token for a different tool, is refused and
        the caller has to go back through the approval flow.
        """
        if not token:
            return False
        with self._lock:
            entry = self._grants.get(token)
            if entry is None:
                return False
            expected, expires_at, remaining = entry
            if monotonic() > expires_at or remaining <= 0:
                self._grants.pop(token, None)
                return False
            if expected != fingerprint:
                return False
            if remaining == 1:
                self._grants.pop(token, None)
            else:
                self._grants[token] = (expected, expires_at, remaining - 1)
            return True

    def peek(self, token: str, fingerprint: str) -> bool:
        """Non-consuming liveness probe used by the reason node to decide
        whether it still has to ask the user (``covers`` would spend the
        single use just by being asked)."""
        if not token:
            return False
        with self._lock:
            entry = self._grants.get(token)
            if entry is None:
                return False
            expected, expires_at, remaining = entry
            if monotonic() > expires_at or remaining <= 0:
                self._grants.pop(token, None)
                return False
            return expected == fingerprint

    def revoke(self, token: str) -> None:
        """Drop every remaining use of a grant (run finished / task failed)."""
        with self._lock:
            self._grants.pop(token, None)

    def revoke_all(self) -> None:
        with self._lock:
            self._grants.clear()


__all__ = ["DEFAULT_GRANT_TTL_SECONDS", "InMemoryApprovalGrantStore"]
