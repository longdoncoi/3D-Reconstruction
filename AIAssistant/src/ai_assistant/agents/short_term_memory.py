"""Bounded in-process short-term memory buffer (ADR 0043).

The client ships the history it still holds on every request; between requests
this buffer remembers what the server last saw, so a run does not lose its
context when the client truncates, restarts its transcript or restarts the
process-side view of the session. It is deliberately:

* **bounded twice** — ``max_messages`` caps one session's window (the oldest
  turn falls off the front) and ``max_sessions`` evicts the least recently
  touched session (``TaskCoordinator._evict_locked`` pattern), so a long-lived
  server cannot leak;
* **in-process** — the contract is ``ShortTermMemoryPort``, so a durable
  implementation can replace it without the agent layer noticing;
* **locked** — sessions execute on worker threads and the buffer is read by
  one and written by another for the same session id.

Turns are what :func:`ai_assistant.domain.memory.conversation_turns` accepts:
the system prompt is rebuilt per run and tool observations are re-derivable.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..ports import ShortTermMemoryPort


@dataclass
class _Session:
    messages: list[dict[str, str]] = field(default_factory=list)
    updated_at: float = 0.0


class BoundedShortTermMemory:
    """A ``ShortTermMemoryPort`` holding a bounded window per session id."""

    #: Sessions are cheap but not free: the map itself must not grow with the
    #: number of session ids a chatty client can invent.
    DEFAULT_MAX_SESSIONS = 256

    def __init__(self, max_messages: int = 40,
                 max_sessions: int = DEFAULT_MAX_SESSIONS) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, _Session] = {}
        self._max_messages = max(1, int(max_messages))
        self._max_sessions = max(1, int(max_sessions))

    def _evict_locked(self) -> None:
        """Drop the least recently touched sessions until the map fits."""
        while len(self._sessions) > self._max_sessions:
            oldest = min(self._sessions, key=lambda key: self._sessions[key].updated_at)
            del self._sessions[oldest]

    def record(self, session_id: str, messages: Sequence[Mapping[str, str]]) -> None:
        """Store the window for *session_id*, replacing what was there.

        The messages of a run already contain the merged history, so
        replacement (not append) is what keeps the buffer self-bounding;
        only the newest ``max_messages`` turns survive.
        """
        turns = [
            {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
            for m in messages
        ]
        if not turns:
            # Recording nothing must not erase the context the buffer exists
            # to keep; the caller had no messages to report.
            return
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = _Session()
                self._sessions[session_id] = session
            session.messages = turns[-self._max_messages:]
            session.updated_at = time.time()
            self._evict_locked()

    def recent(self, session_id: str) -> list[dict[str, str]]:
        """A copy of the remembered window (``[]`` for an unknown session)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            return [dict(m) for m in session.messages]

    def forget(self, session_id: str) -> None:
        """Drop a session's window (explicit reset; unknown ids are a no-op)."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup(self, cutoff: float) -> int:
        """Drop sessions untouched since ``cutoff``; returns how many went."""
        with self._lock:
            expired = [
                key for key, session in self._sessions.items()
                if session.updated_at < cutoff
            ]
            for key in expired:
                del self._sessions[key]
        return len(expired)


__all__ = ["BoundedShortTermMemory"]

#: Structural conformance, checked by mypy in the strict gate: the class must
#: satisfy ``ShortTermMemoryPort`` — it is injected through that contract only.
_ShortTermMemoryImplementation: type[ShortTermMemoryPort] = BoundedShortTermMemory
