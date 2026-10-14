"""Lifecycle coordination for long-running Agent tasks.

Transport-agnostic coordinator supporting cooperative cancellation and status tracking.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskRecord:
    session_id: str
    request_id: str
    status: str = "running"
    cancelled: bool = False
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    #: When the cancel request landed — the start of the recovery window
    #: (SCORECARD #4) measured by :meth:`TaskCoordinator.finish`. Internal
    #: instrumentation state, deliberately not part of the transport snapshot.
    cancelled_at: float | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "request_id": self.request_id,
            "status": self.status,
            "cancelled": self.cancelled,
            "updated_at": self.updated_at,
            **self.metadata,
        }


class TaskCoordinator:
    """Thread-safe in-process task registry with cooperative cancellation.

    Records are bounded: terminal records are evicted oldest-first once
    ``max_records`` is exceeded, so long-lived sessions cannot leak memory
    (live runs are never evicted).

    ``recovery_observer`` receives the seconds between the cancel request and
    the runner stopping (via :meth:`finish`). The coordinator stays free of
    metrics/observability imports (import contract), so the composition root
    injects the Prometheus sink; with none injected this stays a no-op.
    """

    _TERMINAL = {"succeeded", "failed", "cancelled"}

    def __init__(
        self,
        max_records: int = 1000,
        recovery_observer: Callable[[float], None] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, TaskRecord] = {}
        self._max_records = max(16, int(max_records))
        self._recovery_observer = recovery_observer

    def _evict_locked(self) -> None:
        if len(self._records) <= self._max_records:
            return
        terminal = sorted(
            (key for key, record in self._records.items() if record.status in self._TERMINAL),
            key=lambda key: self._records[key].updated_at,
        )
        for key in terminal:
            del self._records[key]
            if len(self._records) <= self._max_records:
                return

    @staticmethod
    def key(session_id: str, request_id: str = "") -> str:
        return f"{session_id}:{request_id or 'active'}"

    def start(self, session_id: str, request_id: str = "", **metadata: Any) -> TaskRecord:
        key = self.key(session_id, request_id)
        with self._lock:
            previous = self._records.get(key)
            record = TaskRecord(session_id=session_id, request_id=request_id, metadata=metadata)
            # A cancel that landed before the run started must survive ``start``:
            # without this, ``POST /tasks/{id}:cancel`` racing the worker thread
            # was lost and the LangGraph loop ran to completion anyway.
            if previous is not None and previous.cancelled and previous.metadata.get("pending_cancel"):
                record.cancelled = True
                record.status = "cancelled"
                record.cancelled_at = previous.cancelled_at
            self._records[key] = record
            self._evict_locked()
        return record

    def update(
        self,
        session_id: str,
        request_id: str = "",
        status: str | None = None,
        **metadata: Any,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(self.key(session_id, request_id))
            if record is None:
                record = self.start(session_id, request_id)
            if status and not (record.status in self._TERMINAL and status == "running"):
                record.status = status
            record.metadata.update(metadata)
            record.updated_at = time.time()
            self._evict_locked()
            return record.snapshot()

    def cancel(self, session_id: str, request_id: str = "",
               create_if_missing: bool = False) -> dict[str, Any] | None:
        """Mark the session's run cancelled.

        ``create_if_missing`` is for callers that already know a run exists
        (the durable task is non-terminal in the store): a cancel that lands
        before the worker registers its record parks a pending-cancel entry
        instead of vanishing. Transport callers without that knowledge keep the
        historical "unknown session → ``None``" contract.
        """
        with self._lock:
            record = self._records.get(self.key(session_id, request_id))
            if record is None:
                candidates = [
                    r for r in self._records.values()
                    if r.session_id == session_id and r.status not in self._TERMINAL
                ]
                record = max(candidates, key=lambda item: item.updated_at, default=None)
            if record is None:
                if not create_if_missing:
                    return None
                # No run registered yet (cancel raced the worker): park a
                # pending-cancel record so ``start`` cannot wipe the intent.
                record = TaskRecord(
                    session_id=session_id, request_id=request_id,
                    status="cancelled", cancelled=True,
                    metadata={"pending_cancel": True},
                    cancelled_at=time.time(),
                )
                self._records[self.key(session_id, request_id)] = record
                return record.snapshot()
            if record.status in self._TERMINAL and not record.cancelled:
                return None
            record.cancelled = True
            record.status = "cancelled"
            record.cancelled_at = time.time()
            record.updated_at = time.time()
            return record.snapshot()

    def is_cancelled(self, session_id: str, request_id: str = "") -> bool:
        with self._lock:
            record = self._records.get(self.key(session_id, request_id))
            if record is not None:
                return record.cancelled
            return any(
                r.session_id == session_id and r.cancelled and r.status == "cancelled"
                for r in self._records.values()
            )

    def finish(
        self,
        session_id: str,
        request_id: str = "",
        success: bool = True,
        **metadata: Any,
    ) -> dict[str, Any]:
        snapshot = self.update(
            session_id,
            request_id,
            status="cancelled" if self.is_cancelled(session_id, request_id) else "succeeded" if success else "failed",
            **metadata,
        )
        self._observe_recovery(session_id, request_id)
        return snapshot

    def _observe_recovery(self, session_id: str, request_id: str = "") -> None:
        """Report cancel-to-stop latency (SCORECARD #4) through the observer.

        Called from :meth:`finish` — the single point where the runner has
        actually stopped. A run that was never cancelled, or one whose record
        has already been evicted, reports nothing.
        """
        if self._recovery_observer is None:
            return
        with self._lock:
            record = self._records.get(self.key(session_id, request_id))
            cancelled_at = record.cancelled_at if record is not None else None
        if cancelled_at is None:
            return
        self._recovery_observer(max(0.0, time.time() - cancelled_at))


__all__ = [
    "TaskCoordinator",
    "TaskRecord",
]
