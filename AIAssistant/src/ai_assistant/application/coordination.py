"""Lifecycle coordination for long-running Agent tasks.

Transport-agnostic coordinator supporting cooperative cancellation and status tracking.
"""
from __future__ import annotations

import threading
import time
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
    """Thread-safe in-process task registry with cooperative cancellation."""

    _TERMINAL = {"succeeded", "failed", "cancelled"}

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, TaskRecord] = {}

    @staticmethod
    def key(session_id: str, request_id: str = "") -> str:
        return f"{session_id}:{request_id or 'active'}"

    def start(self, session_id: str, request_id: str = "", **metadata: Any) -> TaskRecord:
        record = TaskRecord(session_id=session_id, request_id=request_id, metadata=metadata)
        with self._lock:
            self._records[self.key(session_id, request_id)] = record
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
            return record.snapshot()

    def cancel(self, session_id: str, request_id: str = "") -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(self.key(session_id, request_id))
            if record is None:
                candidates = [
                    r for r in self._records.values()
                    if r.session_id == session_id and r.status not in self._TERMINAL
                ]
                record = max(candidates, key=lambda item: item.updated_at, default=None)
            if record is None:
                return None
            record.cancelled = True
            record.status = "cancelled"
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
        return self.update(
            session_id,
            request_id,
            status="cancelled" if self.is_cancelled(session_id, request_id) else "succeeded" if success else "failed",
            **metadata,
        )


coordinator = TaskCoordinator()

__all__ = [
    "TaskCoordinator",
    "TaskRecord",
    "coordinator",
]
