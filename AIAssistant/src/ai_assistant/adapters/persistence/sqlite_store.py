"""SQLite-backed implementation of the TaskStore port."""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from ai_assistant.domain.security import DataClassification, ExecutionOrigin
from ai_assistant.domain.tasks import AgentTask, TaskStatus


class SqliteTaskStore:
    """Durable task store backed by a local SQLite database.

    Thread-safety: each call acquires the module-level ``check_same_thread=False``
    connection through a re-entrant lock so multiple threads can share one store.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _migrate(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id              TEXT PRIMARY KEY,
                    capability      TEXT NOT NULL,
                    message         TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    result          TEXT,
                    error           TEXT,
                    metadata        TEXT NOT NULL DEFAULT '{}',
                    context_id      TEXT,
                    classification  TEXT NOT NULL DEFAULT 'internal',
                    origin          TEXT NOT NULL DEFAULT 'local',
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_events (
                    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id     TEXT NOT NULL REFERENCES tasks(id),
                    kind        TEXT NOT NULL,
                    data        TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_task_context ON tasks(context_id);
                CREATE INDEX IF NOT EXISTS idx_task_created ON tasks(created_at);
                CREATE INDEX IF NOT EXISTS idx_event_task ON task_events(task_id);
            """)
            # A database written before these columns existed would otherwise
            # keep the old shape forever (``CREATE TABLE IF NOT EXISTS`` never
            # alters), and every reload would hand the domain a task whose
            # classification and origin had been silently rewritten.
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(tasks)")}
            if "classification" not in columns:
                self._conn.execute(
                    "ALTER TABLE tasks ADD COLUMN classification TEXT NOT NULL DEFAULT 'internal'",
                )
            if "origin" not in columns:
                self._conn.execute("ALTER TABLE tasks ADD COLUMN origin TEXT NOT NULL DEFAULT 'local'")

    # ── TaskStore protocol ─────────────────────────────────────────────────────

    def create(self, task: AgentTask) -> None:
        now = _now()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO tasks (id, capability, message, status, result, error, metadata, context_id, "
                "classification, origin, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (task.id, task.capability, task.message, str(task.status),
                 json.dumps(task.result, default=str) if task.result else None,
                 task.error,
                 json.dumps(dict(task.metadata), default=str),
                 task.context_id,
                 task.classification.value, task.origin.value, now, now),
            )

    def get(self, task_id: str) -> AgentTask | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, capability, message, status, result, error, metadata, context_id, "
                "classification, origin, created_at, updated_at "
                "FROM tasks WHERE id = ?", (task_id,)
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_task(row)

    def save(self, task: AgentTask) -> None:
        now = _now()
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE tasks SET status=?, result=?, error=?, metadata=?, updated_at=? WHERE id=?",
                (str(task.status),
                 json.dumps(task.result, default=str) if task.result else None,
                 task.error,
                 json.dumps(dict(task.metadata), default=str),
                 now, task.id),
            )

    def save_if_status(self, task: AgentTask, expected: TaskStatus) -> bool:
        """Compare-and-set update: one statement, checked rowcount (no TOCTOU)."""
        now = _now()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE tasks SET status=?, result=?, error=?, metadata=?, updated_at=? "
                "WHERE id=? AND status=?",
                (str(task.status),
                 json.dumps(task.result, default=str) if task.result else None,
                 task.error,
                 json.dumps(dict(task.metadata), default=str),
                 now, task.id, str(expected)),
            )
        return cursor.rowcount == 1

    def append_event(self, task_id: str, kind: str, data: dict) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO task_events (task_id, kind, data, created_at) VALUES (?, ?, ?, ?)",
                (task_id, kind, json.dumps(data, default=str), _now()),
            )

    def events(self, task_id: str, *, since: int = 0) -> Iterable[dict]:
        """Stream only the events after the ``rowid`` cursor (O(new), not O(n)).

        The ``seq`` key is the cursor itself, so SSE subscribers can pass it
        back instead of re-reading and re-slicing the whole log every poll.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT rowid, kind, data, created_at FROM task_events "
                "WHERE task_id = ? AND rowid > ? ORDER BY rowid",
                (task_id, int(since)),
            )
            rows = cursor.fetchall()
        for rowid, kind, data_json, created_at in rows:
            yield {"seq": rowid, "kind": kind, "data": json.loads(data_json), "created_at": created_at}

    def list(self, *, context_id: str | None = None, limit: int = 100) -> Iterable[AgentTask]:
        with self._lock:
            if context_id is not None:
                cursor = self._conn.execute(
                    "SELECT id, capability, message, status, result, error, metadata, context_id, "
                    "classification, origin, created_at, updated_at "
                    "FROM tasks WHERE context_id = ? ORDER BY created_at DESC LIMIT ?",
                    (context_id, limit),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT id, capability, message, status, result, error, metadata, context_id, "
                    "classification, origin, created_at, updated_at "
                    "FROM tasks ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
        for row in rows:
            yield _row_to_task(row)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _epoch(stored: str) -> float:
    """Parse a stored ISO-8601 instant back into the domain's epoch seconds.

    Without this the row mapper rebuilt both timestamps from ``time()``, so a
    restart re-dated every task to the moment it was read — and the A2A payload
    answers with ``updated_at``, so clients would see the original creation
    and status times replaced by the restart.
    """
    return datetime.fromisoformat(stored).timestamp()


def _row_to_task(row: tuple) -> AgentTask:
    (tid, capability, message, status_str, result_json, error, metadata_json, context_id,
     classification, origin, created_at, updated_at) = row
    return AgentTask(
        id=tid,
        capability=capability,
        message=message,
        context_id=context_id,
        metadata=json.loads(metadata_json) if metadata_json else {},
        # Parsed, not defaulted: the classification decides whether this task may
        # leave the trust boundary, and an unparseable value must fail loudly
        # rather than come back promoted to the permissive default.
        classification=DataClassification(classification),
        status=TaskStatus(status_str),
        origin=ExecutionOrigin(origin),
        created_at=_epoch(created_at),
        updated_at=_epoch(updated_at),
        result=json.loads(result_json) if result_json else None,
        error=error,
    )
