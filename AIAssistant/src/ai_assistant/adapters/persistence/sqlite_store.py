"""SQLite-backed implementation of the TaskStore port."""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ai_assistant.domain.security import DataClassification, ExecutionOrigin
from ai_assistant.domain.tasks import AgentTask, TaskStatus

from .storage_crypto import FieldCipher, StorageKeyError, is_sealed, reseal_text


class SqliteTaskStore:
    """Durable task store backed by a local SQLite database.

    Thread-safety: each call acquires the module-level ``check_same_thread=False``
    connection through a re-entrant lock so multiple threads can share one store.
    When a :class:`FieldCipher` is injected, every message/metadata/result/error
    column and every event payload is sealed before it reaches the file.
    """

    def __init__(self, db_path: str | Path, *, cipher: FieldCipher | None = None) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._cipher = cipher
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
            # classification and origin had been silently rewritten. The
            # schema version makes the migration run exactly once instead of
            # probing column names on every startup.
            version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if version < 1:
                columns = {row[1] for row in self._conn.execute("PRAGMA table_info(tasks)")}
                if "classification" not in columns:
                    self._conn.execute(
                        "ALTER TABLE tasks ADD COLUMN classification TEXT NOT NULL DEFAULT 'internal'",
                    )
                if "origin" not in columns:
                    self._conn.execute("ALTER TABLE tasks ADD COLUMN origin TEXT NOT NULL DEFAULT 'local'")
                self._conn.execute("PRAGMA user_version = 1")
            if version < 2:
                columns = {row[1] for row in self._conn.execute("PRAGMA table_info(tasks)")}
                if "owner" not in columns:
                    self._conn.execute(
                        "ALTER TABLE tasks ADD COLUMN owner TEXT NOT NULL DEFAULT 'default'",
                    )
                self._conn.execute("PRAGMA user_version = 2")
            # Version 3 is written only when a key was actually supplied. A
            # store opened without the key keeps the old version so enabling
            # encryption later still triggers the one-time plaintext migration.
            cipher = self._cipher
            if cipher is not None and version < 3:
                self._seal_legacy_rows(cipher)
                self._conn.execute("PRAGMA user_version = 3")

    def _seal_legacy_rows(self, cipher: FieldCipher) -> None:
        """Seal rows written before encryption was enabled (idempotent)."""
        task_rows = self._conn.execute(
            "SELECT id, message, result, error, metadata FROM tasks",
        ).fetchall()
        task_updates = []
        for task_id, message, result, error, metadata in task_rows:
            sealed = (
                _seal_legacy(cipher, message, "task.message"),
                _seal_legacy(cipher, result, "task.result"),
                _seal_legacy(cipher, error, "task.error"),
                _seal_legacy(cipher, metadata, "task.metadata"),
            )
            if sealed != (message, result, error, metadata):
                task_updates.append((*sealed, task_id))
        if task_updates:
            self._conn.executemany(
                "UPDATE tasks SET message=?, result=?, error=?, metadata=? WHERE id=?",
                task_updates,
            )

        event_rows = self._conn.execute("SELECT rowid, data FROM task_events").fetchall()
        event_updates = [
            (_seal_legacy(cipher, data, "event.data"), rowid)
            for rowid, data in event_rows
            if not cipher.is_sealed(data)
        ]
        if event_updates:
            self._conn.executemany(
                "UPDATE task_events SET data=? WHERE rowid=?",
                event_updates,
            )

    # ── TaskStore protocol ─────────────────────────────────────────────────────

    def commit(
        self, task: AgentTask, *,
        expected: TaskStatus | None = None,
        events: Sequence[tuple[str, dict[str, Any]]] = (),
    ) -> bool:
        """Persist the task and its events in one transaction (ADR 0032).

        The state write and the event append are the same fact observed
        twice. Two separate transactions break both halves of that promise: a
        crash between them leaves a task that is ``completed`` with no
        terminal event, and a subscriber waits forever for an event that will
        never arrive. The adapter already knew the pattern — ``fail_orphaned``
        below does it in one transaction — so the port now *asks* for it
        instead of letting the application layer write two statements and
        hope.

        ``expected is None`` inserts; a status makes the write a
        compare-and-set and the return value says whether it landed.
        """
        now = _now()
        result_json, error, metadata_json = _content_columns(task, self._cipher)
        sealed_events = [
            (
                task.id,
                kind,
                _seal_text(self._cipher, json.dumps(data, default=str), "event.data"),
                now,
            )
            for kind, data in events
        ]
        with self._lock, self._conn:
            if expected is None:
                self._conn.execute(
                    "INSERT INTO tasks (id, capability, message, status, result, error, metadata, context_id, "
                    "classification, origin, owner, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (task.id, task.capability,
                     _seal_text(self._cipher, task.message, "task.message"), str(task.status),
                     result_json, error, metadata_json,
                     task.context_id,
                     task.classification.value, task.origin.value, task.owner, now, now),
                )
            else:
                cursor = self._conn.execute(
                    "UPDATE tasks SET status=?, result=?, error=?, metadata=?, updated_at=? "
                    "WHERE id=? AND status=?",
                    (str(task.status), result_json, error, metadata_json, now, task.id, str(expected)),
                )
                if cursor.rowcount != 1:
                    # Lost the race: nothing was written, and no event may be
                    # written either — the event log must not announce a
                    # transition that did not happen.
                    return False
            if sealed_events:
                self._conn.executemany(
                    "INSERT INTO task_events (task_id, kind, data, created_at) VALUES (?, ?, ?, ?)",
                    sealed_events,
                )
        return True

    def create(self, task: AgentTask) -> None:
        now = _now()
        result_json, error, metadata_json = _content_columns(task, self._cipher)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO tasks (id, capability, message, status, result, error, metadata, context_id, "
                "classification, origin, owner, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (task.id, task.capability,
                 _seal_text(self._cipher, task.message, "task.message"), str(task.status),
                 result_json, error, metadata_json,
                 task.context_id,
                 task.classification.value, task.origin.value, task.owner, now, now),
            )

    def get(self, task_id: str) -> AgentTask | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, capability, message, status, result, error, metadata, context_id, "
                "classification, origin, owner, created_at, updated_at "
                "FROM tasks WHERE id = ?", (task_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_task(row, self._cipher)

    def save(self, task: AgentTask) -> None:
        now = _now()
        result_json, error, metadata_json = _content_columns(task, self._cipher)
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE tasks SET status=?, result=?, error=?, metadata=?, updated_at=? WHERE id=?",
                (str(task.status), result_json, error, metadata_json, now, task.id),
            )

    def save_if_status(self, task: AgentTask, expected: TaskStatus) -> bool:
        """Compare-and-set update: one statement, checked rowcount (no TOCTOU)."""
        now = _now()
        result_json, error, metadata_json = _content_columns(task, self._cipher)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE tasks SET status=?, result=?, error=?, metadata=?, updated_at=? "
                "WHERE id=? AND status=?",
                (str(task.status), result_json, error, metadata_json, now, task.id, str(expected)),
            )
        return cursor.rowcount == 1

    def fail_orphaned(self, error: str = "Task was interrupted by a process restart") -> int:
        """Fail tasks a previous process left non-terminal.

        A daemon worker cannot survive the process, so ``submitted``/``working``
        rows are guaranteed stale when a new ``TaskService`` opens the store.
        The update and the status event are one transaction: a caller that
        paginates the event log sees the terminal transition exactly once.
        """
        now = _now()
        sealed_error = _seal_text(self._cipher, error, "task.error")
        sealed_event = _seal_text(
            self._cipher, json.dumps({"status": "failed", "error": error}), "event.data",
        )
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT id FROM tasks WHERE status IN ('submitted', 'working')",
            ).fetchall()
            if not rows:
                return 0
            self._conn.execute(
                "UPDATE tasks SET status='failed', error=?, updated_at=? "
                "WHERE status IN ('submitted', 'working')",
                (sealed_error, now),
            )
            self._conn.executemany(
                "INSERT INTO task_events (task_id, kind, data, created_at) VALUES (?, ?, ?, ?)",
                [(task_id, "status", sealed_event, now) for (task_id,) in rows],
            )
        return len(rows)

    def append_event(self, task_id: str, kind: str, data: dict[str, Any]) -> None:
        sealed = _seal_text(self._cipher, json.dumps(data, default=str), "event.data")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO task_events (task_id, kind, data, created_at) VALUES (?, ?, ?, ?)",
                (task_id, kind, sealed, _now()),
            )

    def events(self, task_id: str, *, since: int = 0) -> Iterable[dict[str, Any]]:
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
            yield {
                "seq": rowid,
                "kind": kind,
                "data": json.loads(_open_text(self._cipher, data_json, "event.data")),
                "created_at": created_at,
            }

    def list(self, *, context_id: str | None = None, limit: int = 100,
             owner: str | None = None) -> Iterable[AgentTask]:
        with self._lock:
            if context_id is not None and owner is not None:
                cursor = self._conn.execute(
                    "SELECT id, capability, message, status, result, error, metadata, context_id, "
                    "classification, origin, owner, created_at, updated_at "
                    "FROM tasks WHERE context_id = ? AND owner = ? ORDER BY created_at DESC LIMIT ?",
                    (context_id, owner, limit),
                )
            elif context_id is not None:
                cursor = self._conn.execute(
                    "SELECT id, capability, message, status, result, error, metadata, context_id, "
                    "classification, origin, owner, created_at, updated_at "
                    "FROM tasks WHERE context_id = ? ORDER BY created_at DESC LIMIT ?",
                    (context_id, limit),
                )
            elif owner is not None:
                cursor = self._conn.execute(
                    "SELECT id, capability, message, status, result, error, metadata, context_id, "
                    "classification, origin, owner, created_at, updated_at "
                    "FROM tasks WHERE owner = ? ORDER BY created_at DESC LIMIT ?",
                    (owner, limit),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT id, capability, message, status, result, error, metadata, context_id, "
                    "classification, origin, owner, created_at, updated_at "
                    "FROM tasks ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
        for row in rows:
            yield _row_to_task(row, self._cipher)

    def prune(self, retention_days: int) -> int:
        """Delete terminal tasks older than the window, events included."""
        if retention_days <= 0:
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        statuses = (
            TaskStatus.COMPLETED.value, TaskStatus.FAILED.value,
            TaskStatus.CANCELED.value, TaskStatus.REJECTED.value,
        )
        parameters = (*statuses, cutoff)
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM task_events WHERE task_id IN ("
                "SELECT id FROM tasks WHERE status IN (?, ?, ?, ?) AND updated_at < ?)",
                parameters,
            )
            cursor = self._conn.execute(
                "DELETE FROM tasks WHERE status IN (?, ?, ?, ?) AND updated_at < ?",
                parameters,
            )
            return int(cursor.rowcount)

    def rekey(self, old: FieldCipher, new: FieldCipher) -> int:
        """Re-encrypt every sealed column and event payload under a new key.

        Returns the number of rows whose content was re-sealed (task rows and
        event rows). The whole rotation runs in **one transaction**: a value
        the old key cannot open aborts it, and the database is left exactly
        as it was — a half-re-encrypted store would be unreadable with either
        key (ADR 0030).

        Re-running the same pair is safe: a value that is already bound to
        the new key is left untouched instead of being sealed twice.

        The cipher the store itself reads with is **not** changed here — the
        rotation is an operator action and the application is restarted with
        the new key afterwards.
        """
        rekeyed = 0
        with self._lock, self._conn:
            for task_id, message, result, error, metadata in self._conn.execute(
                "SELECT id, message, result, error, metadata FROM tasks ORDER BY rowid",
            ).fetchall():
                new_message, message_changed = reseal_text(old, new, message, context="task.message")
                new_result, result_changed = _reseal_optional(old, new, result, "task.result")
                new_error, error_changed = _reseal_optional(old, new, error, "task.error")
                new_metadata, metadata_changed = reseal_text(old, new, metadata, context="task.metadata")
                if message_changed or result_changed or error_changed or metadata_changed:
                    self._conn.execute(
                        "UPDATE tasks SET message=?, result=?, error=?, metadata=? WHERE id=?",
                        (new_message, new_result, new_error, new_metadata, task_id),
                    )
                    rekeyed += 1
            for rowid, data in self._conn.execute(
                "SELECT rowid, data FROM task_events ORDER BY rowid",
            ).fetchall():
                new_data, changed = reseal_text(old, new, data, context="event.data")
                if changed:
                    self._conn.execute(
                        "UPDATE task_events SET data=? WHERE rowid=?", (new_data, rowid),
                    )
                    rekeyed += 1
        return rekeyed

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


def _content_columns(task: AgentTask, cipher: FieldCipher | None) -> tuple[str | None, str | None, str]:
    """Serialize and seal the three nullable/JSON content columns."""
    result_json = json.dumps(task.result, default=str) if task.result else None
    metadata_json = json.dumps(dict(task.metadata), default=str)
    return (
        _seal_optional(cipher, result_json, "task.result"),
        _seal_optional(cipher, task.error, "task.error"),
        _seal_text(cipher, metadata_json, "task.metadata"),
    )


def _seal_text(cipher: FieldCipher | None, value: str, context: str) -> str:
    return value if cipher is None else cipher.seal(value, context=context)


def _seal_optional(cipher: FieldCipher | None, value: str | None, context: str) -> str | None:
    return None if value is None else _seal_text(cipher, value, context)


def _seal_legacy(cipher: FieldCipher, value: str | None, context: str) -> str | None:
    if value is None or is_sealed(value):
        return value
    return cipher.seal(value, context=context)


def _reseal_optional(
    old: FieldCipher,
    new: FieldCipher,
    value: str | None,
    context: str,
) -> tuple[str | None, bool]:
    """Re-seal a nullable column; ``NULL`` stays ``NULL`` (ADR 0030)."""
    if value is None:
        return None, False
    return reseal_text(old, new, value, context=context)


def _open_text(cipher: FieldCipher | None, value: str, context: str) -> str:
    if cipher is None:
        if is_sealed(value):
            message = "encrypted state cannot be read without AI_STORAGE_KEY or its key file"
            raise StorageKeyError(message)
        return value
    return cipher.open(value, context=context)


def _open_optional(cipher: FieldCipher | None, value: str | None, context: str) -> str | None:
    return None if value is None else _open_text(cipher, value, context)


def _row_to_task(row: tuple[str, ...], cipher: FieldCipher | None) -> AgentTask:
    (tid, capability, message, status_str, result_json, error, metadata_json, context_id,
     classification, origin, owner, created_at, updated_at) = row
    return AgentTask(
        id=tid,
        capability=capability,
        message=_open_text(cipher, message, "task.message"),
        context_id=context_id,
        metadata=json.loads(_open_text(cipher, metadata_json, "task.metadata")) if metadata_json else {},
        # Parsed, not defaulted: the classification decides whether this task may
        # leave the trust boundary, and an unparseable value must fail loudly
        # rather than come back promoted to the permissive default.
        classification=DataClassification(classification),
        status=TaskStatus(status_str),
        origin=ExecutionOrigin(origin),
        owner=owner,
        created_at=_epoch(created_at),
        updated_at=_epoch(updated_at),
        result=json.loads(_open_text(cipher, result_json, "task.result")) if result_json else None,
        error=_open_optional(cipher, error, "task.error"),
    )
