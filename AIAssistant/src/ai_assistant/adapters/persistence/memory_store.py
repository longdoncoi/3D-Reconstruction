"""Sealed SQLite store for long-term memory records (ADR 0043).

The store is dumb on purpose (the ``LongTermMemoryPort`` contract): it
persists already-validated records and hands candidate sets back. What it
*does* own is the security half, and it owns it the way the other sealed
stores do (ADR 0027/0036):

* the cipher is **required** — an unkeyed store is refused at construction,
  never a plaintext fallback;
* the payload (owner, kind, text, metadata) is sealed with a **row-bound**
  context, so a ciphertext copied to another row fails authentication;
* a row that cannot be opened, parsed, or whose sealed identity does not
  match the plain-text filter columns is **skipped**, never served — the
  columns are a filter hint, the payload is the truth;
* an unknown ``kind`` is refused on the way in *and* on the way out: the
  vocabulary is closed, and a tampered file must not smuggle a kind the
  recall path has no policy for.

Thread-safety mirrors ``SqliteTaskStore``: one connection with
``check_same_thread=False`` behind an ``RLock`` — runs execute on worker
threads while prune/close run on the lifespan thread.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

from ai_assistant.domain.memory import MemoryRecord, is_memory_kind
from ai_assistant.ports import (
    FieldCipher,
    LongTermMemoryPort,
    StorageDecryptionError,
    StorageKeyError,
)

logger = logging.getLogger("ai_assistant.persistence.memory_store")

#: The AEAD context family of every memory payload. Distinct from the task
#: store's contexts, so a ciphertext cannot be moved between the two files.
MEMORY_CONTEXT_PREFIX = "memory.v1"


def memory_context(record_id: str) -> str:
    """The AEAD associated data of one memory payload.

    Bound to the row id (the ADR 0036 row-binding rule): a payload copied to
    another row fails authentication instead of being served as that row's
    memory.
    """
    return f"{MEMORY_CONTEXT_PREFIX}:{record_id}:payload"


def _as_float(value: object) -> float | None:
    """Read one numeric column, or ``None`` when it is not numeric.

    The columns are written by this adapter, but the file is attacker-writable
    (ADR 0036 treats the whole store as untrusted). ``float()`` on a tampered
    value raises, and a raise here would escape the per-row skip and drop
    **every** candidate of the scan instead of the one bad row — the exact
    opposite of the "bad rows are skipped, never served" contract. Returning
    ``None`` keeps the failure local to the row.
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class MemoryStore:
    """``LongTermMemoryPort`` over SQLite with row-bound sealed payloads."""

    def __init__(self, path: str | Path, *, cipher: FieldCipher | None = None) -> None:
        if cipher is None:
            # Fail closed (ADR 0027): memory records carry user facts and run
            # traces, so a store that cannot seal them must not be built.
            msg = "the memory store requires the at-rest key (no plaintext fallback)"
            raise StorageKeyError(msg)
        self.path = str(path)
        self._cipher = cipher
        self._lock = threading.RLock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id           TEXT PRIMARY KEY,
                    owner        TEXT NOT NULL,
                    kind         TEXT NOT NULL,
                    salience     REAL NOT NULL,
                    created_at   REAL NOT NULL,
                    last_used_at REAL NOT NULL,
                    payload      TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_owner_kind
                    ON memories(owner, kind, created_at);
            """)

    def add(self, record: MemoryRecord) -> None:
        """Persist one record, replacing a row with the same id.

        ``INSERT OR REPLACE`` is what makes the procedural record's
        signature-derived id a *reuse* rather than a pile: rerunning the same
        task refreshes its plan instead of adding a copy.

        Replacement is scoped to the **same owner**. The service namespaces
        ids by owner (``domain.memory._owner_namespace``), so a cross-owner
        collision should be impossible; refusing it here keeps a bug in that
        naming, or a caller writing the store directly, from erasing another
        tenant's row. The refusal is an error, not a silent skip: a write
        that cannot be applied is a caller defect the consolidation layer
        logs and counts as a miss.
        """
        kind = record["kind"]
        if not is_memory_kind(kind):
            msg = f"refusing to store unknown memory kind {kind!r}"
            raise ValueError(msg)
        payload = self._cipher.seal(
            json.dumps({
                "owner": record["owner"],
                "kind": kind,
                "text": record["text"],
                "metadata": record["metadata"],
            }, ensure_ascii=False, default=str),
            context=memory_context(record["id"]),
        )
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT owner FROM memories WHERE id = ?", (record["id"],),
            ).fetchone()
            if existing is not None and existing[0] != record["owner"]:
                msg = (f"refusing to replace memory {record['id']!r}: "
                       "the row belongs to another owner")
                raise ValueError(msg)
            self._conn.execute(
                "INSERT OR REPLACE INTO memories "
                "(id, owner, kind, salience, created_at, last_used_at, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (record["id"], record["owner"], kind, float(record["salience"]),
                 float(record["created_at"]), float(record["last_used_at"]), payload),
            )

    def candidates(self, owner: str, kinds: frozenset[str],
                   *, limit: int = 200) -> list[MemoryRecord]:
        """The owner's rows in *kinds*, newest first; bad rows are skipped.

        The plain-text columns only narrow the scan. Everything a caller
        receives was authenticated as *this* row by the sealed payload: an
        unopenable or unparseable payload, a sealed kind outside the
        vocabulary, or a sealed identity that disagrees with the columns all
        drop the row (ADR 0036: never serve what cannot be authenticated).
        """
        # An empty ``kinds`` yields the empty placeholder list and SQLite's
        # ``kind IN ()``, which matches nothing — no early return needed, and
        # the empty set therefore cannot mean "every kind" by accident.
        # The only interpolation is the placeholder list itself — one "?"
        # per kind, joined by commas; every bound value stays a parameter.
        placeholders = ",".join("?" for _ in kinds)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, kind, salience, created_at, last_used_at, payload "  # noqa: S608
                f"FROM memories WHERE owner = ? AND kind IN ({placeholders}) "
                f"ORDER BY created_at DESC, id ASC LIMIT ?",
                (owner, *sorted(kinds), int(limit)),
            ).fetchall()
        records: list[MemoryRecord] = []
        for row_id, kind, salience, created_at, last_used_at, payload in rows:
            record = self._rehydrate(row_id, kind, salience, created_at,
                                     last_used_at, payload, owner)
            if record is not None:
                records.append(record)
        return records

    def _rehydrate(self, row_id: str, kind: str, salience: float,
                   created_at: float, last_used_at: float, payload: str,
                   owner: str) -> MemoryRecord | None:
        """Open one row's payload and re-check it against the columns."""
        try:
            data = json.loads(self._cipher.open(payload, context=memory_context(row_id)))
        except (StorageDecryptionError, ValueError, TypeError) as error:
            logger.warning("Skipping unauthentic memory row %s: %s", row_id, error)
            return None
        if not isinstance(data, dict):
            logger.warning("Skipping memory row %s with a non-object payload", row_id)
            return None
        sealed_kind = data.get("kind")
        if not is_memory_kind(sealed_kind) or sealed_kind != kind or data.get("owner") != owner:
            logger.warning("Skipping memory row %s with a mismatched sealed identity", row_id)
            return None
        created = _as_float(created_at)
        last_used = _as_float(last_used_at)
        salience_value = _as_float(salience)
        if created is None or last_used is None or salience_value is None:
            logger.warning("Skipping memory row %s with non-numeric columns", row_id)
            return None
        text = data.get("text")
        metadata = data.get("metadata")
        return {
            "id": row_id,
            "kind": sealed_kind,
            "owner": owner,
            "text": text if isinstance(text, str) else "",
            "created_at": created,
            "last_used_at": last_used,
            "salience": salience_value,
            "metadata": metadata if isinstance(metadata, dict) else {},
        }

    def prune(self, retention_days: int, *, max_bytes: int = 0) -> int:
        """Drop rows past the retention window, then shrink to the byte cap.

        Both halves evict oldest-first; the byte half exists because sealed
        payloads grow with every consolidated run and a desktop install must
        not accumulate forever. Returns how many rows went.
        """
        cutoff = time.time() - max(1, int(retention_days)) * 86400.0
        with self._lock, self._conn:
            removed = self._conn.execute(
                "DELETE FROM memories WHERE created_at < ?", (cutoff,),
            ).rowcount
            if max_bytes > 0:
                removed += self._shrink_to_bytes(int(max_bytes))
        return max(0, removed)

    def _shrink_to_bytes(self, max_bytes: int) -> int:
        """Delete oldest rows until the sealed payloads fit *max_bytes*.

        ``max_bytes > 0`` and a total above it imply at least one row exists
        (every sealed payload is non-empty), so the loop always finds a row
        to delete while it is over budget.
        """
        removed = 0
        while True:
            (total,) = self._conn.execute(
                "SELECT COALESCE(SUM(LENGTH(payload)), 0) FROM memories",
            ).fetchone()
            if total <= max_bytes:
                return removed
            oldest = self._conn.execute(
                "SELECT id FROM memories ORDER BY created_at ASC, id ASC LIMIT 1",
            ).fetchone()
            self._conn.execute("DELETE FROM memories WHERE id = ?", (oldest[0],))
            removed += 1

    def close(self) -> None:
        with self._lock:
            self._conn.close()


#: Structural conformance, checked by mypy in the strict gate: the adapter
#: satisfies ``LongTermMemoryPort`` — it is injected through that contract
#: only, and the composition root never names the concrete class to a caller.
_LongTermMemoryImplementation: type[LongTermMemoryPort] = MemoryStore

__all__ = ["MEMORY_CONTEXT_PREFIX", "MemoryStore", "memory_context"]
