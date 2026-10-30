"""The sealed long-term memory store: refusal, row-binding, retention (ADR 0043).

The store is the trust boundary of the memory subsystem, so the tests here
are the ADR 0036 shape: a store with a key never serves what it cannot
authenticate. Each refusal path is pinned — no cipher at construction, an
unknown kind on the way in, an unopenable payload, a non-object payload, a
sealed identity that disagrees with the plain-text filter columns, an
unknown sealed kind — and the retention rules (age window, byte cap,
oldest-first) are measured rather than assumed.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from ai_assistant.adapters.persistence.memory_store import (
    MemoryStore,
    memory_context,
)
from ai_assistant.adapters.persistence.storage_crypto import (
    AesGcmFieldCipher,
    StorageKeyError,
)
from ai_assistant.domain.memory import MemoryRecord

#: The store's retention arithmetic runs against the wall clock, so the
#: fixtures anchor to ``time.time()`` — a fixed epoch would make every row
#: look years old the moment the calendar moved past it.
NOW = time.time()
DAY = 86400.0


def _record(**overrides: object) -> MemoryRecord:
    base: dict[str, object] = {
        "id": "m1",
        "kind": "semantic",
        "owner": "u",
        "text": "nguoi dung thich mau xanh",
        "created_at": NOW,
        "last_used_at": NOW,
        "salience": 0.5,
        "metadata": {"session_id": "s1"},
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


class MemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "memory.sqlite"
        self.cipher = AesGcmFieldCipher(b"m" * 32)

    def _open(self) -> MemoryStore:
        store = MemoryStore(self.path, cipher=self.cipher)
        self.addCleanup(store.close)
        return store

    def _insert_raw(self, row_id: str, payload: str) -> None:
        """Insert one row straight through SQL — the tamper surface."""
        connection = sqlite3.connect(self.path)
        with connection:
            connection.execute(
                "INSERT INTO memories VALUES (?, 'u', 'semantic', 0.5, ?, ?, ?)",
                (row_id, NOW, NOW, payload),
            )
        connection.close()

    def _ensure_table(self) -> None:
        store = MemoryStore(self.path, cipher=self.cipher)
        store.close()

    def test_a_store_without_a_key_is_refused_not_degraded(self) -> None:
        """ADR 0027: an unkeyed store would write user facts in the clear."""
        with self.assertRaises(StorageKeyError):
            MemoryStore(self.path, cipher=None)

    def test_round_trip_returns_the_record_the_service_wrote(self) -> None:
        store = self._open()
        store.add(_record())
        found = store.candidates("u", frozenset({"semantic"}))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["text"], "nguoi dung thich mau xanh")
        self.assertEqual(found[0]["metadata"], {"session_id": "s1"})
        self.assertEqual(found[0]["kind"], "semantic")

    def test_a_second_store_instance_reads_the_same_rows(self) -> None:
        """Durability, not just an in-memory round trip."""
        first = MemoryStore(self.path, cipher=self.cipher)
        first.add(_record())
        first.close()
        reopened = self._open()
        self.assertEqual(len(reopened.candidates("u", frozenset({"semantic"}))), 1)

    def test_the_payload_on_disk_is_sealed_not_plaintext(self) -> None:
        store = self._open()
        store.add(_record(text="cong nghe 3d reconstruction"))
        connection = sqlite3.connect(self.path)
        raw = connection.execute("SELECT payload FROM memories").fetchone()[0]
        connection.close()
        self.assertTrue(raw.startswith("enc:v1:"))
        self.assertNotIn("cong nghe", raw)

    def test_the_context_is_bound_to_the_row_id(self) -> None:
        """The AEAD context carries the id, so a payload cannot be reparented."""
        self.assertEqual(memory_context("abc"), "memory.v1:abc:payload")

    def test_an_unknown_kind_is_refused_on_the_way_in(self) -> None:
        store = self._open()
        with self.assertRaises(ValueError):
            store.add(_record(kind="dream"))  # type: ignore[arg-type]
        self.assertEqual(store.candidates("u", frozenset({"semantic"})), [])

    def test_rerunning_the_same_id_replaces_instead_of_piling_up(self) -> None:
        """The signature-derived procedural id makes reuse a replace."""
        store = self._open()
        store.add(_record(id="proc-x", kind="procedural", text="buoc 1"))
        store.add(_record(id="proc-x", kind="procedural", text="buoc 2"))
        found = store.candidates("u", frozenset({"procedural"}))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["text"], "buoc 2")

    def test_a_cross_owner_id_collision_is_refused_not_replaced(self) -> None:
        """Replacement is scoped to the owner: the id namespace should keep
        this from happening, and the store must not erase another tenant's
        row if it ever does (ADR 0043)."""
        store = self._open()
        store.add(_record(id="shared", owner="alice", text="alice fact"))
        with self.assertRaises(ValueError):
            store.add(_record(id="shared", owner="bob", text="bob fact"))
        self.assertEqual(
            [r["text"] for r in store.candidates("alice", frozenset({"semantic"}))],
            ["alice fact"],
        )
        self.assertEqual(store.candidates("bob", frozenset({"semantic"})), [])

    def test_an_empty_kind_set_scans_nothing(self) -> None:
        store = self._open()
        store.add(_record())
        self.assertEqual(store.candidates("u", frozenset()), [])

    def test_another_owner_rows_are_never_candidates(self) -> None:
        store = self._open()
        store.add(_record(owner="alice"))
        self.assertEqual(store.candidates("bob", frozenset({"semantic"})), [])

    def test_limit_and_order_newest_first(self) -> None:
        store = self._open()
        store.add(_record(id="old", created_at=NOW - 3 * DAY))
        store.add(_record(id="new", created_at=NOW))
        found = store.candidates("u", frozenset({"semantic"}), limit=1)
        self.assertEqual([r["id"] for r in found], ["new"])

    def test_a_row_whose_payload_cannot_be_opened_is_skipped(self) -> None:
        """ADR 0036: tampered or foreign-key ciphertext is never served."""
        store = self._open()
        store.add(_record(id="good"))
        store.add(_record(id="bad"))
        connection = sqlite3.connect(self.path)
        with connection:
            connection.execute(
                "UPDATE memories SET payload = 'enc:v1:AAAAAAAA' WHERE id = 'bad'",
            )
        connection.close()
        found = store.candidates("u", frozenset({"semantic"}))
        self.assertEqual([r["id"] for r in found], ["good"])

    def test_a_payload_sealed_under_another_context_is_skipped(self) -> None:
        """A ciphertext moved from another store fails the row-bound context."""
        self._ensure_table()
        foreign = self.cipher.seal(
            json.dumps({"kind": "semantic", "owner": "u", "text": "x", "metadata": {}}),
            context="task.v2:t1:message",
        )
        self._insert_raw("mover", foreign)
        store = self._open()
        self.assertEqual(store.candidates("u", frozenset({"semantic"})), [])

    def test_a_non_object_payload_is_skipped(self) -> None:
        self._ensure_table()
        sealed = self.cipher.seal(json.dumps(["not", "an", "object"]),
                                  context=memory_context("listy"))
        self._insert_raw("listy", sealed)
        store = self._open()
        self.assertEqual(store.candidates("u", frozenset({"semantic"})), [])

    def test_a_sealed_kind_outside_the_vocabulary_is_skipped(self) -> None:
        self._ensure_table()
        sealed = self.cipher.seal(
            json.dumps({"kind": "dream", "owner": "u", "text": "x", "metadata": {}}),
            context=memory_context("dreamer"),
        )
        self._insert_raw("dreamer", sealed)
        store = self._open()
        self.assertEqual(store.candidates("u", frozenset({"semantic"})), [])

    def test_a_sealed_identity_that_disagrees_with_the_columns_is_skipped(self) -> None:
        """The columns are a filter hint; the payload is the truth."""
        self._ensure_table()
        sealed = self.cipher.seal(
            json.dumps({"kind": "semantic", "owner": "mallory", "text": "x", "metadata": {}}),
            context=memory_context("spoof"),
        )
        self._insert_raw("spoof", sealed)
        store = self._open()
        self.assertEqual(store.candidates("u", frozenset({"semantic"})), [])

    def test_a_non_string_text_or_metadata_degrades_to_safe_defaults(self) -> None:
        self._ensure_table()
        sealed = self.cipher.seal(
            json.dumps({"kind": "semantic", "owner": "u", "text": 42, "metadata": "nope"}),
            context=memory_context("weird"),
        )
        self._insert_raw("weird", sealed)
        store = self._open()
        found = store.candidates("u", frozenset({"semantic"}))
        self.assertEqual(found[0]["text"], "")
        self.assertEqual(found[0]["metadata"], {})

    def test_a_non_numeric_column_skips_only_that_row(self) -> None:
        """A tampered timestamp must not take the healthy rows with it.

        ``float()`` used to run outside the per-row guard, so one non-numeric
        column raised out of ``candidates`` and the caller lost the whole
        recall (``MemoryService`` contains it as "remember nothing"). The row
        has to be dropped locally instead — the ADR 0036 shape.
        """
        store = self._open()
        store.add(_record(id="good"))
        store.add(_record(id="bad"))
        connection = sqlite3.connect(self.path)
        with connection:
            connection.execute(
                "UPDATE memories SET salience = 'not-a-number' WHERE id = 'bad'",
            )
        connection.close()
        found = store.candidates("u", frozenset({"semantic"}))
        self.assertEqual([r["id"] for r in found], ["good"])

    def test_a_blob_in_a_numeric_column_is_skipped_too(self) -> None:
        """``float(b"\\x00")`` raises ``TypeError``, the other coercion failure."""
        store = self._open()
        store.add(_record(id="good"))
        store.add(_record(id="bad"))
        connection = sqlite3.connect(self.path)
        with connection:
            connection.execute(
                "UPDATE memories SET created_at = X'00' WHERE id = 'bad'",
            )
        connection.close()
        found = store.candidates("u", frozenset({"semantic"}))
        self.assertEqual([r["id"] for r in found], ["good"])

    def test_prune_drops_only_rows_past_the_retention_window(self) -> None:
        store = self._open()
        store.add(_record(id="old", created_at=NOW - 100 * DAY))
        store.add(_record(id="kept", created_at=NOW - 1 * DAY))
        removed = store.prune(90)
        self.assertEqual(removed, 1)
        self.assertEqual(
            [r["id"] for r in store.candidates("u", frozenset({"semantic"}))], ["kept"],
        )

    def test_a_non_positive_retention_never_wipes_the_store(self) -> None:
        """``max(1, days)``: a misconfigured 0 must not delete everything."""
        store = self._open()
        store.add(_record(created_at=NOW))
        self.assertEqual(store.prune(0), 0)
        self.assertEqual(len(store.candidates("u", frozenset({"semantic"}))), 1)

    def test_prune_without_a_byte_cap_reports_only_the_age_evictions(self) -> None:
        store = self._open()
        store.add(_record(created_at=NOW - 1 * DAY))
        self.assertEqual(store.prune(90, max_bytes=0), 0)

    def test_the_byte_cap_evicts_oldest_first(self) -> None:
        store = self._open()
        store.add(_record(id="old", created_at=NOW - 2 * DAY, text="x" * 500))
        store.add(_record(id="new", created_at=NOW, text="y" * 500))
        # The sealed payload is base64-inflated, so the cap is measured, not
        # guessed: one record's worth of bytes keeps the newest, evicts the
        # oldest and must not empty the store.
        connection = sqlite3.connect(self.path)
        (one_record,) = connection.execute(
            "SELECT LENGTH(payload) FROM memories WHERE id = 'new'",
        ).fetchone()
        connection.close()
        removed = store.prune(90, max_bytes=one_record + 10)
        self.assertEqual(removed, 1)
        self.assertEqual(
            [r["id"] for r in store.candidates("u", frozenset({"semantic"}))], ["new"],
        )

    def test_a_generous_byte_cap_evicts_nothing(self) -> None:
        store = self._open()
        store.add(_record(text="ngan"))
        self.assertEqual(store.prune(90, max_bytes=10_000), 0)

    def test_close_is_idempotent(self) -> None:
        store = MemoryStore(self.path, cipher=self.cipher)
        store.close()
        store.close()


if __name__ == "__main__":
    unittest.main()
