"""SQLite task state is encrypted before it reaches the database file.

These cases exercise the injected cipher, the one-time plaintext migration and
the fail-closed read path when the key is unavailable or wrong.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from ai_assistant.adapters.persistence.sqlite_store import SqliteTaskStore
from ai_assistant.adapters.persistence.storage_crypto import (
    AesGcmFieldCipher,
    StorageDecryptionError,
    StorageKeyError,
    is_sealed,
)
from ai_assistant.domain.tasks import AgentTask, TaskStatus

_KEY = b"a" * 32


def _task(**overrides) -> AgentTask:
    fields = {"capability": "chat", "message": "sensitive prompt"}
    fields.update(overrides)
    return AgentTask.new(**fields)


def _cipher() -> AesGcmFieldCipher:
    return AesGcmFieldCipher(_KEY)


class EncryptedSqliteTaskStoreTests(unittest.TestCase):
    def test_task_and_event_content_is_sealed_on_disk_and_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            store = SqliteTaskStore(path, cipher=_cipher())
            try:
                task = _task(metadata={"secret": "metadata-secret"})
                store.create(task)
                store.append_event(task.id, "status", {"secret": "event-secret"})
                updated = task.transition(TaskStatus.FAILED, error="failure-secret")
                store.save(updated)

                loaded = store.get(task.id)
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded.message, "sensitive prompt")
                self.assertEqual(loaded.metadata, {"secret": "metadata-secret"})
                self.assertEqual(loaded.error, "failure-secret")
                self.assertEqual([item.message for item in store.list()], ["sensitive prompt"])
                self.assertEqual(
                    [item["data"]["secret"] for item in store.events(task.id)],
                    ["event-secret"],
                )
            finally:
                store.close()

            connection = sqlite3.connect(path)
            try:
                row = connection.execute(
                    "SELECT message, error, metadata, result FROM tasks",
                ).fetchone()
                event_data = connection.execute("SELECT data FROM task_events").fetchone()[0]
            finally:
                connection.close()
            self.assertTrue(all(is_sealed(value) for value in row if value is not None))
            self.assertTrue(is_sealed(event_data))
            for value in (*row, event_data):
                self.assertNotIn("secret", value or "")

    def test_completed_result_is_sealed_and_read_back(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SqliteTaskStore(Path(directory) / "tasks.sqlite", cipher=_cipher())
            try:
                task = _task()
                store.create(task)
                store.save(task.transition(TaskStatus.COMPLETED, result={"content": "secret-result"}))
                self.assertEqual(store.get(task.id).result, {"content": "secret-result"})
            finally:
                store.close()

    def test_cas_update_seals_the_new_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            store = SqliteTaskStore(path, cipher=_cipher())
            try:
                task = _task()
                store.create(task)
                self.assertTrue(
                    store.save_if_status(
                        task.transition(TaskStatus.COMPLETED, result={"content": "cas-secret"}),
                        TaskStatus.SUBMITTED,
                    ),
                )
                self.assertEqual(store.get(task.id).result, {"content": "cas-secret"})
            finally:
                store.close()
            connection = sqlite3.connect(path)
            try:
                result = connection.execute("SELECT result FROM tasks").fetchone()[0]
            finally:
                connection.close()
            self.assertTrue(is_sealed(result))

    def test_legacy_plaintext_rows_are_migrated_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            plain = SqliteTaskStore(path)
            try:
                task = _task()
                plain.create(task)
                plain.append_event(task.id, "status", {"secret": "legacy-event"})
            finally:
                plain.close()

            encrypted = SqliteTaskStore(path, cipher=_cipher())
            try:
                self.assertEqual(
                    encrypted._conn.execute("PRAGMA user_version").fetchone()[0], 3,
                )
                self.assertEqual(encrypted.get(task.id).message, "sensitive prompt")
                self.assertEqual(encrypted.events(task.id).__next__()["data"]["secret"], "legacy-event")
            finally:
                encrypted.close()

            connection = sqlite3.connect(path)
            try:
                message = connection.execute("SELECT message FROM tasks").fetchone()[0]
                event_data = connection.execute("SELECT data FROM task_events").fetchone()[0]
            finally:
                connection.close()
            self.assertTrue(is_sealed(message))
            self.assertTrue(is_sealed(event_data))

            # Force the version back to 2: the migration must skip rows that
            # are already sealed instead of double-wrapping them.
            connection = sqlite3.connect(path)
            try:
                connection.execute("PRAGMA user_version = 2")
                connection.commit()
            finally:
                connection.close()
            reopened = SqliteTaskStore(path, cipher=_cipher())
            try:
                self.assertEqual(reopened.get(task.id).message, "sensitive prompt")
            finally:
                reopened.close()

            # A second open sees version 3 and does not re-run the migration.
            reopened = SqliteTaskStore(path, cipher=_cipher())
            try:
                self.assertEqual(reopened.get(task.id).message, "sensitive prompt")
            finally:
                reopened.close()

    def test_a_substituted_plaintext_column_is_refused(self):
        """An unsealed column in a migrated store is not authentic.

        The AEAD exists to reject a value the platform did not write.
        Accepting an unsealed value let anyone with write access to the file
        replace a sealed column with plaintext JSON and have it served as if
        the platform had written it — the GCM tag never got the chance to
        reject it. The binary artifacts already refuse an unsealed file for
        exactly this reason; the text columns now behave the same way.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            store = SqliteTaskStore(path, cipher=_cipher())
            try:
                task = _task()
                store.create(task)
                store.append_event(task.id, "status", {"secret": "event-secret"})
            finally:
                store.close()

            # Substitute the sealed columns, as a process with write access
            # to the state file would.
            connection = sqlite3.connect(path)
            try:
                connection.execute("UPDATE tasks SET message = ? WHERE id = ?", ("substituted", task.id))
                connection.execute(
                    "UPDATE task_events SET data = ? WHERE task_id = ?",
                    ('{"secret": "substituted"}', task.id),
                )
                connection.commit()
            finally:
                connection.close()

            reopened = SqliteTaskStore(path, cipher=_cipher())
            try:
                with self.assertRaises(StorageDecryptionError):
                    reopened.get(task.id)
                with self.assertRaises(StorageDecryptionError):
                    list(reopened.events(task.id))
            finally:
                reopened.close()

    def test_a_store_without_a_cipher_still_serves_plaintext(self):
        """The sealed regime is a property of the *deployment*, not of the
        file: a store opened without a key is a plaintext store by design,
        and refusing its rows would break the opt-in path."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            store = SqliteTaskStore(path)
            try:
                task = _task()
                store.create(task)
                self.assertEqual(store.get(task.id).message, "sensitive prompt")
            finally:
                store.close()

    def test_fail_orphaned_seals_its_error_and_status_event(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            store = SqliteTaskStore(path, cipher=_cipher())
            try:
                task = _task()
                store.create(task)
                self.assertEqual(store.fail_orphaned("restart-secret"), 1)
                self.assertEqual(store.get(task.id).error, "restart-secret")
                self.assertEqual(
                    list(store.events(task.id))[-1]["data"]["error"], "restart-secret",
                )
            finally:
                store.close()
            connection = sqlite3.connect(path)
            try:
                error = connection.execute("SELECT error FROM tasks").fetchone()[0]
                data = connection.execute("SELECT data FROM task_events").fetchone()[0]
            finally:
                connection.close()
            self.assertTrue(is_sealed(error))
            self.assertTrue(is_sealed(data))

    def test_sealed_state_cannot_be_read_without_the_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            store = SqliteTaskStore(path, cipher=_cipher())
            try:
                task = _task()
                store.create(task)
            finally:
                store.close()
            blind = SqliteTaskStore(path)
            try:
                with self.assertRaises(StorageKeyError):
                    blind.get(task.id)
                with self.assertRaises(StorageKeyError):
                    list(blind.list())
            finally:
                blind.close()

    def test_a_wrong_key_or_tampered_row_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            store = SqliteTaskStore(path, cipher=_cipher())
            try:
                task = _task()
                store.create(task)
            finally:
                store.close()

            wrong = SqliteTaskStore(path, cipher=AesGcmFieldCipher(b"b" * 32))
            try:
                with self.assertRaises(StorageDecryptionError):
                    wrong.get(task.id)
            finally:
                wrong.close()

            connection = sqlite3.connect(path)
            try:
                message = connection.execute("SELECT message FROM tasks").fetchone()[0]
                connection.execute("UPDATE tasks SET metadata = ?", (message,))
                connection.commit()
            finally:
                connection.close()
            # The message envelope was moved to the metadata column: the field
            # context differs, so authentication fails before JSON parsing.
            swapped = SqliteTaskStore(path, cipher=_cipher())
            try:
                with self.assertRaises(StorageDecryptionError):
                    swapped.get(task.id)
            finally:
                swapped.close()

    def test_empty_result_and_error_columns_stay_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            store = SqliteTaskStore(path, cipher=_cipher())
            try:
                task = _task(metadata={})
                store.create(task)
                loaded = store.get(task.id)
                self.assertIsNone(loaded.result)
                self.assertIsNone(loaded.error)
                self.assertEqual(loaded.metadata, {})
            finally:
                store.close()
            connection = sqlite3.connect(path)
            try:
                row = connection.execute("SELECT result, error, metadata FROM tasks").fetchone()
            finally:
                connection.close()
            self.assertIsNone(row[0])
            self.assertIsNone(row[1])
            self.assertTrue(is_sealed(row[2]))


if __name__ == "__main__":
    unittest.main()
