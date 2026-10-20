"""Re-keying the local state — the rotation path ADR 0027 left open (ADR 0030).

The task store, the pending-action store and the RAG cache are sealed with
**one** key (ADR 0027/0028). Until ADR 0030 there was no way to change that
key: a new ``AI_STORAGE_KEY`` made every row unreadable. These cases pin the
rotation that closed the gap:

* a round trip — seal with the old key, rotate, read with the new key, and
  the old key can no longer open the state;
* a wrong old key aborts and leaves the state exactly as it was (both
  stores);
* an empty store is a no-op that does not fail;
* running the rotation twice leaves the state readable (nothing sealed
  twice);
* the command-line tool verifies the key, previews the rotation without
  writing, and applies it.
"""
from __future__ import annotations

import base64
import importlib.util
import io
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from ai_assistant.adapters.persistence.pending_store import PendingActionStore
from ai_assistant.adapters.persistence.sqlite_store import SqliteTaskStore
from ai_assistant.adapters.persistence.storage_crypto import (
    AesGcmFieldCipher,
    StorageCryptoError,
    StorageDecryptionError,
    is_sealed,
)
from ai_assistant.config.paths import resolve_paths
from ai_assistant.domain.tasks import AgentTask, TaskStatus

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "rekey_storage.py"

_OLD_KEY = b"a" * 32
_NEW_KEY = b"b" * 32
_THIRD_KEY = b"c" * 32


def _cipher(key: bytes) -> AesGcmFieldCipher:
    return AesGcmFieldCipher(key)


def _key_text(key: bytes) -> str:
    """The key in the ``base64:<url-safe>`` form the tooling accepts."""
    return "base64:" + base64.urlsafe_b64encode(key).decode("ascii")


def _task(message: str = "sensitive prompt") -> AgentTask:
    return AgentTask.new(capability="chat", message=message, metadata={"secret": "metadata"})


def _load_script():
    spec = importlib.util.spec_from_file_location("rekey_storage", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # The dataclass decorator looks the module up in ``sys.modules``; a module
    # that is not registered there cannot be executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _raw_rows(db_path: Path) -> list[tuple[object, ...]]:
    """The sealed columns of every task row, as they sit on disk."""
    connection = sqlite3.connect(str(db_path))
    try:
        return connection.execute(
            "SELECT id, message, result, error, metadata FROM tasks ORDER BY rowid",
        ).fetchall()
    finally:
        connection.close()


class TaskStoreRekeyTests(unittest.TestCase):
    """The SQLite task store rotates every sealed column and event payload."""

    def test_rekey_rotates_every_column_and_event_and_retires_the_old_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            store = SqliteTaskStore(path, cipher=_cipher(_OLD_KEY))
            try:
                task = _task()
                store.create(task)
                store.append_event(task.id, "status", {"secret": "event-secret"})
                store.save(task.transition(
                    TaskStatus.COMPLETED, result={"content": "result-secret"}, error="error-secret",
                ))
            finally:
                store.close()

            store = SqliteTaskStore(path)
            try:
                self.assertEqual(store.rekey(_cipher(_OLD_KEY), _cipher(_NEW_KEY)), 2)
            finally:
                store.close()

            rotated = SqliteTaskStore(path, cipher=_cipher(_NEW_KEY))
            try:
                loaded = rotated.get(task.id)
                self.assertEqual(loaded.message, "sensitive prompt")
                self.assertEqual(loaded.metadata, {"secret": "metadata"})
                self.assertEqual(loaded.result, {"content": "result-secret"})
                self.assertEqual(loaded.error, "error-secret")
                self.assertEqual(
                    [item["data"]["secret"] for item in rotated.events(task.id)],
                    ["event-secret"],
                )
            finally:
                rotated.close()

            retired = SqliteTaskStore(path, cipher=_cipher(_OLD_KEY))
            try:
                with self.assertRaises(StorageDecryptionError):
                    retired.get(task.id)
                with self.assertRaises(StorageDecryptionError):
                    list(retired.events(task.id))
            finally:
                retired.close()

    def test_rekey_with_a_wrong_old_key_aborts_and_keeps_the_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            store = SqliteTaskStore(path, cipher=_cipher(_OLD_KEY))
            try:
                first = _task("first")
                second = _task("second")
                store.create(first)
                store.create(second)
            finally:
                store.close()

            # The second row is sealed with a third key: the rotation must
            # refuse to continue instead of leaving one row re-encrypted.
            connection = sqlite3.connect(str(path))
            try:
                connection.execute(
                    "UPDATE tasks SET message=? WHERE id=?",
                    (_cipher(_THIRD_KEY).seal("x", context="task.message"), second.id),
                )
                connection.commit()
            finally:
                connection.close()
            before = _raw_rows(path)

            store = SqliteTaskStore(path, cipher=_cipher(_OLD_KEY))
            try:
                with self.assertRaises(StorageDecryptionError):
                    store.rekey(_cipher(_THIRD_KEY), _cipher(_NEW_KEY))
                # The rollback left every row readable with the old key —
                # including the row the rotation had already rewritten.
                self.assertEqual(store.get(first.id).message, "first")
            finally:
                store.close()
            self.assertEqual(_raw_rows(path), before)

    def test_rekey_twice_leaves_the_state_readable_and_unsealed_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            store = SqliteTaskStore(path, cipher=_cipher(_OLD_KEY))
            try:
                task = _task()
                store.create(task)
                store.append_event(task.id, "status", {"secret": "event-secret"})
            finally:
                store.close()

            store = SqliteTaskStore(path)
            try:
                self.assertEqual(store.rekey(_cipher(_OLD_KEY), _cipher(_NEW_KEY)), 2)
            finally:
                store.close()
            rotated = _raw_rows(path)

            store = SqliteTaskStore(path)
            try:
                # The second run finds the value already bound to the new key:
                # nothing is sealed a second time.
                self.assertEqual(store.rekey(_cipher(_OLD_KEY), _cipher(_NEW_KEY)), 0)
            finally:
                store.close()
            self.assertEqual(_raw_rows(path), rotated)

            reader = SqliteTaskStore(path, cipher=_cipher(_NEW_KEY))
            try:
                self.assertEqual(reader.get(task.id).message, "sensitive prompt")
                self.assertEqual(
                    [item["data"]["secret"] for item in reader.events(task.id)],
                    ["event-secret"],
                )
            finally:
                reader.close()

    def test_rekey_of_an_empty_store_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            store = SqliteTaskStore(path, cipher=_cipher(_OLD_KEY))
            try:
                self.assertEqual(store.rekey(_cipher(_OLD_KEY), _cipher(_NEW_KEY)), 0)
            finally:
                store.close()

    def test_rekey_keeps_null_columns_null_and_seals_legacy_plaintext(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            plain = SqliteTaskStore(path)
            try:
                task = _task()
                plain.create(task)
            finally:
                plain.close()

            store = SqliteTaskStore(path)
            try:
                self.assertEqual(store.rekey(_cipher(_OLD_KEY), _cipher(_NEW_KEY)), 1)
            finally:
                store.close()

            row = _raw_rows(path)[0]
            self.assertIsNone(row[2])
            self.assertIsNone(row[3])
            self.assertTrue(is_sealed(row[1]))
            self.assertTrue(is_sealed(row[4]))

            reader = SqliteTaskStore(path, cipher=_cipher(_NEW_KEY))
            try:
                loaded = reader.get(task.id)
                self.assertEqual(loaded.message, "sensitive prompt")
                self.assertIsNone(loaded.result)
                self.assertIsNone(loaded.error)
            finally:
                reader.close()


class PendingStoreRekeyTests(unittest.TestCase):
    """The pending-action store rotates its single sealed payload."""

    def test_rekey_rotates_the_payload_and_retires_the_old_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.json"
            store = PendingActionStore(path, cipher=_cipher(_OLD_KEY))
            store["act-1"] = {"tool": "write_file", "params": {"path": "secret.txt"}}
            store.save()

            self.assertEqual(store.rekey(_cipher(_OLD_KEY), _cipher(_NEW_KEY)), 1)

            rotated = PendingActionStore(path, cipher=_cipher(_NEW_KEY))
            rotated.load()
            self.assertEqual(rotated["act-1"]["tool"], "write_file")

            retired = PendingActionStore(path, cipher=_cipher(_OLD_KEY))
            with self.assertRaises(StorageDecryptionError):
                retired.load()

    def test_rekey_with_a_wrong_old_key_keeps_the_file_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.json"
            store = PendingActionStore(path, cipher=_cipher(_OLD_KEY))
            store["act-1"] = {"tool": "write_file"}
            store.save()
            before = path.read_bytes()

            with self.assertRaises(StorageDecryptionError):
                store.rekey(_cipher(_THIRD_KEY), _cipher(_NEW_KEY))
            self.assertEqual(path.read_bytes(), before)

            rotated = PendingActionStore(path, cipher=_cipher(_OLD_KEY))
            rotated.load()
            self.assertEqual(rotated["act-1"]["tool"], "write_file")

    def test_rekey_of_an_empty_store_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "pending.json"
            self.assertEqual(PendingActionStore(missing).rekey(_cipher(_OLD_KEY), _cipher(_NEW_KEY)), 0)

            empty = Path(directory) / "empty.json"
            empty.write_text("", encoding="utf-8")
            self.assertEqual(PendingActionStore(empty).rekey(_cipher(_OLD_KEY), _cipher(_NEW_KEY)), 0)
            self.assertEqual(empty.read_text(encoding="utf-8"), "")

    def test_rekey_twice_leaves_the_payload_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.json"
            store = PendingActionStore(path, cipher=_cipher(_OLD_KEY))
            store["act-1"] = {"tool": "write_file"}
            store.save()

            self.assertEqual(store.rekey(_cipher(_OLD_KEY), _cipher(_NEW_KEY)), 1)
            rotated = path.read_bytes()
            self.assertEqual(store.rekey(_cipher(_OLD_KEY), _cipher(_NEW_KEY)), 0)
            self.assertEqual(path.read_bytes(), rotated)

            reader = PendingActionStore(path, cipher=_cipher(_NEW_KEY))
            reader.load()
            self.assertEqual(reader["act-1"]["tool"], "write_file")

    def test_rekey_reports_a_state_file_it_cannot_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.json"
            path.mkdir()
            with self.assertRaises(StorageCryptoError):
                PendingActionStore(path).rekey(_cipher(_OLD_KEY), _cipher(_NEW_KEY))


class RekeyToolTests(unittest.TestCase):
    """The operator tool: verify, preview and apply the rotation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = _load_script()

    def _workspace(self, directory: Path) -> Path:
        """A state directory with one task, one event and one pending action."""
        state = directory / "state"
        state.mkdir()
        store = SqliteTaskStore(state / "tasks.sqlite", cipher=_cipher(_OLD_KEY))
        task = _task()
        store.create(task)
        store.append_event(task.id, "status", {"secret": "event-secret"})
        store.close()
        pending = PendingActionStore(state / "pending_agent_actions.json", cipher=_cipher(_OLD_KEY))
        pending["act-1"] = {"tool": "write_file"}
        pending.save()
        cache = state / "Cache"
        cache.mkdir()
        for name in self.script.CACHE_ARTIFACTS:
            (cache / name).write_bytes(b"sealed-artifact")
        return state

    def _run(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = self.script.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_verify_reports_what_the_configured_key_can_open(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self._workspace(Path(directory))
            code, out, err = self._run(
                "--data-dir", str(state), "--old-key", _key_text(_OLD_KEY), "--verify",
            )
        self.assertEqual(code, 0)
        self.assertIn("task rows readable        : 1/1", out)
        self.assertIn("event payloads readable   : 1/1", out)
        self.assertIn("pending payloads readable : 1/1", out)
        self.assertEqual(err, "")

    def test_verify_fails_with_a_wrong_key(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self._workspace(Path(directory))
            code, out, err = self._run(
                "--data-dir", str(state), "--old-key", _key_text(_THIRD_KEY), "--verify",
            )
        self.assertEqual(code, 1)
        self.assertIn("VERIFY FAIL", err)
        self.assertIn("1 task row(s)", err)
        self.assertIn("task rows readable        : 0/1", out)

    def test_verify_uses_the_configured_key_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self._workspace(Path(directory))
            with mock.patch.dict(os.environ, {"AI_STORAGE_KEY": _key_text(_OLD_KEY)}):
                code, out, _ = self._run("--data-dir", str(state), "--verify")
            self.assertEqual(code, 0)
            self.assertIn("OK: the configured key opens every sealed value", out)

            with mock.patch.dict(os.environ, {"AI_STORAGE_KEY": _key_text(_THIRD_KEY)}):
                code, _, err = self._run("--data-dir", str(state), "--verify")
            self.assertEqual(code, 1)
            self.assertIn("VERIFY FAIL", err)

    def test_verify_reads_the_configured_key_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self._workspace(root)
            key_file = root / "old.key"
            key_file.write_text(_key_text(_OLD_KEY), encoding="ascii")
            with mock.patch.dict(os.environ, {"AI_STORAGE_KEY": "", "AI_STORAGE_KEY_FILE": str(key_file)}):
                code, _, _ = self._run("--data-dir", str(state), "--verify")
            self.assertEqual(code, 0)

    def test_verify_fails_when_no_key_is_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self._workspace(root)
            environment = {
                "AI_STORAGE_KEY": "",
                "AI_STORAGE_KEY_FILE": str(root / "missing.key"),
                "LOCALAPPDATA": str(root),
                "APPDATA": str(root),
                "XDG_CONFIG_HOME": str(root),
            }
            with mock.patch.dict(os.environ, environment):
                code, _, err = self._run("--data-dir", str(state), "--verify")
            self.assertEqual(code, 1)
            self.assertIn("does not exist", err)

            with mock.patch.dict(os.environ, {**environment, "AI_STORAGE_KEY_FILE": ""}):
                code, _, err = self._run("--data-dir", str(state), "--verify")
            self.assertEqual(code, 1)
            self.assertIn("no at-rest key is configured", err)

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self._workspace(Path(directory))
            db_before = (state / "tasks.sqlite").read_bytes()
            pending_before = (state / "pending_agent_actions.json").read_bytes()

            code, out, err = self._run(
                "--data-dir", str(state),
                "--old-key", _key_text(_OLD_KEY),
                "--new-key", _key_text(_NEW_KEY),
                "--dry-run",
            )
            self.assertEqual(code, 0)
            self.assertIn("DRY RUN", out)
            self.assertIn("rows that would be re-encrypted      : 2", out)
            self.assertIn("pending payloads that would be sealed: 1", out)
            self.assertEqual(err, "")

            self.assertEqual((state / "tasks.sqlite").read_bytes(), db_before)
            self.assertEqual((state / "pending_agent_actions.json").read_bytes(), pending_before)
            for name in self.script.CACHE_ARTIFACTS:
                self.assertTrue((state / "Cache" / name).exists())

    def test_the_rotation_rekeys_the_state_and_drops_the_derived_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self._workspace(Path(directory))
            code, out, err = self._run(
                "--data-dir", str(state),
                "--old-key", _key_text(_OLD_KEY),
                "--new-key", _key_text(_NEW_KEY),
            )
            self.assertEqual(code, 0)
            self.assertIn("REKEY OK", out)
            self.assertIn("rows re-encrypted (tasks + events)  : 2", out)
            self.assertIn("pending payload re-encrypted        : 1", out)
            self.assertIn("RAG cache files deleted             : 4", out)
            self.assertIn("AI_STORAGE_KEY", out)
            self.assertEqual(err, "")

            for name in self.script.CACHE_ARTIFACTS:
                self.assertFalse((state / "Cache" / name).exists())

            rotated = SqliteTaskStore(state / "tasks.sqlite", cipher=_cipher(_NEW_KEY))
            try:
                self.assertEqual(
                    [task.message for task in rotated.list()],
                    ["sensitive prompt"],
                )
            finally:
                rotated.close()
            pending = PendingActionStore(
                state / "pending_agent_actions.json", cipher=_cipher(_NEW_KEY),
            )
            pending.load()
            self.assertEqual(pending["act-1"]["tool"], "write_file")

    def test_a_malformed_key_aborts_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self._workspace(Path(directory))
            pending_before = (state / "pending_agent_actions.json").read_bytes()

            code, _, err = self._run(
                "--data-dir", str(state),
                "--old-key", _key_text(_OLD_KEY),
                "--new-key", "hex:not-hex",
            )
            self.assertEqual(code, 1)
            self.assertIn("REKEY FAIL", err)
            self.assertEqual((state / "pending_agent_actions.json").read_bytes(), pending_before)

            code, _, err = self._run(
                "--data-dir", str(state),
                "--old-key", _key_text(_OLD_KEY),
            )
            self.assertEqual(code, 1)
            self.assertIn("needs a new key", err)

    def test_a_wrong_old_key_aborts_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self._workspace(Path(directory))
            pending_before = (state / "pending_agent_actions.json").read_bytes()

            code, _, err = self._run(
                "--data-dir", str(state),
                "--old-key", _key_text(_THIRD_KEY),
                "--new-key", _key_text(_NEW_KEY),
            )
            self.assertEqual(code, 1)
            self.assertIn("REKEY FAIL", err)
            self.assertIn("was sealed with another key", err)
            self.assertEqual((state / "pending_agent_actions.json").read_bytes(), pending_before)

    def test_a_missing_state_aborts(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "empty"
            empty.mkdir()
            code, _, err = self._run(
                "--data-dir", str(empty),
                "--old-key", _key_text(_OLD_KEY),
                "--new-key", _key_text(_NEW_KEY),
            )
            self.assertEqual(code, 1)
            self.assertIn("no local state found", err)

    def test_a_corrupt_task_store_aborts_with_a_clear_message(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            state.mkdir()
            (state / "tasks.sqlite").write_bytes(b"not a database")
            code, _, err = self._run(
                "--data-dir", str(state),
                "--old-key", _key_text(_OLD_KEY),
                "--verify",
            )
            self.assertEqual(code, 1)
            self.assertIn("the task store could not be read", err)

    def test_the_cache_artifact_names_match_the_application_paths(self):
        paths = resolve_paths(Path(__file__).resolve().parents[2] / "src")
        self.assertEqual(
            tuple(path.name for path in (
                paths.cache_index, paths.cache_chunks, paths.cache_bm25, paths.cache_metadata,
            )),
            self.script.CACHE_ARTIFACTS,
        )


if __name__ == "__main__":
    unittest.main()
