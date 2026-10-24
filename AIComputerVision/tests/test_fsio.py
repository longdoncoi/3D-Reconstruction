"""Unit tests for the shared filesystem primitives."""

from __future__ import annotations

import os
import socket
import tempfile
import time
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

from cvtrain import fsio as fsio_mod
from cvtrain.errors import ConcurrentRunError, ExitCode, StorageError
from cvtrain.fsio import (
    BACKUP_SUFFIX,
    LOCK_NAME,
    RUN_ALIVE_SECONDS,
    RUN_MARKER_NAME,
    atomic_copy,
    atomic_replace_with_backup,
    atomic_write_text,
    claim_run_marker,
    clear_run_marker,
    directory_lock,
    file_size,
    prune_run_directories,
    remove_tree,
    sha256_file,
    touch_lock,
    touch_run_marker,
)


def _leftovers(directory: Path) -> list[str]:
    return [p.name for p in directory.iterdir() if p.name.endswith(".tmp")]


class Sha256Tests(unittest.TestCase):
    def test_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.bin"
            path.write_bytes(b"")
            self.assertEqual(sha256_file(path), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

    def test_content_change_changes_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob.bin"
            path.write_bytes(b"a")
            first = sha256_file(path)
            path.write_bytes(b"b")
            self.assertNotEqual(sha256_file(path), first)

    def test_an_unreadable_file_is_a_typed_exit_2_not_a_traceback(self) -> None:
        # Provenance and verification depend on this digest: "cannot vouch"
        # (exit 2, one actionable line) instead of an OSError traceback.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StorageError) as ctx:
                sha256_file(Path(tmp) / "missing.bin")
        self.assertEqual(ctx.exception.exit_code, ExitCode.PREFLIGHT)
        self.assertIn("missing.bin", str(ctx.exception))


class FileSizeTests(unittest.TestCase):
    def test_reports_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.onnx"
            path.write_bytes(b"12345")
            self.assertEqual(file_size(path), 5)

    def test_an_unmeasurable_file_is_typed_like_an_unhashable_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StorageError) as ctx:
                file_size(Path(tmp) / "missing.onnx")
        self.assertEqual(ctx.exception.exit_code, ExitCode.PREFLIGHT)


class DirectoryFsyncTests(unittest.TestCase):
    """The directory-entry fsync behind every atomic publish is best-effort."""

    def test_an_unopenable_directory_skips_the_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("cvtrain.fsio.os.open", side_effect=OSError("cannot open dirs here")):
                fsio_mod._fsync_dir(Path(tmp))

    def test_a_failed_directory_fsync_still_closes_the_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch("cvtrain.fsio.os.open", return_value=123),
                mock.patch("cvtrain.fsio.os.fsync", side_effect=OSError("read-only")),
                mock.patch("cvtrain.fsio.os.close") as close,
            ):
                fsio_mod._fsync_dir(Path(tmp))
            close.assert_called_once_with(123)


class AtomicWriteTextTests(unittest.TestCase):
    def test_creates_missing_parents_and_leaves_no_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a" / "b" / "manifest.json"
            atomic_write_text(target, '{"ok": true}\n')
            self.assertEqual(target.read_text(encoding="utf-8"), '{"ok": true}\n')
            self.assertEqual(_leftovers(target.parent), [])

    def test_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "f.txt"
            target.write_text("old", encoding="utf-8")
            atomic_write_text(target, "new")
            self.assertEqual(target.read_text(encoding="utf-8"), "new")

    def test_forces_lf_newlines(self) -> None:
        # The manifest is diffed by Git; CRLF would churn every Windows clone.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "f.txt"
            atomic_write_text(target, "one\ntwo\n")
            self.assertEqual(target.read_bytes(), b"one\ntwo\n")

    def test_failed_publish_cleans_up_and_keeps_the_old_file(self) -> None:
        # A full disk is an environment fact: typed exit 2, no traceback, and
        # the previous manifest still on disk must survive the failed attempt.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "f.txt"
            target.write_bytes(b"previous")

            with mock.patch("cvtrain.fsio.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(StorageError) as ctx:
                    atomic_write_text(target, "new")

            self.assertEqual(target.read_bytes(), b"previous")
            self.assertEqual(_leftovers(root), [])
            self.assertIn("disk full", str(ctx.exception))

    def test_a_directory_that_cannot_be_staged_is_typed_too(self) -> None:
        # Same verdict for the step *before* the rename: a read-only Models/
        # is not a bug in the pipeline, it is storage this run cannot write.
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("cvtrain.fsio.tempfile.mkstemp", side_effect=OSError("permission denied")):
                with self.assertRaises(StorageError) as ctx:
                    atomic_write_text(Path(tmp) / "f.txt", "new")
        self.assertIn("permission denied", str(ctx.exception))


class AtomicCopyTests(unittest.TestCase):
    def test_copies_bytes_and_replaces_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "generated.onnx"
            dst = root / "Models" / "yolo11n.onnx"
            src.write_bytes(b"model-v1")
            atomic_copy(src, dst)
            self.assertEqual(dst.read_bytes(), b"model-v1")

            src.write_bytes(b"model-v2-longer")
            atomic_copy(src, dst)
            self.assertEqual(dst.read_bytes(), b"model-v2-longer")
            self.assertEqual(_leftovers(dst.parent), [])

    def test_failed_copy_leaves_previous_artifact_intact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dst = root / "out.onnx"
            dst.write_bytes(b"previous-good-artifact")

            missing = root / "does-not-exist.onnx"
            with self.assertRaises(OSError):
                atomic_copy(missing, dst)

            # The publish target must never be truncated by a failed attempt.
            self.assertEqual(dst.read_bytes(), b"previous-good-artifact")
            self.assertEqual(_leftovers(root), [])

    def test_a_write_failure_is_typed_exit_2_not_a_traceback(self) -> None:
        # A rename the OS refuses (read-only Models/, a Windows sharing
        # violation) is storage the run cannot use, not a pipeline bug: the
        # operator gets one actionable line and exit 2, and the published
        # artifact is left as it was.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "generated.onnx"
            dst = root / "out.onnx"
            src.write_bytes(b"model-v2")
            dst.write_bytes(b"previous-good-artifact")

            with mock.patch("cvtrain.fsio.os.replace", side_effect=OSError("sharing violation")):
                with self.assertRaises(StorageError) as ctx:
                    atomic_copy(src, dst)

            self.assertIn("sharing violation", str(ctx.exception))
            self.assertEqual(dst.read_bytes(), b"previous-good-artifact")
            self.assertEqual(_leftovers(root), [])


class DiscardTempTests(unittest.TestCase):
    """Cleanup must never replace the error it is cleaning up after."""

    def test_a_temp_file_that_cannot_be_removed_never_masks_the_real_error(self) -> None:
        # ``finally`` runs while the StorageError is on its way out: a second
        # OSError raised there would supplant it and turn an exit-2 verdict
        # back into a traceback.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "generated.onnx"
            src.write_bytes(b"model")

            with (
                mock.patch("cvtrain.fsio.os.replace", side_effect=OSError("disk full")),
                mock.patch("cvtrain.fsio.os.unlink", side_effect=OSError("locked")),
                self.assertLogs("cvtrain.fsio", level="WARNING") as captured,
            ):
                with self.assertRaises(StorageError) as write:
                    atomic_write_text(root / "manifest.json", "{}")
                with self.assertRaises(StorageError) as copy:
                    atomic_copy(src, root / "out.onnx")

            # The typed verdict survives the failed cleanup — both writers.
            self.assertIn("disk full", str(write.exception))
            self.assertIn("disk full", str(copy.exception))
            # ...and each leftover temp file is reported instead of raising.
            warnings = [line for line in captured.output if "Could not remove" in line]
            self.assertEqual(len(warnings), 2)


class AtomicReplaceWithBackupTests(unittest.TestCase):
    """Publishing keeps one generation of the artifact it replaces."""

    def test_first_publish_leaves_no_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "generated.onnx"
            dst = root / "Models" / "yolo11n.onnx"
            src.write_bytes(b"model-v1")

            atomic_replace_with_backup(src, dst)

            self.assertEqual(dst.read_bytes(), b"model-v1")
            self.assertFalse(dst.with_name(dst.name + BACKUP_SUFFIX).exists())

    def test_replacing_keeps_the_previous_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "generated.onnx"
            dst = root / "Models" / "yolo11n.onnx"
            backup = dst.with_name(dst.name + BACKUP_SUFFIX)
            src.write_bytes(b"model-v1")
            atomic_replace_with_backup(src, dst)

            src.write_bytes(b"model-v2")
            atomic_replace_with_backup(src, dst)

            self.assertEqual(dst.read_bytes(), b"model-v2")
            self.assertEqual(backup.read_bytes(), b"model-v1")
            self.assertEqual(_leftovers(dst.parent), [])

    def test_a_failed_backup_publish_leaves_the_artifact_intact(self) -> None:
        # The backup is taken before the replace: if it cannot be written the
        # published model is never touched, so a crash cannot cost us both.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "generated.onnx"
            dst = root / "out.onnx"
            dst.write_bytes(b"previous-good-artifact")

            with mock.patch("cvtrain.fsio.atomic_copy", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    atomic_replace_with_backup(src, dst)

            self.assertEqual(dst.read_bytes(), b"previous-good-artifact")


class DirectoryLockTests(unittest.TestCase):
    """The advisory lock that keeps two runs out of one ``Models/`` directory."""

    def test_creates_the_directory_and_releases_on_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            with directory_lock(target) as lock:
                self.assertTrue(lock.exists())
                self.assertEqual(lock.name, LOCK_NAME)
            self.assertFalse(lock.exists())

    def test_a_directory_that_cannot_be_created_is_typed(self) -> None:
        # Locking needs to *make* Models/ first; a permission there is an
        # environment fault like any other write this package performs.
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
                with self.assertRaises(StorageError) as ctx:
                    with directory_lock(Path(tmp) / "Models"):
                        pass  # pragma: no cover - the lock is never acquired
        self.assertIn("permission denied", str(ctx.exception))

    def test_a_lock_that_cannot_be_created_is_typed(self) -> None:
        # Beyond mkdir: the O_EXCL create itself can be refused (read-only
        # volume, full disk). Same verdict — exit 2, one line, no traceback.
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("cvtrain.fsio.os.open", side_effect=OSError("read-only volume")):
                with self.assertRaises(StorageError) as ctx:
                    with directory_lock(Path(tmp) / "Models"):
                        pass  # pragma: no cover - the lock is never acquired
        self.assertIn("read-only volume", str(ctx.exception))

    def test_a_lock_whose_token_cannot_be_written_is_taken_back(self) -> None:
        # Release only deletes a lock that still carries our token, so a lock
        # file left behind by a failed write would block every later run until
        # the stale window expired. Remove it, then report.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            with mock.patch("cvtrain.fsio.os.fdopen", side_effect=OSError("disk full")):
                with self.assertRaises(StorageError) as ctx:
                    with directory_lock(target):
                        pass  # pragma: no cover - the lock is never acquired
            self.assertIn("disk full", str(ctx.exception))
            self.assertFalse((target / LOCK_NAME).exists())

    def test_a_lock_that_cannot_be_deleted_warns_instead_of_failing_the_run(self) -> None:
        # Cleanup runs while the run's own work is being reported: a failure
        # here must never become the run's verdict.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            with mock.patch("cvtrain.fsio.os.unlink", side_effect=OSError("locked by another process")):
                with self.assertLogs("cvtrain.fsio", level="WARNING") as captured:
                    with directory_lock(target) as lock:
                        self.assertTrue(lock.exists())
        self.assertTrue(any("Could not remove" in line for line in captured.output))

    def test_releases_the_lock_when_the_body_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            with self.assertRaises(RuntimeError):
                with directory_lock(target) as lock:
                    raise RuntimeError("boom")
            self.assertFalse(lock.exists())

    def test_release_tolerates_an_already_removed_lock(self) -> None:
        # Something outside the helper deleted the file: exiting must not raise.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            with directory_lock(target) as lock:
                lock.unlink()
            self.assertFalse(lock.exists())

    def test_release_leaves_a_foreign_lock_alone(self) -> None:
        # A steal that landed while the body ran (possible past stale_after):
        # exiting must not unlock the new owner's run.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            with self.assertLogs("cvtrain.fsio", level="WARNING") as captured:
                with directory_lock(target) as lock:
                    lock.write_text("owner pid=999999 token=foreign acquired=now\n", encoding="utf-8")
            self.assertTrue(lock.exists())
            self.assertIn("foreign", lock.read_text(encoding="utf-8"))
            self.assertTrue(any("changed hands" in line for line in captured.output))

    def test_release_warns_when_ownership_cannot_be_verified(self) -> None:
        # An unreadable lock file must fail closed: leave it, warn, no raise.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            with mock.patch.object(Path, "read_text", side_effect=OSError("denied")):
                with self.assertLogs("cvtrain.fsio", level="WARNING") as captured:
                    with directory_lock(target) as lock:
                        pass
            self.assertTrue(lock.exists())
            self.assertTrue(any("leaving it in place" in line for line in captured.output))

    def test_a_held_lock_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            target.mkdir()
            (target / LOCK_NAME).write_text("held", encoding="utf-8")

            with self.assertRaises(ConcurrentRunError) as ctx:
                with directory_lock(target):
                    pass  # pragma: no cover - the lock is never acquired

            self.assertIn(str(target / LOCK_NAME), str(ctx.exception))

    def test_a_stale_lock_is_stolen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            target.mkdir()
            lock = target / LOCK_NAME
            lock.write_text("stale", encoding="utf-8")
            old = time.time() - 10
            os.utime(lock, (old, old))

            with self.assertLogs("cvtrain.fsio", level="WARNING") as captured:
                with directory_lock(target, stale_after=1):
                    self.assertTrue(lock.exists())

            self.assertTrue(any("Stealing stale lock" in line for line in captured.output))

    def test_a_lock_whose_age_cannot_be_read_is_refused(self) -> None:
        # stat() failing between the failed create and the age check means the
        # holder vanished: fail closed rather than assume the lock is free.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            target.mkdir()
            (target / LOCK_NAME).write_text("held", encoding="utf-8")

            with mock.patch("cvtrain.fsio._lock_age", return_value=None):
                with self.assertRaises(ConcurrentRunError):
                    with directory_lock(target):
                        pass  # pragma: no cover - the lock is never acquired

    def test_lock_age_is_none_when_the_file_is_gone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(fsio_mod._lock_age(Path(tmp) / LOCK_NAME))

    def test_a_lock_whose_owner_is_dead_is_stolen_before_the_lease_expires(self) -> None:
        # A crash between writing the lock and starting the heartbeat leaves a
        # fresh file. Waiting 24 hours for it is unnecessary: the owner named in
        # the lock is provably gone.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            target.mkdir()
            lock = target / LOCK_NAME
            lock.write_text(
                f"owner pid=4242 host={socket.gethostname()} token=dead acquired=now\n",
                encoding="utf-8",
            )

            with mock.patch("cvtrain.fsio._process_alive", return_value=False):
                with self.assertLogs("cvtrain.fsio", level="WARNING") as captured:
                    with directory_lock(target, stale_after=10_000):
                        self.assertTrue(lock.exists())
            self.assertTrue(any("Stealing stale lock" in line for line in captured.output))

    def test_a_live_owner_is_never_stolen_from_even_with_an_expired_lease(self) -> None:
        # The safe direction for a destructive-ish operation is to keep the
        # lock: a suspended process or a slow volume must not let a second
        # writer into Models/.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            target.mkdir()
            lock = target / LOCK_NAME
            lock.write_text(
                f"owner pid=4242 host={socket.gethostname()} token=live acquired=now\n",
                encoding="utf-8",
            )
            old = time.time() - 10_000
            os.utime(lock, (old, old))

            with mock.patch("cvtrain.fsio._process_alive", return_value=True):
                with self.assertRaises(ConcurrentRunError):
                    with directory_lock(target, stale_after=1):
                        pass  # pragma: no cover - the lock is never acquired

    def test_the_process_probe_sees_this_process(self) -> None:
        self.assertIs(fsio_mod._process_alive(os.getpid()), True)

    def test_the_process_probe_is_the_one_this_platform_can_run(self) -> None:
        # ``os.kill(pid, 0)`` terminates a process on Windows, so the choice
        # cannot be a runtime branch that both sides share: each platform gets
        # exactly the probe whose semantics it implements.
        expected = fsio_mod._windows_process_alive if os.name == "nt" else fsio_mod._posix_process_alive
        self.assertIs(fsio_mod._probe_process, expected)

    def test_an_invalid_pid_is_unknown_not_dead(self) -> None:
        # Deleting another run's lock on an unprovable answer is the unsafe
        # direction; only a real positive PID is ever probed.
        for pid in (0, -1, True, "4242", 3.5, None):
            with self.subTest(pid=pid):
                self.assertIsNone(fsio_mod._process_alive(pid))  # type: ignore[arg-type]

    def test_a_malformed_pid_token_yields_no_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / LOCK_NAME
            path.write_text("owner pid=not-a-number host=here token=x\n", encoding="utf-8")
            self.assertEqual(fsio_mod._read_owner(path), (None, "here"))

    def test_an_unreadable_owner_file_yields_no_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(fsio_mod._read_owner(Path(tmp) / LOCK_NAME), (None, None))

    def test_an_unknown_owner_falls_back_to_the_lease_age(self) -> None:
        # A probe that cannot answer (access denied, another host, an older
        # lock) must not be read as "dead": the age rule still decides.
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / LOCK_NAME
            lock.write_text(f"owner pid=4242 host={socket.gethostname()} token=x\n", encoding="utf-8")
            with mock.patch("cvtrain.fsio._process_alive", return_value=None):
                self.assertFalse(fsio_mod._is_stale(lock, age=1.0, stale_after=100.0))
                self.assertTrue(fsio_mod._is_stale(lock, age=1_000.0, stale_after=100.0))

    def test_a_held_lock_keeps_its_lease_fresh_in_the_background(self) -> None:
        # The heartbeat is the lease: a live run must not look abandoned during
        # a long epoch that never reaches the epoch callback.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            with directory_lock(target, heartbeat_interval=0.05) as lock:
                before = lock.stat().st_mtime
                time.sleep(0.2)
                after = lock.stat().st_mtime
            self.assertGreater(after, before)

    def test_an_unstealable_lock_is_refused_after_retrying(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            target.mkdir()
            (target / LOCK_NAME).write_text("held", encoding="utf-8")

            with mock.patch("cvtrain.fsio._lock_age", return_value=10_000.0), mock.patch(
                "cvtrain.fsio.os.open", side_effect=FileExistsError
            ):
                with self.assertRaises(ConcurrentRunError):
                    with directory_lock(target, stale_after=1):
                        pass  # pragma: no cover - the lock is never acquired

    def _stale_lock(self, target: Path) -> Path:
        lock = target / LOCK_NAME
        lock.write_text("stale", encoding="utf-8")
        old = time.time() - 10
        os.utime(lock, (old, old))
        return lock

    def test_a_fresh_steal_claim_blocks_a_second_stealer(self) -> None:
        # Two processes can both observe a stale lock. The sidecar claim makes
        # the second one refuse instead of deleting the lock the first one is
        # about to replace.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            target.mkdir()
            lock = self._stale_lock(target)
            claim = target / (LOCK_NAME + fsio_mod.STEAL_SUFFIX)
            claim.write_text("another stealer", encoding="utf-8")

            with self.assertRaises(ConcurrentRunError):
                with directory_lock(target, stale_after=1):
                    pass  # pragma: no cover - the lock is never acquired
            self.assertTrue(lock.exists())

    def test_an_abandoned_steal_claim_is_replaced(self) -> None:
        # A stealer that crashed after taking the claim must not wedge every
        # future run: the claim ages out on the same principle as the lock.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            target.mkdir()
            lock = self._stale_lock(target)
            claim = target / (LOCK_NAME + fsio_mod.STEAL_SUFFIX)
            claim.write_text("crashed stealer", encoding="utf-8")
            old = time.time() - 10_000
            os.utime(claim, (old, old))

            with self.assertLogs("cvtrain.fsio", level="WARNING") as captured:
                with directory_lock(target, stale_after=1):
                    self.assertTrue(lock.exists())
            self.assertTrue(any("stale lock-steal claim" in line for line in captured.output))
            self.assertFalse(claim.exists())

    def test_a_claim_that_cannot_be_created_is_typed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            target.mkdir()
            self._stale_lock(target)
            real_open = os.open

            def _open(path: str | Path, flags: int) -> int:
                if str(path).endswith(fsio_mod.STEAL_SUFFIX):
                    raise OSError("claim denied")
                return real_open(path, flags)

            with mock.patch("cvtrain.fsio.os.open", side_effect=_open):
                with self.assertRaises(StorageError) as ctx:
                    with directory_lock(target, stale_after=1):
                        pass  # pragma: no cover - the lock is never acquired
            self.assertIn("claim denied", str(ctx.exception))

    def test_a_claim_whose_token_cannot_be_written_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            target.mkdir()
            lock = self._stale_lock(target)
            with mock.patch("cvtrain.fsio.os.fdopen", side_effect=OSError("disk full")):
                with self.assertRaises(StorageError):
                    with directory_lock(target, stale_after=1):
                        pass  # pragma: no cover - the lock is never acquired
            self.assertFalse((target / (LOCK_NAME + fsio_mod.STEAL_SUFFIX)).exists())
            self.assertTrue(lock.exists())

    def test_a_lock_that_turns_fresh_while_stealing_is_not_deleted(self) -> None:
        # The claim closes the window between observing a stale lock and
        # deleting it: if the lock changed hands, the recheck sees a fresh age.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            target.mkdir()
            lock = self._stale_lock(target)
            with mock.patch("cvtrain.fsio._lock_age", side_effect=[10_000.0, 0.0]):
                with self.assertRaises(ConcurrentRunError):
                    with directory_lock(target, stale_after=1):
                        pass  # pragma: no cover - the lock is never acquired
            self.assertTrue(lock.exists())
            self.assertFalse((target / (LOCK_NAME + fsio_mod.STEAL_SUFFIX)).exists())

    def test_a_lock_that_survives_two_steal_attempts_is_refused(self) -> None:
        # Defensive exit: both attempts found the same stale lock and could not
        # remove it. The run must be refused, never proceed without ownership.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            target.mkdir()
            lock = self._stale_lock(target)
            real_open = os.open
            real_unlink = fsio_mod._unlink_quietly

            def _open(path: str | Path, flags: int) -> int:
                if str(path).endswith(LOCK_NAME):
                    raise FileExistsError
                return real_open(path, flags)

            def _unlink(path: Path) -> None:
                if path.name == LOCK_NAME:
                    return
                real_unlink(path)

            with (
                mock.patch("cvtrain.fsio.os.open", side_effect=_open),
                mock.patch("cvtrain.fsio._unlink_quietly", side_effect=_unlink),
            ):
                with self.assertRaises(ConcurrentRunError):
                    with directory_lock(target, stale_after=1):
                        pass  # pragma: no cover - the lock is never acquired
            self.assertTrue(lock.exists())

    def test_a_lock_that_vanishes_while_stealing_is_recreated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Models"
            target.mkdir()
            lock = self._stale_lock(target)
            calls = 0

            def _age(path: Path) -> float | None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return 10_000.0
                path.unlink()
                return None

            with mock.patch("cvtrain.fsio._lock_age", side_effect=_age):
                with directory_lock(target, stale_after=1) as acquired:
                    self.assertTrue(acquired.exists())
            self.assertFalse(lock.exists())


class RemoveTreeTests(unittest.TestCase):
    """The one destructive primitive: a refused deletion is a typed error."""

    def test_a_tree_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            (root / "nested").mkdir(parents=True)
            (root / "nested" / "file.bin").write_bytes(b"payload")

            remove_tree(root)

            self.assertFalse(root.exists())

    def test_a_removal_the_os_refuses_is_a_typed_exit_2_not_a_traceback(self) -> None:
        # A read-only volume or a Windows sharing violation is an environment
        # fact: exit 2, one actionable line, never an OSError traceback.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            root.mkdir()
            with mock.patch("cvtrain.fsio.shutil.rmtree", side_effect=OSError("denied")):
                with self.assertRaises(StorageError) as ctx:
                    remove_tree(root)
        self.assertIn("Cannot remove", str(ctx.exception))


class PruneRunDirectoriesTests(unittest.TestCase):
    """``runs/`` grows without bound; the pruner is the retention policy."""

    def _make_run(self, model_dir: Path, name: str, *, age: float, checkpoint_age: float | None = None) -> Path:
        """Create a run directory whose activity time is ``age`` seconds old.

        ``checkpoint_age`` writes the ``weights/last.pt`` the framework
        refreshes every epoch — the file whose freshness proves a run is
        alive — and keeps the directory's own mtime separate from it.
        """

        run_dir = model_dir / name
        run_dir.mkdir(parents=True)
        if checkpoint_age is not None:
            weights = run_dir / "weights"
            weights.mkdir()
            checkpoint = weights / "last.pt"
            checkpoint.write_bytes(b"weights")
            stamp = time.time() - checkpoint_age
            os.utime(checkpoint, (stamp, stamp))
        stamp = time.time() - age
        os.utime(run_dir, (stamp, stamp))
        return run_dir

    def test_the_newest_run_survives_and_the_older_ones_are_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            model_dir = runs_dir / "detect"
            model_dir.mkdir(parents=True)
            oldest = self._make_run(model_dir, "train-oldest", age=5 * RUN_ALIVE_SECONDS)
            middle = self._make_run(model_dir, "train-middle", age=4 * RUN_ALIVE_SECONDS)
            newest = self._make_run(model_dir, "train-newest", age=3 * RUN_ALIVE_SECONDS)

            deleted = prune_run_directories(runs_dir, keep=1, prefix="train-")

            self.assertEqual(set(deleted), {oldest, middle})
            self.assertFalse(oldest.exists())
            self.assertFalse(middle.exists())
            self.assertTrue(newest.exists())

    def test_a_run_that_is_still_alive_is_never_deleted(self) -> None:
        # The framework rewrites ``weights/last.pt`` after every epoch, so a
        # fresh checkpoint is the proof that a run is still going — whatever
        # its position in the history. Deleting it would destroy the very run
        # ``--resume`` would continue from.
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            model_dir = runs_dir / "detect"
            model_dir.mkdir(parents=True)
            live = self._make_run(model_dir, "train-live", age=10 * RUN_ALIVE_SECONDS, checkpoint_age=10)
            dead = self._make_run(model_dir, "train-dead", age=10 * RUN_ALIVE_SECONDS)

            deleted = prune_run_directories(runs_dir, keep=0, prefix="train-")

            self.assertEqual(deleted, [dead])
            self.assertTrue(live.exists())

    def test_a_run_whose_owner_is_alive_is_kept_even_without_recent_writes(self) -> None:
        # A long epoch may not touch ``last.pt`` for a while. The run's lease
        # names the process that is writing it, so liveness does not depend on
        # one timestamp.
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            model_dir = runs_dir / "detect"
            model_dir.mkdir(parents=True)
            live = self._make_run(model_dir, "train-live", age=10 * RUN_ALIVE_SECONDS)
            touch_run_marker(live)
            old = time.time() - 10 * RUN_ALIVE_SECONDS
            for path in (live, live / RUN_MARKER_NAME):
                os.utime(path, (old, old))

            deleted = prune_run_directories(runs_dir, keep=0, prefix="train-")

            self.assertEqual(deleted, [])
            self.assertTrue(live.exists())

    def test_a_marker_from_another_host_still_counts_as_recent_activity(self) -> None:
        # A shared volume can be written by another machine: its PID cannot be
        # probed here, so the marker's own freshness is the remaining evidence
        # that the run is alive. The lease age decides.
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            model_dir = runs_dir / "detect"
            model_dir.mkdir(parents=True)
            remote = self._make_run(model_dir, "train-remote", age=10 * RUN_ALIVE_SECONDS)
            marker = remote / RUN_MARKER_NAME
            marker.write_text("owner pid=4242 host=some-other-host\n", encoding="utf-8")
            old = time.time() - 10 * RUN_ALIVE_SECONDS
            os.utime(remote, (old, old))

            deleted = prune_run_directories(runs_dir, keep=0, prefix="train-")

            self.assertEqual(deleted, [])
            self.assertTrue(remote.exists())

    def test_a_run_whose_owner_is_dead_is_pruned_when_old(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            model_dir = runs_dir / "detect"
            model_dir.mkdir(parents=True)
            dead = self._make_run(model_dir, "train-dead", age=10 * RUN_ALIVE_SECONDS)
            touch_run_marker(dead)
            old = time.time() - 10 * RUN_ALIVE_SECONDS
            # Creating the marker changes the directory's own mtime, so age it
            # again: the test is about an old run whose owner is gone, not
            # about an old marker in a fresh directory.
            for path in (dead, dead / RUN_MARKER_NAME):
                os.utime(path, (old, old))

            with mock.patch("cvtrain.fsio._process_alive", return_value=False):
                deleted = prune_run_directories(runs_dir, keep=0, prefix="train-")

            self.assertEqual(deleted, [dead])
            self.assertFalse(dead.exists())

    def test_a_directory_that_is_not_a_run_is_never_deleted(self) -> None:
        # Only this pipeline's own run directories are ours to remove.
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            model_dir = runs_dir / "detect"
            model_dir.mkdir(parents=True)
            foreign = self._make_run(model_dir, "not-a-run", age=10 * RUN_ALIVE_SECONDS)

            deleted = prune_run_directories(runs_dir, keep=0, prefix="train-")

            self.assertEqual(deleted, [])
            self.assertTrue(foreign.exists())

    def test_every_model_keeps_its_own_newest_run(self) -> None:
        # The rule is per model, not per directory: pruning the tracking runs
        # must not cost the detection runs their checkpoint.
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            for model in ("detect", "tracking"):
                model_dir = runs_dir / model
                model_dir.mkdir(parents=True)
                self._make_run(model_dir, "train-old", age=5 * RUN_ALIVE_SECONDS)
                self._make_run(model_dir, "train-new", age=4 * RUN_ALIVE_SECONDS)

            prune_run_directories(runs_dir, keep=1, prefix="train-")

            for model in ("detect", "tracking"):
                with self.subTest(model=model):
                    self.assertEqual([p.name for p in (runs_dir / model).iterdir()], ["train-new"])

    def test_two_runs_written_in_the_same_second_are_ordered_deterministically(self) -> None:
        # The name embeds the creation timestamp, so it breaks the tie the
        # modification time cannot: without it, which run survives would
        # depend on the order the filesystem happened to list them in. Both
        # runs are given *exactly* the same modification time here, so only
        # the name can decide.
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            model_dir = runs_dir / "detect"
            model_dir.mkdir(parents=True)
            older = self._make_run(model_dir, "train-20260101-000000-1", age=5 * RUN_ALIVE_SECONDS)
            newer = self._make_run(model_dir, "train-20260102-000000-2", age=5 * RUN_ALIVE_SECONDS)
            same = time.time() - 5 * RUN_ALIVE_SECONDS
            os.utime(older, (same, same))
            os.utime(newer, (same, same))

            deleted = prune_run_directories(runs_dir, keep=1, prefix="train-")

            self.assertEqual(deleted, [older])
            self.assertTrue(newer.exists())

    def test_an_absent_runs_directory_prunes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            deleted = prune_run_directories(Path(tmp) / "never-used", keep=1, prefix="train-")
        self.assertEqual(deleted, [])

    def test_a_negative_count_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                prune_run_directories(Path(tmp), keep=-1, prefix="train-")

    def test_an_unreadable_runs_directory_is_a_typed_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            runs_dir.mkdir()
            with mock.patch("pathlib.Path.iterdir", side_effect=OSError("denied")):
                with self.assertRaises(StorageError) as ctx:
                    prune_run_directories(runs_dir, keep=1, prefix="train-")
        self.assertIn("Cannot read", str(ctx.exception))

    def test_an_unreadable_model_directory_is_a_typed_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            (runs_dir / "detect").mkdir(parents=True)
            real_iterdir = Path.iterdir

            def _fail_on_model(self: Path) -> Iterator[Path]:
                if self.name == "detect":
                    raise OSError("denied")
                return real_iterdir(self)

            with mock.patch("pathlib.Path.iterdir", _fail_on_model):
                with self.assertRaises(StorageError) as ctx:
                    prune_run_directories(runs_dir, keep=1, prefix="train-")
        self.assertIn("Cannot read", str(ctx.exception))

    def test_an_unstatable_run_is_treated_as_alive(self) -> None:
        # Unknown age is not evidence of abandonment: deleting the only copy
        # of a run whose age could not be proved is the unsafe direction, so
        # an unreadable run sorts as "active now" and survives this sweep.
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            model_dir = runs_dir / "detect"
            model_dir.mkdir(parents=True)
            run_dir = self._make_run(model_dir, "train-x", age=5 * RUN_ALIVE_SECONDS)
            real_stat = Path.stat

            def _fail_on_run(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
                if self == run_dir:
                    raise OSError("denied")
                return real_stat(self, follow_symlinks=follow_symlinks)

            before = time.time()
            with mock.patch("pathlib.Path.stat", _fail_on_run):
                activity = fsio_mod._run_activity_mtime(run_dir)
        self.assertGreaterEqual(activity, before)


class RunMarkerTests(unittest.TestCase):
    """The run lease: process liveness for ``--prune-runs``."""

    def test_creates_the_marker_with_the_owner_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "train-x"
            touch_run_marker(run_dir)
            text = (run_dir / RUN_MARKER_NAME).read_text(encoding="utf-8")
            self.assertIn(f"pid={os.getpid()}", text)
            self.assertIn(f"host={socket.gethostname()}", text)

    def test_refreshes_the_marker_instead_of_rewriting_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "train-x"
            run_dir.mkdir()
            marker = run_dir / RUN_MARKER_NAME
            marker.write_text("owner pid=1 host=other\n", encoding="utf-8")
            old = time.time() - 10_000
            os.utime(marker, (old, old))

            touch_run_marker(run_dir)

            self.assertGreater(marker.stat().st_mtime, old + 5_000)
            # The owner line is preserved: refreshing a lease is not taking it.
            # Compare the whole document: ``pid=1`` would also match a rewritten
            # marker whose owner PID happens to start with ``1`` (19232 does),
            # which is exactly the survivor this assertion exists to kill.
            self.assertEqual(marker.read_text(encoding="utf-8"), "owner pid=1 host=other\n")

    def test_claiming_replaces_a_dead_owners_lease(self) -> None:
        # A resumed run reuses the interrupted run's directory. Refreshing the
        # old marker would keep the killed PID, so --prune-runs would read a
        # live resumed run as abandoned for its whole first epoch.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "train-x"
            run_dir.mkdir()
            marker = run_dir / RUN_MARKER_NAME
            marker.write_text("owner pid=1 host=other-host started=2026-01-01T00:00:00+00:00\n", encoding="utf-8")
            old = time.time() - 10_000
            os.utime(marker, (old, old))

            claim_run_marker(run_dir)

            text = marker.read_text(encoding="utf-8")
            self.assertIn(f"pid={os.getpid()}", text)
            self.assertIn(f"host={socket.gethostname()}", text)
            self.assertGreater(marker.stat().st_mtime, old + 5_000)

    def test_claiming_creates_the_run_directory_and_its_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "detect" / "train-x"
            claim_run_marker(run_dir)
            self.assertTrue((run_dir / RUN_MARKER_NAME).is_file())

    def test_an_unwritable_claim_warns_instead_of_failing_the_run(self) -> None:
        # Best effort like the epoch refresh: the run already owns the
        # directory, so a lease that cannot be written must not crash it.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "train-x"
            with mock.patch.object(Path, "write_text", side_effect=OSError("denied")):
                with self.assertLogs("cvtrain.fsio", level="WARNING") as captured:
                    claim_run_marker(run_dir)
        self.assertTrue(any("Could not claim run marker" in line for line in captured.output))

    def test_clear_removes_the_marker_and_tolerates_its_absence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "train-x"
            touch_run_marker(run_dir)
            clear_run_marker(run_dir)
            self.assertFalse((run_dir / RUN_MARKER_NAME).exists())
            clear_run_marker(run_dir)  # idempotent

    def test_an_unwritable_marker_warns_instead_of_failing_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "train-x"
            with mock.patch.object(Path, "write_text", side_effect=OSError("denied")):
                with self.assertLogs("cvtrain.fsio", level="WARNING") as captured:
                    touch_run_marker(run_dir)
        self.assertTrue(any("Could not refresh run marker" in line for line in captured.output))


class TouchLockTests(unittest.TestCase):
    """The heartbeat that keeps a live run from looking like a crashed one."""

    def test_refreshes_the_mtime_of_a_held_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / LOCK_NAME
            lock.write_text("held", encoding="utf-8")
            old = time.time() - 10_000
            os.utime(lock, (old, old))

            touch_lock(lock)

            self.assertGreater(lock.stat().st_mtime, old + 5_000)

    def test_a_lock_that_cannot_be_refreshed_warns_instead_of_raising(self) -> None:
        # Best effort by contract: the run already owns this lock, so failing
        # to extend it must not turn a training epoch into a crash.
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / LOCK_NAME
            lock.write_text("held", encoding="utf-8")
            with mock.patch("cvtrain.fsio.os.utime", side_effect=OSError("denied")):
                with self.assertLogs("cvtrain.fsio", level="WARNING") as captured:
                    touch_lock(lock)
        self.assertTrue(any("Could not refresh lock" in line for line in captured.output))


if __name__ == "__main__":
    unittest.main()
