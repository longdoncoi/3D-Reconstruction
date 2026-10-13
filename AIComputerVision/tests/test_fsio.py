"""Unit tests for the shared filesystem primitives."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from cvtrain import fsio as fsio_mod
from cvtrain.errors import ConcurrentRunError, ExitCode, StorageError
from cvtrain.fsio import (
    BACKUP_SUFFIX,
    LOCK_NAME,
    atomic_copy,
    atomic_replace_with_backup,
    atomic_write_text,
    directory_lock,
    file_size,
    sha256_file,
    touch_lock,
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
