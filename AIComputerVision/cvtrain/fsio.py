"""Filesystem primitives: hashing and crash-safe writes.

Both the manifest writer and the model publisher need "write a file without
ever leaving a half-written one behind". Keeping that logic here means there is
exactly one temp-file + rename implementation to get right (and to test), and
one hashing routine so a digest computed during training can never disagree
with one recomputed by ``--verify-manifest``.

Reads the pipeline *depends on* (hashing, measuring) and writes it performs
(staging a temp file, ``fsync``, the rename) raise
:class:`~cvtrain.errors.StorageError` instead of ``OSError``: a file that
vanished, a permission denied or a dying disk are environment facts with exit
code ``2``, not bugs with a traceback. The single carve-out is a *source*
that should exist and does not (see :func:`atomic_copy`).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from .errors import ConcurrentRunError, StorageError

__all__ = [
    "BACKUP_SUFFIX",
    "CHUNK_SIZE",
    "LOCK_NAME",
    "LOCK_STALE_SECONDS",
    "atomic_copy",
    "atomic_replace_with_backup",
    "atomic_write_text",
    "directory_lock",
    "file_size",
    "sha256_file",
    "touch_lock",
]

LOG = logging.getLogger("cvtrain.fsio")

CHUNK_SIZE = 1 << 20
"""Hashing block size (1 MiB): large enough to be fast, small enough for RAM."""

BACKUP_SUFFIX = ".bak"
"""Suffix of the one-generation rollback copy kept beside a replaced artifact."""

LOCK_NAME = ".cvtrain.lock"
"""Name of the advisory lock file that marks a models directory as busy."""

LOCK_STALE_SECONDS = 24 * 60 * 60
"""Age after which a lock file is treated as abandoned (its process crashed)."""


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of ``path``, streamed in :data:`CHUNK_SIZE` chunks.

    Raises :class:`~cvtrain.errors.StorageError` when the file cannot be read:
    provenance and verification *depend on* this digest, so an unreadable file
    is exit code ``2`` ("cannot vouch"), not an ``OSError`` traceback.
    """

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise StorageError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def file_size(path: Path) -> int:
    """Return the size of ``path`` in bytes.

    The typed twin of :func:`sha256_file` for the manifest's ``size_bytes``:
    same rule — an unmeasurable artifact is a :class:`StorageError`, not a
    traceback.
    """

    try:
        return path.stat().st_size
    except OSError as exc:
        raise StorageError(f"cannot measure {path}: {exc}") from exc


def _fsync_dir(directory: Path) -> None:
    """Best-effort durability for the directory entry a rename just created.

    ``fsync`` on the file guarantees the *contents* survive a crash, but on a
    non-journalled filesystem the *name* itself can still be lost without an
    ``fsync`` of the containing directory. Platforms that cannot open a
    directory (Windows raises here) simply skip the step: the rename is still
    atomic, only the crash-durability is best-effort there.
    """

    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8", newline: str = "\n") -> Path:
    """Write ``text`` to ``path`` atomically (temp file in the target directory + rename).

    The payload is flushed and ``fsync``-ed before the rename so the published
    file is durable, not just readable. ``newline="\\n"`` keeps text files
    byte-identical across platforms, which matters for a JSON manifest that is
    diffed by Git.

    Every failure here — an unwritable directory, a full disk, a rename that
    the OS refuses — is an environment fact, so it is re-raised as
    :class:`~cvtrain.errors.StorageError` (exit ``2``): the manifest that
    carries provenance is one this run could not write, not a pipeline bug.
    """

    tmp_name = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        with os.fdopen(handle, "w", encoding=encoding, newline=newline) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
        _fsync_dir(path.parent)
    except OSError as exc:
        raise StorageError(f"Cannot write {path}: {exc}") from exc
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def atomic_copy(src: Path, dst: Path) -> Path:
    """Copy ``src`` onto ``dst`` atomically; returns ``dst``.

    Readers therefore see either the previous artifact or the complete new one,
    never a truncated ONNX file mid-copy. The temp copy is ``fsync``-ed before
    the rename so the published artifact is durable, not just complete.

    ``src`` is opened *before* the typed block on purpose: a framework output
    that was supposed to exist and does not is a bug, and keeps its traceback
    (ADR 0005). Everything that writes ``dst`` — staging, ``fsync``, rename —
    is an environment fact and becomes a
    :class:`~cvtrain.errors.StorageError` instead, which is exactly what an
    operator needs when the OS refuses the rename (a read-only ``Models/``, a
    Windows sharing violation, a full disk).
    """

    tmp_name = ""
    with src.open("rb") as source:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            handle, tmp_name = tempfile.mkstemp(dir=str(dst.parent), prefix=f".{dst.name}.", suffix=".tmp")
            # Copy and fsync through the mkstemp descriptor: on Windows reopening
            # the temp path would fail (mkstemp marks the file delete-on-close).
            with os.fdopen(handle, "wb") as stream:
                shutil.copyfileobj(source, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_name, dst)
            _fsync_dir(dst.parent)
        except OSError as exc:
            raise StorageError(f"Cannot publish {dst}: {exc}") from exc
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
    return dst


def atomic_replace_with_backup(src: Path, dst: Path) -> Path:
    """Publish ``src`` at ``dst``, keeping the artifact it replaces at ``<dst>.bak``.

    One generation of rollback: an export that turns out to be unusable
    (truncated weights, broken ONNX conversion, wrong image size) can be put back
    without retraining. The copy is made — and fsync-ed — *before* the replace,
    so a failure while backing up leaves the published artifact untouched.

    The backup is a recovery artifact, not a provenance one: it deliberately has
    no manifest entry, and ``manifest.verify_manifest``'s ``*.onnx`` scan does
    not match ``yolo11n.onnx.bak``, so it can never be mistaken for a model the
    Qt app should load.
    """

    if dst.exists():
        atomic_copy(dst, dst.with_name(dst.name + BACKUP_SUFFIX))
    return atomic_copy(src, dst)


def _unlink_quietly(lock: Path) -> None:
    """Delete ``lock``, tolerating the case where it is already gone."""

    try:
        os.unlink(lock)
    except FileNotFoundError:
        pass


def _remove_lock(lock: Path, expected_token: str) -> None:
    """Release ``lock``, but only when it is still ours.

    The file carries ``token=<pid>-<ns>`` from acquisition. If the content no
    longer carries our token, someone stole the lock while the body ran (only
    possible past ``stale_after``) and deleting it would unlock *their* run —
    so it is left in place with a warning. A lock that is already gone needs
    no release at all.
    """

    try:
        content = lock.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    except OSError as exc:
        LOG.warning("Could not verify ownership of %s: %s; leaving it in place", lock, exc)
        return
    if f"token={expected_token}" not in content:
        LOG.warning("Lock %s changed hands while held; leaving the new owner's lock in place", lock)
        return
    _unlink_quietly(lock)


def _lock_age(lock: Path) -> float | None:
    """Seconds since ``lock`` was last written, or ``None`` when it is gone."""

    try:
        return max(0.0, time.time() - lock.stat().st_mtime)
    except OSError:
        return None


def _held_message(lock: Path) -> str:
    return f"{lock} is held by another cvtrain run; wait for it to finish, or delete the file if no run is active"


def _acquire_lock(lock: Path, stale_after: float) -> str:
    """Create ``lock`` exclusively, replacing it only when it is provably stale.

    Returns the ownership token written into the file, so the release path can
    prove the lock is still ours before deleting it.
    """

    token = f"{os.getpid()}-{time.time_ns()}"
    last_error: FileExistsError | None = None
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            last_error = exc
            age = _lock_age(lock)
            if age is not None and age > stale_after:
                LOG.warning("Stealing stale lock %s (age %.0fs > %.0fs)", lock, age, stale_after)
                _unlink_quietly(lock)
                continue
            raise ConcurrentRunError(_held_message(lock)) from exc
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(f"owner pid={os.getpid()} token={token} acquired={datetime.now(timezone.utc).isoformat()}\n")
        return token
    raise ConcurrentRunError(_held_message(lock)) from last_error


def touch_lock(lock: Path) -> None:
    """Refresh ``lock``'s mtime: "alive now", the heartbeat of a run holding it.

    :func:`directory_lock` treats a lock older than ``LOCK_STALE_SECONDS`` as
    abandoned and replaces it, which is how a crashed run stops wedging the
    directory. That rule only stays correct if a *live* run keeps its lock
    young, so training refreshes it once per epoch — otherwise a single
    long-running model (``MAX_EPOCHS`` is deliberately generous) would be
    mistaken for a crash halfway through and a second writer could be let in.
    Best effort: failing to refresh cannot stop a run that already owns the
    lock, so it warns and returns rather than raising.
    """

    try:
        os.utime(lock)
    except OSError as exc:
        LOG.warning("Could not refresh lock %s: %s", lock, exc)


@contextlib.contextmanager
def directory_lock(directory: Path, *, stale_after: float = LOCK_STALE_SECONDS) -> Iterator[Path]:
    """Hold an advisory lock on ``directory`` for the duration of the block.

    The lock is a file created with ``O_CREAT | O_EXCL`` — the one atomic
    "create if absent" primitive every supported platform shares — so a second
    ``cvtrain`` run aimed at the same directory fails fast
    (:class:`~cvtrain.errors.ConcurrentRunError`) instead of racing on
    ``manifest.json``. It is advisory: anything that does not go through this
    helper can still write; the point is to turn the pipeline's documented
    single-writer assumption into an *enforced* one.

    A lock older than ``stale_after`` is treated as abandoned (its process
    crashed) and replaced with a warning, so a crash cannot wedge the next run
    forever; a run that keeps working keeps its lock young with
    :func:`touch_lock`. The lock is always released on the way out — unless it
    changed hands while the body ran, in which case the new owner's file is
    left alone (see :func:`_remove_lock`).
    """

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageError(f"Cannot prepare the models directory {directory}: {exc}") from exc
    lock = directory / LOCK_NAME
    token = _acquire_lock(lock, stale_after)
    try:
        yield lock
    finally:
        _remove_lock(lock, token)
