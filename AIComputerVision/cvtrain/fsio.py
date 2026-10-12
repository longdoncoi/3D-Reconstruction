"""Filesystem primitives: hashing and crash-safe writes.

Both the manifest writer and the model publisher need "write a file without
ever leaving a half-written one behind". Keeping that logic here means there is
exactly one temp-file + rename implementation to get right (and to test), and
one hashing routine so a digest computed during training can never disagree
with one recomputed by ``--verify-manifest``.
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

from .errors import ConcurrentRunError

__all__ = [
    "BACKUP_SUFFIX",
    "CHUNK_SIZE",
    "LOCK_NAME",
    "LOCK_STALE_SECONDS",
    "atomic_copy",
    "atomic_replace_with_backup",
    "atomic_write_text",
    "directory_lock",
    "sha256_file",
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
    """Return the hex SHA-256 of ``path``, streamed in :data:`CHUNK_SIZE` chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8", newline: str = "\n") -> Path:
    """Write ``text`` to ``path`` atomically (temp file in the target directory + rename).

    The payload is flushed and ``fsync``-ed before the rename so the published
    file is durable, not just readable. ``newline="\\n"`` keeps text files
    byte-identical across platforms, which matters for a JSON manifest that is
    diffed by Git.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding=encoding, newline=newline) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def atomic_copy(src: Path, dst: Path) -> Path:
    """Copy ``src`` onto ``dst`` atomically; returns ``dst``.

    Readers therefore see either the previous artifact or the complete new one,
    never a truncated ONNX file mid-copy. The temp copy is ``fsync``-ed before
    the rename so the published artifact is durable, not just complete.
    """

    dst.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(dst.parent), prefix=f".{dst.name}.", suffix=".tmp")
    try:
        # Copy and fsync through the mkstemp descriptor: on Windows reopening
        # the temp path would fail (mkstemp marks the file delete-on-close).
        with os.fdopen(handle, "wb") as stream:
            with src.open("rb") as source:
                shutil.copyfileobj(source, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, dst)
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


def _remove_lock(lock: Path) -> None:
    """Delete ``lock``, tolerating the case where it is already gone."""

    try:
        os.unlink(lock)
    except FileNotFoundError:
        pass


def _lock_age(lock: Path) -> float | None:
    """Seconds since ``lock`` was last written, or ``None`` when it is gone."""

    try:
        return max(0.0, time.time() - lock.stat().st_mtime)
    except OSError:
        return None


def _held_message(lock: Path) -> str:
    return f"{lock} is held by another cvtrain run; wait for it to finish, or delete the file if no run is active"


def _acquire_lock(lock: Path, stale_after: float) -> None:
    """Create ``lock`` exclusively, replacing it only when it is provably stale."""

    last_error: FileExistsError | None = None
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            last_error = exc
            age = _lock_age(lock)
            if age is not None and age > stale_after:
                LOG.warning("Stealing stale lock %s (age %.0fs > %.0fs)", lock, age, stale_after)
                _remove_lock(lock)
                continue
            raise ConcurrentRunError(_held_message(lock)) from exc
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(f"owner pid={os.getpid()} acquired={datetime.now(timezone.utc).isoformat()}\n")
        return
    raise ConcurrentRunError(_held_message(lock)) from last_error


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
    forever. The lock is always removed on the way out, including when the body
    raises.
    """

    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / LOCK_NAME
    _acquire_lock(lock, stale_after)
    try:
        yield lock
    finally:
        _remove_lock(lock)
