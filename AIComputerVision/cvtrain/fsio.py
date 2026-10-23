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
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from .errors import ConcurrentRunError, StorageError

__all__ = [
    "BACKUP_SUFFIX",
    "CHUNK_SIZE",
    "LOCK_HEARTBEAT_SECONDS",
    "LOCK_NAME",
    "LOCK_STALE_SECONDS",
    "RUN_ALIVE_SECONDS",
    "RUN_MARKER_NAME",
    "STEAL_STALE_SECONDS",
    "STEAL_SUFFIX",
    "atomic_copy",
    "atomic_replace_with_backup",
    "atomic_write_text",
    "clear_run_marker",
    "directory_lock",
    "file_size",
    "prune_run_directories",
    "remove_tree",
    "sha256_file",
    "touch_lock",
    "touch_run_marker",
]

LOG = logging.getLogger("cvtrain.fsio")

CHUNK_SIZE = 1 << 20
"""Hashing block size (1 MiB): large enough to be fast, small enough for RAM."""

BACKUP_SUFFIX = ".bak"
"""Suffix of the one-generation rollback copy kept beside a replaced artifact."""

LOCK_NAME = ".cvtrain.lock"
"""Name of the advisory lock file that marks a models directory as busy."""

LOCK_STALE_SECONDS = 24 * 60 * 60
"""Age after which a lock file is treated as abandoned (its process crashed).

This is the *fallback* rule: a lock written by this build names its owner's
host and PID, so a same-host process that is gone can be detected immediately
(see :func:`_is_stale`). The age still matters for a lock written by an older
build, for a shared volume where the owner is on another host, and for the
startup window before the owner's heartbeat thread writes.
"""

LOCK_HEARTBEAT_SECONDS = 30
"""How often a held lock is refreshed while the run is alive.

Training used to refresh the lock only at epoch boundaries, so a single epoch
longer than the stale window made a live run look crashed. The heartbeat owns
the lease now; the epoch callback still refreshes too (cheap, and it keeps the
lock fresh exactly at the moment progress is visible).
"""

RUN_MARKER_NAME = ".cvtrain-alive"
"""Lease file inside a run directory, naming the process that is writing it.

``--prune-runs`` consults it together with the filesystem age: a run whose
owner is still alive is never deleted, even when a long epoch has not touched
``weights/last.pt`` for a while. The owner removes it when training returns; a
killed process leaves it behind and its dead PID lets the pruner age the run
normally.
"""

STEAL_SUFFIX = ".steal"
"""Suffix of the sidecar claim that serialises stale-lock stealing."""

STEAL_STALE_SECONDS = 60
"""Age after which a lock-steal claim is itself treated as abandoned.

A process that crashes between claiming and deleting the stale lock would
otherwise block every future steal forever. The window is short because the
claim is held only for the few filesystem operations that follow it.
"""

RUN_ALIVE_SECONDS = 24 * 60 * 60
"""Age after which a training run directory is considered abandoned.

Pruning is the only destructive operation in the package, so its safe
direction is *keep*: a run that was still writing recently is never deleted,
whatever its position in the history. The window is a day — the same order as
:data:`LOCK_STALE_SECONDS` — because the framework rewrites its checkpoint
after every epoch, so a live run refreshes this age continuously and a single
epoch cannot outlast it.
"""


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
        _discard_temp(tmp_name)
    return path


def _discard_temp(tmp_name: str) -> None:
    """Remove a staged temp file, never letting cleanup replace the real error.

    ``finally`` runs while an exception is on its way out: an ``OSError`` raised
    here would *supplant* the ``StorageError`` that explains what actually went
    wrong, turning an exit-2 verdict back into a traceback.
    """

    if not os.path.exists(tmp_name):
        return
    try:
        os.unlink(tmp_name)
    except OSError as exc:
        LOG.warning("Could not remove %s: %s", tmp_name, exc)


def atomic_copy(src: Path, dst: Path) -> Path:
    """Copy ``src`` onto ``dst`` atomically; returns ``dst``.

    Readers therefore see either the previous artifact or the complete new one,
    never a truncated ONNX file mid-copy. The temp copy is ``fsync``-ed before
    the rename so the published artifact is durable, not just complete.

    ``src`` is opened *before* the typed block on purpose: a framework output
    that was supposed to exist and does not is a bug, and keeps its traceback
    (ADR 0005, narrowed by ADR 0006). Everything that writes ``dst`` — staging,
    ``fsync``, rename — is an environment fact and becomes a
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
            _discard_temp(tmp_name)
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
    """Delete ``lock``; a failure here is cleanup noise, never a verdict.

    This runs from the release path and the stale-steal path, both of which are
    already handling another problem: letting ``OSError`` escape would *replace*
    the error actually in flight (or turn "I could not take the lock" into a
    traceback), which is exactly the mix-up the typed taxonomy exists to stop.
    """

    try:
        os.unlink(lock)
    except FileNotFoundError:
        pass
    except OSError as exc:
        LOG.warning("Could not remove %s: %s", lock, exc)


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


def _read_owner(path: Path) -> tuple[int | None, str | None]:
    """``(pid, host)`` recorded in a lease file, or ``(None, None)``.

    Both the lock and the run marker use the same tiny ``key=value`` format;
    an unreadable or older file simply yields no owner and the age rule speaks.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None, None
    pid: int | None = None
    host: str | None = None
    for token in text.split():
        key, separator, value = token.partition("=")
        if not separator:
            continue
        if key == "pid":
            try:
                pid = int(value)
            except ValueError:
                pid = None
        elif key == "host":
            host = value or None
    return pid, host


def _process_alive(pid: int) -> bool | None:
    """Whether ``pid`` names a live process; ``None`` when that cannot be known.

    ``os.kill(pid, 0)`` is the POSIX probe. It is *not* the Windows one: there
    ``os.kill`` calls ``TerminateProcess`` for every signal except the console
    control events, so probing with ``0`` would kill the process it was asked
    about. Windows gets a ``SYNCHRONIZE`` handle and a zero-time wait instead.

    ``None`` is deliberate: a platform or permission that cannot answer must
    fall back to the age rule, never to "dead" — deleting another run's lock on
    an unknown would be the unsafe direction.
    """

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    return _probe_process(pid)


def _posix_process_alive(pid: int) -> bool | None:  # pragma: no cover - POSIX only
    """POSIX PID probe: signal 0 asks "does this exist?" without delivering.

    ``PermissionError`` means the process exists but belongs to another user —
    alive, and definitely not ours to declare dead. The platform split is a
    module-level choice (see :data:`_probe_process`), so each CI OS exercises
    the one probe it can actually run; the shared decision above is covered on
    both, and the mutation set pins the direction of both answers.
    """

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _windows_process_alive(pid: int) -> bool | None:  # pragma: no cover - Windows only
    """Windows PID probe; see :func:`_process_alive` for why ``os.kill`` is not used."""

    import ctypes

    SYNCHRONIZE = 0x00100000
    WAIT_OBJECT_0 = 0x00000000
    WAIT_TIMEOUT = 0x00000102
    ERROR_INVALID_PARAMETER = 87
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel32.OpenProcess(SYNCHRONIZE, 0, pid)
        if not handle:
            # ERROR_INVALID_PARAMETER is "no such process"; access denied means
            # the process exists but belongs to someone else. Anything else is
            # unknown, and unknown falls back to the age rule.
            return False if ctypes.get_last_error() == ERROR_INVALID_PARAMETER else None  # type: ignore[attr-defined]
        try:
            # ``ctypes`` is untyped, so the status is cast explicitly: the
            # function's contract is ``bool | None``, not ``Any``. Only a
            # timeout proves the process is alive; an exited handle proves it
            # is gone; anything else (``WAIT_FAILED``) is unknown and falls
            # back to the lease age rather than declaring an owner dead.
            status = int(kernel32.WaitForSingleObject(handle, 0))
            if status == WAIT_TIMEOUT:
                return True
            if status == WAIT_OBJECT_0:
                return False
            return None
        finally:
            kernel32.CloseHandle(handle)
    except (AttributeError, OSError):
        return None


#: The PID probe this platform can actually run.
#:
#: Chosen once at import time rather than branched per call: the dispatch then
#: has no platform-shaped branch for coverage to miss, and neither probe is
#: reachable on the wrong OS (``ctypes.WinDLL`` does not exist on POSIX, and
#: ``os.kill(pid, 0)`` terminates the process on Windows).
_probe_process = _windows_process_alive if os.name == "nt" else _posix_process_alive


def _is_stale(lock: Path, age: float, stale_after: float) -> bool:
    """Whether a lock older than ``age`` seconds belongs to a crashed run.

    Two independent signals, combined in the safe direction:

    * a same-host owner that is **alive** keeps the lock however old its lease
      is — a suspended process or a slow volume must not let a second writer
      into ``Models/``;
    * a same-host owner that is provably **dead** makes the lock stale at once —
      a crash between writing the lock and starting the heartbeat does not have
      to wait out the whole window;
    * when the owner cannot be probed (an older lock, another host, a
      permission), the lease age decides: ``age > stale_after``.

    The last case is the fallback for a lock from an older build or a shared
    volume, where the PID means nothing on this machine.
    """

    pid, host = _read_owner(lock)
    if pid is not None and host == socket.gethostname():
        alive = _process_alive(pid)
        if alive is True:
            return False
        if alive is False:
            return True
    return age > stale_after


def _held_message(lock: Path) -> str:
    return f"{lock} is held by another cvtrain run; wait for it to finish, or delete the file if no run is active"


def _claim_stale_lock(lock: Path) -> Path:
    """Take a sidecar claim before deleting a stale lock.

    ``O_EXCL`` makes *acquiring* the lock atomic, but the stale path used to
    have a check-then-delete race: two processes could both see an old mtime,
    both unlink, and the slower one could delete the lock the faster one had
    just created. The claim is a second ``O_EXCL`` file; only its owner may
    unlink a stale lock, so a second stealer is refused instead of racing. A
    claim crashed mid-steal ages out after :data:`STEAL_STALE_SECONDS`.

    Returns the claim path; the caller releases it in a ``finally``.
    """

    claim = lock.with_name(lock.name + STEAL_SUFFIX)
    token = f"{os.getpid()}-{time.time_ns()}"
    for _ in range(2):
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            age = _lock_age(claim)
            if age is not None and age > STEAL_STALE_SECONDS:
                # A previous stealer died between claiming and unlinking; its
                # claim is as abandoned as the lock it was going to steal.
                LOG.warning("Removing stale lock-steal claim %s (age %.0fs)", claim, age)
                _unlink_quietly(claim)
                continue
            raise ConcurrentRunError(_held_message(lock)) from exc
        except OSError as exc:
            raise StorageError(f"Cannot create the lock claim {claim}: {exc}") from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(
                    f"owner pid={os.getpid()} token={token} acquired={datetime.now(timezone.utc).isoformat()}\n"
                )
        except OSError as exc:
            with contextlib.suppress(OSError):
                os.close(fd)
            _unlink_quietly(claim)
            raise StorageError(f"Cannot write the lock claim {claim}: {exc}") from exc
        return claim
    raise ConcurrentRunError(_held_message(lock))


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
            if age is not None and _is_stale(lock, age, stale_after):
                claim = _claim_stale_lock(lock)
                try:
                    # Re-check *under the claim*: another process may have
                    # released the old lock (or replaced it with a fresh one)
                    # while this process was taking the claim. Deleting a lock
                    # that changed hands is the race the claim exists to stop.
                    current_age = _lock_age(lock)
                    if current_age is None:
                        continue
                    if not _is_stale(lock, current_age, stale_after):
                        raise ConcurrentRunError(_held_message(lock))
                    LOG.warning("Stealing stale lock %s (age %.0fs > %.0fs)", lock, current_age, stale_after)
                    _unlink_quietly(lock)
                finally:
                    _unlink_quietly(claim)
                continue
            raise ConcurrentRunError(_held_message(lock)) from exc
        except OSError as exc:
            # Anything else the filesystem can refuse — a read-only volume, a
            # permission, a full disk — is storage this run cannot write, not a
            # bug: exit 2, one line, no traceback.
            raise StorageError(f"Cannot create the lock {lock}: {exc}") from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(
                    f"owner pid={os.getpid()} host={socket.gethostname()} token={token} "
                    f"acquired={datetime.now(timezone.utc).isoformat()}\n"
                )
        except OSError as exc:
            # The file exists but now carries no token of ours, and release
            # only deletes a lock that still names it — leaving this one would
            # block every later run until the stale window expired. Take it
            # back first, then report why.
            with contextlib.suppress(OSError):
                os.close(fd)
            _unlink_quietly(lock)
            raise StorageError(f"Cannot write the lock {lock}: {exc}") from exc
        return token
    raise ConcurrentRunError(_held_message(lock)) from last_error


def touch_lock(lock: Path) -> None:
    """Refresh ``lock``'s mtime: "alive now", the lease of a run holding it.

    :func:`directory_lock` treats an expired lease as abandoned and replaces
    it, which is how a crashed run stops wedging the directory. That rule only
    stays correct if a *live* run keeps its lock young: the heartbeat thread in
    :func:`directory_lock` refreshes it every :data:`LOCK_HEARTBEAT_SECONDS`,
    and training also refreshes it once per epoch so progress and liveness stay
    in step. Best effort: failing to refresh cannot stop a run that already
    owns the lock, so it warns and returns rather than raising.
    """

    try:
        os.utime(lock)
    except OSError as exc:
        LOG.warning("Could not refresh lock %s: %s", lock, exc)


@contextlib.contextmanager
def directory_lock(
    directory: Path,
    *,
    stale_after: float = LOCK_STALE_SECONDS,
    heartbeat_interval: float | None = None,
) -> Iterator[Path]:
    """Hold an advisory lock on ``directory`` for the duration of the block.

    The lock is a file created with ``O_CREAT | O_EXCL`` — the one atomic
    "create if absent" primitive every supported platform shares — so a second
    ``cvtrain`` run aimed at the same directory fails fast
    (:class:`~cvtrain.errors.ConcurrentRunError`) instead of racing on
    ``manifest.json``. It is advisory: anything that does not go through this
    helper can still write; the point is to turn the pipeline's documented
    single-writer assumption into an *enforced* one.

    Liveness is a **lease**, not a guess from one timestamp: the lock names its
    owner (host + PID), a daemon heartbeat refreshes its mtime while the block
    runs, and a same-host owner that is provably gone is detectable without
    waiting out :data:`LOCK_STALE_SECONDS`. A lock from an older build or from
    another host falls back to the age rule. The lock is always released on
    the way out — unless it changed hands while the body ran, in which case the
    new owner's file is left alone (see :func:`_remove_lock`).
    """

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageError(f"Cannot prepare the models directory {directory}: {exc}") from exc
    lock = directory / LOCK_NAME
    token = _acquire_lock(lock, stale_after)
    interval = heartbeat_interval
    if interval is None:
        # A quarter of the window, bounded: frequent enough that a live run
        # never looks abandoned, cheap enough that a shared volume is not
        # hammered by a multi-hour training run.
        interval = min(LOCK_HEARTBEAT_SECONDS, max(1.0, stale_after / 4))
    stop = threading.Event()

    def _beat() -> None:
        while not stop.wait(interval):
            touch_lock(lock)

    heartbeat = threading.Thread(target=_beat, name="cvtrain-lock-heartbeat", daemon=True)
    heartbeat.start()
    try:
        yield lock
    finally:
        stop.set()
        heartbeat.join(timeout=5)
        _remove_lock(lock, token)


def touch_run_marker(run_dir: Path) -> None:
    """Create/refresh the run's lease file naming this process.

    Called by the backend at every epoch boundary and read by the pruner. The
    marker is a *process-liveness* signal; its mtime is a secondary one. Both
    are best effort: failing to write a lease must not fail a training run that
    is already underway, so this warns and returns.
    """

    marker = run_dir / RUN_MARKER_NAME
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text(
                f"owner pid={os.getpid()} host={socket.gethostname()} started={datetime.now(timezone.utc).isoformat()}\n",
                encoding="utf-8",
            )
        os.utime(marker)
    except OSError as exc:
        LOG.warning("Could not refresh run marker %s: %s", marker, exc)


def clear_run_marker(run_dir: Path) -> None:
    """Remove the run's lease when training returns (success or failure).

    A finished run is not live even though its process is: leaving the marker
    would make the pruner keep every run this session ever trained. A killed
    process cannot run this, and its marker's dead PID lets the normal age rule
    take over.
    """

    _unlink_quietly(run_dir / RUN_MARKER_NAME)


def _run_owner_alive(run_dir: Path) -> bool:
    """Whether the process named by the run's lease is still alive on this host."""

    pid, host = _read_owner(run_dir / RUN_MARKER_NAME)
    if pid is None or host is None or host != socket.gethostname():
        # A foreign host cannot be probed from here, and an absent/malformed
        # marker is not evidence of life. The age rule still applies.
        return False
    return _process_alive(pid) is True


def remove_tree(path: Path) -> None:
    """Delete a directory tree, or raise :class:`StorageError`.

    The one destructive primitive in the package, named so the rule it obeys
    is written down in one place: a refused deletion — a read-only volume, a
    Windows sharing violation, a file in use — is an *environment fact* and
    becomes a typed exit-2 error, never an ``OSError`` traceback from the
    middle of a cleanup.
    """

    try:
        shutil.rmtree(path)
    except OSError as exc:
        raise StorageError(f"Cannot remove {path}: {exc}") from exc


def _run_activity_mtime(run_dir: Path) -> float:
    """The newest mtime that proves this run wrote something recently.

    The directory's own mtime only moves when an *entry* is added or removed,
    so a run that keeps overwriting its checkpoint would look abandoned on
    NTFS. The checkpoint the framework rewrites after every epoch is consulted
    as well: it is the file whose freshness actually means "this run is
    alive". ``0.0`` means "unknown", which sorts the run last and lets the
    ordinary deletion rules speak.
    """

    try:
        newest = run_dir.stat().st_mtime
    except OSError:
        return 0.0
    checkpoint = run_dir / "weights" / "last.pt"
    try:
        newest = max(newest, checkpoint.stat().st_mtime)
    except OSError:
        pass
    marker = run_dir / RUN_MARKER_NAME
    try:
        newest = max(newest, marker.stat().st_mtime)
    except OSError:
        pass
    return newest


def prune_run_directories(runs_dir: Path, *, keep: int, prefix: str) -> list[Path]:
    """Delete all but the newest ``keep`` run directories under ``runs_dir``.

    ``runs_dir`` holds one sub-directory per model (``detect``, ``segment``,
    ``tracking``), each of which holds that model's runs. Only directories
    whose name starts with ``prefix`` are candidates: anything else in there
    was not written by this pipeline and is not ours to delete.

    Three rules make the operation safe, and each is a decision worth
    stating rather than a detail buried in the code:

    * the newest ``keep`` runs of *every* model survive — so ``--resume``
      still finds a checkpoint and a rollback is still possible;
    * a run whose last write is younger than :data:`RUN_ALIVE_SECONDS` is
      still running and is never deleted, whatever its position in the
      history. For a destructive operation the safe direction is to keep;
    * a directory that cannot be read or removed is a typed
      :class:`StorageError`, not a traceback — and the runs already deleted
      stay deleted, because a cleanup that cannot finish is not a rollback.

    Returns the paths that were deleted, in deletion order.
    """

    if keep < 0:
        raise ValueError(f"keep must not be negative, got {keep}")
    deleted: list[Path] = []
    try:
        model_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    except FileNotFoundError:
        # Nothing has ever been run here: there is nothing to prune, and an
        # absent directory is not a failure.
        return deleted
    except OSError as exc:
        raise StorageError(f"Cannot read {runs_dir}: {exc}") from exc
    for model_dir in model_dirs:
        try:
            candidates = [path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)]
        except OSError as exc:
            raise StorageError(f"Cannot read {model_dir}: {exc}") from exc
        # Newest first: the activity time is the only ordering the framework
        # gives us, and it is the same signal the resume rule uses to pick the
        # checkpoint a run would continue from. The name breaks a tie, and it
        # breaks it deterministically: our own run names embed their creation
        # timestamp (``train-<YYYYmmdd-HHMMSS-ffffff>-<pid>``), so the
        # lexicographic order *is* the chronological one. Without it, two
        # runs written in the same second would be kept or deleted according
        # to the order the filesystem happened to list them in.
        candidates.sort(key=lambda path: (_run_activity_mtime(path), path.name), reverse=True)
        for index, candidate in enumerate(candidates):
            if index < keep:
                continue
            age = time.time() - _run_activity_mtime(candidate)
            if age < RUN_ALIVE_SECONDS or _run_owner_alive(candidate):
                LOG.info(
                    "Keeping %s: still active (owner alive or last write %.0fs ago)",
                    candidate,
                    age,
                )
                continue
            remove_tree(candidate)
            deleted.append(candidate)
    return deleted
