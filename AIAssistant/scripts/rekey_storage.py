"""Rotate the at-rest key of the local state (ADR 0030).

The task store, the pending-action store and the RAG cache are sealed with
**one** key (ADR 0027/0028). Until now there was no way to change that key:
installing a new ``AI_STORAGE_KEY`` made every row unreadable, so the state
had to be thrown away. This tool closes that gap:

* ``--verify`` is read-only. It reports how many task rows, event payloads
  and pending payloads the configured key can actually open and exits
  non-zero when it cannot. Run it *before* a rotation to be sure the old key
  is the one the state on disk was sealed with.
* the default action re-encrypts the state under a new key and deletes the
  RAG cache — a derived artifact that is rebuilt from the source documents,
  so it is dropped instead of being re-keyed.
* ``--dry-run`` reports what would be rewritten and writes nothing.

Everything fails closed: a malformed key, a key that cannot open the state
or a missing state file aborts with a message and a non-zero exit code. The
state is never left half-rotated — the rotation is a transaction (task
store) and an atomic file replacement (pending store), and a rotation that
was interrupted can simply be run again with the same key pair.

Usage::

    python AIAssistant/scripts/rekey_storage.py --verify
    python AIAssistant/scripts/rekey_storage.py --new-key base64:<url-safe>
    python AIAssistant/scripts/rekey_storage.py --new-key-file new.key --dry-run

Exit codes: 0 = success, 1 = the state could not be read or rotated,
2 = usage error.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_assistant.adapters.persistence.pending_store import PendingActionStore
from ai_assistant.adapters.persistence.sqlite_store import SqliteTaskStore
from ai_assistant.adapters.persistence.storage_crypto import (
    AesGcmFieldCipher,
    StorageCryptoError,
    StorageDecryptionError,
    StorageKeyError,
    cipher_from_key_file,
    cipher_from_key_text,
    reseal_text,
)
from ai_assistant.config.paths import resolve_paths
from ai_assistant.settings import ArchitectureSettings, storage_key, storage_key_location

#: The composition root's base directory (``…/AIAssistant/src``): the paths
#: below are resolved exactly the way ``bootstrap.server`` resolves them.
BASE_DIR = ROOT / "src"

#: The state files the rotation touches, inside the state directory.
TASKS_DB = "tasks.sqlite"
PENDING_FILE = "pending_agent_actions.json"

#: The RAG cache artifacts the rotation invalidates. The names mirror
#: ``AppPaths.cache_index`` / ``cache_chunks`` / ``cache_bm25`` /
#: ``cache_metadata``; the test suite pins the two lists together so a
#: rename cannot leave a stale cache behind.
CACHE_ARTIFACTS = ("faiss_index.bin", "chunks.pkl", "bm25.pkl", "metadata.json")

#: The sealed columns of a task row and the context each one was sealed with.
#: The context is the AEAD associated data: it must not change during a
#: rotation or the ciphertext would not authenticate any more.
TASK_COLUMNS = ("task.message", "task.result", "task.error", "task.metadata")


@dataclass
class _Report:
    """What the key could open, and what a rotation would rewrite."""

    tasks: int = 0
    task_failures: int = 0
    events: int = 0
    event_failures: int = 0
    pending: int = 0
    pending_failures: int = 0
    changed_rows: int = 0
    changed_pending: int = 0

    @property
    def failures(self) -> int:
        """The number of sealed values the key could not open."""
        return self.task_failures + self.event_failures + self.pending_failures


def _state_files(state_dir: Path) -> tuple[Path, Path]:
    """The task database and the pending-action file inside ``state_dir``."""
    return state_dir / TASKS_DB, state_dir / PENDING_FILE


def _state_dirs(data_dir: str) -> tuple[Path, Path]:
    """Return the state directory and the RAG cache directory.

    Without an override the paths are the ones the composition root uses:
    the task store and the pending store live in ``settings.data_dir`` and the
    RAG cache in ``paths.cache_dir``. ``--data-dir`` points at the directory
    that holds ``tasks.sqlite`` and ``pending_agent_actions.json``; the cache
    is then expected in ``<data-dir>/Cache``.
    """
    if data_dir:
        state = Path(data_dir).expanduser().resolve()
        return state, state / "Cache"
    paths = resolve_paths(BASE_DIR)
    settings = ArchitectureSettings.load(BASE_DIR)
    return settings.data_dir, paths.cache_dir


def _open_row(
    row: tuple[str | None, ...],
    cipher: AesGcmFieldCipher,
    new: AesGcmFieldCipher | None,
    report: _Report,
) -> bool:
    """Open one task row; return True when a value could not be opened."""
    changed = False
    for value, context in zip(row, TASK_COLUMNS):
        if value is None:
            continue
        try:
            cipher.open(value, context=context)
        except StorageCryptoError:
            return True
        if new is not None and reseal_text(cipher, new, value, context=context)[1]:
            changed = True
    if changed:
        report.changed_rows += 1
    return False


def _open_tasks(
    db_path: Path,
    cipher: AesGcmFieldCipher,
    new: AesGcmFieldCipher | None,
    report: _Report,
) -> None:
    """Open every task row and event payload with ``cipher``."""
    connection = sqlite3.connect(str(db_path))
    try:
        task_rows = connection.execute(
            "SELECT message, result, error, metadata FROM tasks ORDER BY rowid",
        ).fetchall()
        event_rows = connection.execute(
            "SELECT data FROM task_events ORDER BY rowid",
        ).fetchall()
    finally:
        connection.close()
    for row in task_rows:
        report.tasks += 1
        if _open_row(row, cipher, new, report):
            report.task_failures += 1
    for (data,) in event_rows:
        report.events += 1
        try:
            cipher.open(data, context="event.data")
        except StorageCryptoError:
            report.event_failures += 1
            continue
        if new is not None and reseal_text(cipher, new, data, context="event.data")[1]:
            report.changed_rows += 1


def _open_pending(
    path: Path,
    cipher: AesGcmFieldCipher,
    new: AesGcmFieldCipher | None,
    report: _Report,
) -> None:
    """Open the pending payload with ``cipher``."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        report.pending += 1
        report.pending_failures += 1
        return
    if not raw.strip():
        return
    report.pending += 1
    try:
        cipher.open(raw, context="pending.actions")
    except StorageCryptoError:
        report.pending_failures += 1
        return
    if new is not None and reseal_text(cipher, new, raw, context="pending.actions")[1]:
        report.changed_pending += 1


def _inspect(
    state_dir: Path,
    cipher: AesGcmFieldCipher,
    new: AesGcmFieldCipher | None = None,
) -> _Report:
    """Open every sealed value with ``cipher`` and count what it could open.

    With ``new`` the values a rotation would rewrite are counted as well, so
    the dry run can report them without writing anything.
    """
    report = _Report()
    db_path, pending_path = _state_files(state_dir)
    if db_path.exists():
        _open_tasks(db_path, cipher, new, report)
    if pending_path.exists():
        _open_pending(pending_path, cipher, new, report)
    return report


def _old_cipher(args: argparse.Namespace) -> AesGcmFieldCipher:
    """The key the state on disk was sealed with.

    Without an explicit flag the *configured* key is used: ``AI_STORAGE_KEY``,
    then ``AI_STORAGE_KEY_FILE``, then the key file the application
    generated. The file is never created here — a rotation must not bring a
    new key into existence, that key could not open the state anyway.
    """
    if args.old_key:
        return cipher_from_key_text(args.old_key, source="--old-key")
    if args.old_key_file:
        return cipher_from_key_file(Path(args.old_key_file))
    configured = storage_key()
    if configured:
        return cipher_from_key_text(configured, source="AI_STORAGE_KEY")
    path, explicit = storage_key_location()
    if path.exists():
        return cipher_from_key_file(path)
    if explicit:
        message = f"AI_STORAGE_KEY_FILE does not exist: {path}"
        raise StorageKeyError(message)
    message = (
        "no at-rest key is configured: set AI_STORAGE_KEY / AI_STORAGE_KEY_FILE "
        "or pass --old-key / --old-key-file"
    )
    raise StorageKeyError(message)


def _new_cipher(args: argparse.Namespace) -> AesGcmFieldCipher:
    """The key the state is re-encrypted under."""
    if args.new_key:
        return cipher_from_key_text(args.new_key, source="--new-key")
    if args.new_key_file:
        return cipher_from_key_file(Path(args.new_key_file))
    message = "the rotation needs a new key: pass --new-key or --new-key-file"
    raise StorageKeyError(message)


def _delete_cache(cache_dir: Path) -> int:
    """Delete the RAG cache: it is derived, so it is rebuilt, not re-keyed."""
    deleted = 0
    for name in CACHE_ARTIFACTS:
        path = cache_dir / name
        if path.exists():
            path.unlink()
            deleted += 1
    return deleted


def _existing_cache(cache_dir: Path) -> list[Path]:
    """The cache artifacts that are on disk right now."""
    return [cache_dir / name for name in CACHE_ARTIFACTS if (cache_dir / name).exists()]


def _verify(state_dir: Path, cipher: AesGcmFieldCipher) -> int:
    """Report what the configured key can open (read-only)."""
    report = _inspect(state_dir, cipher)
    print("VERIFY")
    print(f"  state directory           : {state_dir}")
    print(f"  task rows readable        : {report.tasks - report.task_failures}/{report.tasks}")
    print(f"  event payloads readable   : {report.events - report.event_failures}/{report.events}")
    print(f"  pending payloads readable : {report.pending - report.pending_failures}/{report.pending}")
    if report.failures:
        sys.stdout.flush()
        print(
            f"VERIFY FAIL: the configured key cannot open {report.failures} sealed value(s) "
            f"({report.task_failures} task row(s), {report.event_failures} event(s), "
            f"{report.pending_failures} pending payload(s)) - the state was sealed with "
            "another key",
            file=sys.stderr,
        )
        return 1
    print("  OK: the configured key opens every sealed value in the local state.")
    return 0


def _print_dry_run(state_dir: Path, cache_dir: Path, report: _Report) -> None:
    """Report what a rotation would rewrite — nothing is written."""
    print("DRY RUN - nothing was written")
    print(f"  state directory                      : {state_dir}")
    print(f"  rows that would be re-encrypted      : {report.changed_rows}")
    print(f"  pending payloads that would be sealed: {report.changed_pending}")
    print(f"  RAG cache files that would be deleted: {len(_existing_cache(cache_dir))}")
    print("")
    print("Re-run without --dry-run to apply the rotation.")


def _next_step(args: argparse.Namespace) -> list[str]:
    """The exact steps the operator has to take after the rotation.

    The key material itself is never echoed: it would end up in a terminal
    history and in whatever the operator pastes the output into.
    """
    if args.new_key_file:
        return [
            f"  set AI_STORAGE_KEY_FILE={args.new_key_file}",
            "  (that file already holds the new key)",
        ]
    return [
        "  set AI_STORAGE_KEY=<the new key you passed to --new-key>",
        "  (or write the new key to a file and set AI_STORAGE_KEY_FILE=<that file>)",
    ]


def _rotate(
    state_dir: Path,
    cache_dir: Path,
    old: AesGcmFieldCipher,
    new: AesGcmFieldCipher,
    args: argparse.Namespace,
) -> int:
    """Re-encrypt the state under ``new`` and drop the derived RAG cache."""
    db_path, pending_path = _state_files(state_dir)
    pending = 0
    rows = 0
    if pending_path.exists():
        pending = PendingActionStore(pending_path).rekey(old, new)
    if db_path.exists():
        store = SqliteTaskStore(db_path)
        try:
            rows = store.rekey(old, new)
        finally:
            store.close()
    deleted = _delete_cache(cache_dir)
    print("REKEY OK")
    print(f"  rows re-encrypted (tasks + events)  : {rows}")
    print(f"  pending payload re-encrypted        : {pending}")
    print(f"  RAG cache files deleted             : {deleted}")
    print("")
    print("Next step - install the new key before the assistant starts again:")
    for line in _next_step(args):
        print(line)
    print("  then restart the assistant: the state on disk is now sealed with the new key.")
    return 0


def _execute(args: argparse.Namespace) -> int:
    """Run the requested mode (verify, dry run or rotation)."""
    state_dir, cache_dir = _state_dirs(args.data_dir)
    db_path, pending_path = _state_files(state_dir)
    if not db_path.exists() and not pending_path.exists():
        message = f"no local state found under {state_dir}"
        raise FileNotFoundError(message)
    old = _old_cipher(args)
    if args.verify:
        return _verify(state_dir, old)
    new = _new_cipher(args)
    report = _inspect(state_dir, old, new)
    if report.failures:
        message = (
            f"the configured key cannot open the local state under {state_dir}: "
            f"{report.task_failures} task row(s), {report.event_failures} event(s) and "
            f"{report.pending_failures} pending payload(s) failed to authenticate - "
            "the state was sealed with another key"
        )
        raise StorageDecryptionError(message)
    if args.dry_run:
        _print_dry_run(state_dir, cache_dir, report)
        return 0
    return _rotate(state_dir, cache_dir, old, new, args)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rotate (or verify) the at-rest key of the local state.",
    )
    new_key = parser.add_mutually_exclusive_group()
    new_key.add_argument(
        "--new-key",
        default="",
        help="The new key: base64:<url-safe>, hex:<digits> or bare url-safe base64 (32 bytes).",
    )
    new_key.add_argument("--new-key-file", default="", help="A file holding the new key.")
    old_key = parser.add_mutually_exclusive_group()
    old_key.add_argument(
        "--old-key",
        default="",
        help="The key the state on disk was sealed with (default: the configured key).",
    )
    old_key.add_argument("--old-key-file", default="", help="A file holding the old key.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Read-only: report what the configured key can open.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be re-encrypted; write nothing.",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Directory holding tasks.sqlite and pending_agent_actions.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = "VERIFY" if args.verify else "REKEY"
    try:
        return _execute(args)
    except StorageCryptoError as error:
        sys.stdout.flush()
        print(f"{mode} FAIL: {error}", file=sys.stderr)
        return 1
    except sqlite3.Error as error:
        sys.stdout.flush()
        print(f"{mode} FAIL: the task store could not be read: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        sys.stdout.flush()
        print(f"{mode} FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
