"""Selectable LangGraph checkpoint backend with safe local fallback."""
from __future__ import annotations

import logging
import os
import sqlite3
import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from ai_assistant.ports import FieldCipher
from ai_assistant.settings import load_agent_runtime_settings

logger = logging.getLogger(__name__)

#: The associated-data context the checkpoint blobs are bound to. It is part
#: of the envelope, so a sealed checkpoint moved to another store fails
#: authentication instead of being read as if it belonged there.
CHECKPOINT_CONTEXT = "langgraph.checkpoint"

#: The cipher name recorded next to a sealed checkpoint. LangGraph splits the
#: type tag on "+", so the name must not contain one.
_CHECKPOINT_CIPHER_NAME = "aes-256-gcm"

# UUIDv6 uses the UUID epoch (1582-10-15) in 100 ns ticks. LangGraph creates
# every native checkpoint id with this format, so its age remains observable
# without decrypting the checkpoint payload. A legacy/custom id is never
# guessed: retention preserves that whole thread instead of risking data loss.
_UUID_EPOCH_OFFSET_100NS = 0x01B21DD213814000
_SECONDS_PER_DAY = 86_400


class CheckpointerHandle:
    """A built LangGraph saver plus the resource that owns it.

    The Postgres and Redis factories are context managers: the saver exists
    only while the manager is entered, and the connection is released on
    ``__exit__``. The SQLite backend owns a raw connection instead. Returning
    the bare saver (the old shape) leaked every one of those handles — and,
    worse, called ``setup()`` on the *manager* rather than the saver, so a
    configured external backend silently fell back to ``MemorySaver``.

    The composition root owns the handle and closes it from the lifespan;
    ``close`` is idempotent because shutdown paths are allowed to run twice
    (embedded hosts, test fixtures).
    """

    def __init__(self, saver: Any, close: Callable[[], None] | None = None) -> None:
        self.saver = saver
        self._close = close
        self._closed = False

    def close(self) -> None:
        """Release the backing connection, once. Errors propagate to the owner."""
        if self._closed:
            return
        self._closed = True
        closer = self._close
        if closer is not None:
            closer()


class _FieldCipherAdapter:
    """Adapt the platform's field cipher to LangGraph's ``CipherProtocol``.

    One key for everything the app writes (ADR 0027): the task store, the
    pending actions, the RAG cache and now the LangGraph checkpoints. The
    checkpoint holds the system prompt, the user's messages, the tool
    arguments and the tool results — the very material the other stores seal.
    Leaving it in the clear made ``Cache/`` the one plaintext copy the
    at-rest story had not closed (ADR 0032).
    """

    def __init__(self, cipher: FieldCipher) -> None:
        self._cipher = cipher

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        return (
            _CHECKPOINT_CIPHER_NAME,
            self._cipher.seal_bytes(plaintext, context=CHECKPOINT_CONTEXT),
        )

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        if ciphername != _CHECKPOINT_CIPHER_NAME:
            msg = f"checkpoint was sealed with an unknown cipher {ciphername!r}"
            raise ValueError(msg)
        return self._cipher.open_bytes(ciphertext, context=CHECKPOINT_CONTEXT)


def _serializer(cipher: FieldCipher | None) -> Any:
    """Return the LangGraph serializer for ``cipher``.

    ``None`` means "no key configured": the checkpoint is then stored in the
    clear, which is only acceptable for the in-memory saver.
    """
    if cipher is None:
        return None
    from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

    return EncryptedSerializer(_FieldCipherAdapter(cipher))


def _fallback_checkpoint_extension(path: Path) -> Path:
    """Fallback between .sql and .sqlite if the requested extension doesn't exist."""
    if not path.exists():
        if path.suffix == ".sqlite" and path.with_suffix(".sql").exists():
            return path.with_suffix(".sql")
        if path.suffix == ".sql" and path.with_suffix(".sqlite").exists():
            return path.with_suffix(".sqlite")
    return path


def _migrate_legacy_checkpoint(target: Path) -> None:
    """Migrate legacy project-relative checkpoint.sql to centralized cache directory."""
    if target.exists():
        return
    for legacy_cand in (
        Path("AIAssistant/Cache/checkpoint.sql"),
        Path("Cache/checkpoint.sql"),
        Path.cwd() / "Cache/checkpoint.sql",
        Path.cwd() / "AIAssistant/Cache/checkpoint.sql",
    ):
        if legacy_cand.is_file() and legacy_cand.resolve() != target.resolve():
            import shutil

            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(legacy_cand), str(target))
                for ext in ("-wal", "-shm"):
                    wal = legacy_cand.with_name(legacy_cand.name + ext)
                    if wal.exists():
                        shutil.move(str(wal), str(target.parent / (target.name + ext)))
                break
            except OSError:
                pass


def _resolve_app_data_checkpoint(p: Path, app_data_env: str) -> Path:
    """Resolve checkpoint path inside APP_DATA_DIR."""
    data_root = Path(app_data_env).resolve()
    if data_root.name.casefold() != "aiassistant":
        data_root = data_root / "AIAssistant"
    parts = p.parts
    if parts and parts[0].casefold() == "aiassistant":
        parts = parts[1:]
    resolved = data_root / Path(*parts)
    _migrate_legacy_checkpoint(resolved)
    return _fallback_checkpoint_extension(resolved)


def _resolve_checkpoint_path(raw_path: str) -> str:
    """Normalize checkpoint path so relative paths resolve to the right Cache directory.

    Avoids duplicate nesting when the process working directory is already AIAssistant
    and raw_path starts with 'AIAssistant/'.
    """
    if not raw_path:
        return "Cache/checkpoint.sql"

    p = Path(raw_path)
    if p.is_absolute() or os.path.isabs(raw_path):
        # Preserve the caller's spelling.  ``Path`` normalizes separators on
        # Windows, which is unnecessary for SQLite and makes diagnostics (and
        # mocks around the filesystem boundary) report a different path from
        # the one the deployment configured.
        return raw_path

    app_data_env = os.environ.get("APP_DATA_DIR")
    if app_data_env:
        return str(_resolve_app_data_checkpoint(p, app_data_env))

    cwd = Path.cwd()
    parts = p.parts

    if cwd.name == "AIAssistant" and len(parts) > 1 and parts[0] == "AIAssistant":
        resolved = Path(*parts[1:])
    elif cwd.name != "AIAssistant" and (cwd / "AIAssistant").is_dir() and len(parts) > 0 and parts[0] == "Cache":
        resolved = Path("AIAssistant") / p
    else:
        resolved = p

    return str(_fallback_checkpoint_extension(resolved))


def _ensure_parent_directory(path: str) -> None:
    """Create the checkpoint file's parent directory when the path has one.

    ``os.path.dirname("checkpoints.sqlite")`` is ``""`` and ``os.makedirs("")``
    raises, which used to turn a bare relative ``AGENT_CHECKPOINT_PATH`` into
    the silent ``MemorySaver`` fallback. A path with no directory component is
    relative to the working directory, which already exists.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _open_context_saver(factory: Callable[[str], Any], url: str) -> CheckpointerHandle:
    """Enter a saver factory's context manager and keep it open until close.

    ``PostgresSaver.from_conn_string`` and ``RedisSaver.from_conn_string`` are
    ``@contextmanager`` functions: calling one returns a generator context
    manager, not a saver. The manager must stay entered for the life of the
    handle (the connection is closed by its ``__exit__``), and a failing
    ``setup()`` must release it instead of leaking the connection.
    """
    stack = ExitStack()
    try:
        saver = stack.enter_context(factory(url))
        saver.setup()
    except BaseException:
        with suppress(Exception):
            stack.close()
        raise
    return CheckpointerHandle(saver, stack.close)


def _build_configured_saver(backend: str, url: str, path: str,
                            cipher: FieldCipher | None = None) -> CheckpointerHandle | None:
    """Build the explicitly configured saver, or ``None`` when unusable here.

    The SQLite backend — the one that lands in the app's own data directory —
    seals its blobs with the platform's at-rest key. The Postgres and Redis
    savers keep their published constructor signature: their
    ``from_conn_string`` factory takes no ``serde``, so a deployment that
    stores checkpoints in an external database protects them with the
    database's own encryption (the same split the runbook draws between
    field-level and volume-level encryption).
    """
    if backend == "postgres" and url:
        from langgraph.checkpoint.postgres import PostgresSaver

        handle = _open_context_saver(PostgresSaver.from_conn_string, url)
        logger.info("Using Postgres LangGraph checkpointer")
        return handle
    if backend == "redis" and url:
        from langgraph.checkpoint.redis import RedisSaver

        handle = _open_context_saver(RedisSaver.from_conn_string, url)
        logger.info("Using Redis LangGraph checkpointer")
        return handle
    if backend in ("sqlite", "default"):
        from langgraph.checkpoint.sqlite import SqliteSaver

        norm_path = _resolve_checkpoint_path(path)
        _ensure_parent_directory(norm_path)
        conn = sqlite3.connect(norm_path, check_same_thread=False)
        try:
            saver = SqliteSaver(conn, serde=_serializer(cipher))
            saver.setup()
        except BaseException:
            conn.close()
            raise
        logger.info("Using SQLite LangGraph checkpointer at %s", norm_path)
        return CheckpointerHandle(saver, conn.close)
    return None


def build_checkpointer(cipher: FieldCipher | None = None) -> CheckpointerHandle:
    """Return a handle around the configured saver, otherwise ``MemorySaver``.

    With ``AGENT_CHECKPOINT_REQUIRED=1`` (production gate, ADR 0023) the
    in-memory fallback is refused: a run that silently degrades to a
    process-local checkpointer is not durable, so startup fails instead.

    ``cipher`` is the platform's at-rest key. A durable backend that receives
    it seals the checkpoint blobs with the same key the task store uses, so
    the conversation state is not a second plaintext copy (ADR 0032).
    """
    runtime_settings = load_agent_runtime_settings()
    backend = runtime_settings.checkpoint_backend
    url = runtime_settings.checkpoint_url
    required = runtime_settings.checkpoint_required
    if required and backend == "memory":
        msg = ("AGENT_CHECKPOINT_REQUIRED=1 refuses the in-memory checkpointer; "
               "configure AGENT_CHECKPOINT_BACKEND=sqlite|postgres|redis.")
        raise RuntimeError(msg)
    try:
        handle = _build_configured_saver(
            backend, url, runtime_settings.checkpoint_path, cipher,
        )
        if handle is not None:
            return handle
    except Exception as error:
        if required:
            msg = (f"Checkpoint backend {backend!r} failed to initialize; "
                   "AGENT_CHECKPOINT_REQUIRED=1 refuses the memory fallback.")
            raise RuntimeError(msg) from error
        logger.exception("Checkpoint backend unavailable; using MemorySaver")
        return CheckpointerHandle(MemorySaver())
    if required:
        msg = (f"Checkpoint backend {backend!r} is not usable; "
               "AGENT_CHECKPOINT_REQUIRED=1 refuses the memory fallback.")
        raise RuntimeError(msg)
    if backend != "memory":
        logger.warning("Checkpoint backend %s is not configured; using MemorySaver", backend)
    return CheckpointerHandle(MemorySaver())


def cleanup_old_checkpoints(max_days: int = 30, max_size_mb: int = 50) -> None:
    """Prune inactive SQLite checkpoint threads and compact oversized stores.

    A thread is expired only when its newest verifiable UUIDv6 checkpoint is
    older than ``max_days``. Deleting whole threads preserves the checkpoint
    chain of every active session. Threads containing a legacy/custom id are
    retained fail-closed because their age cannot be proven from the schema.
    """
    runtime_settings = load_agent_runtime_settings()
    backend = runtime_settings.checkpoint_backend
    if backend not in ("sqlite", "default"):
        return

    db_path = _resolve_checkpoint_path(runtime_settings.checkpoint_path)
    if not os.path.exists(db_path):
        return

    try:
        stat = os.stat(db_path)
        size_mb = stat.st_size / (1024 * 1024)
        oversized = size_mb > max_size_mb
        if oversized:
            logger.info("Checkpoint database size (%.1fMB) exceeds limit (%.1fMB); running VACUUM", size_mb, max_size_mb)
        cutoff_unix = time.time() - max_days * _SECONDS_PER_DAY
        _vacuum_database(
            db_path,
            cutoff_unix=cutoff_unix,
            prune_to_recent_threads=oversized,
        )
    except Exception as error:
        logger.warning("Could not clean up checkpoints: %s", error)


def _uuid6_unix_seconds(checkpoint_id: str) -> float | None:
    """Return a LangGraph UUIDv6 timestamp, or ``None`` for an unknown id."""
    try:
        parsed = uuid.UUID(checkpoint_id)
    except (AttributeError, ValueError):
        return None
    if parsed.version != 6:
        return None
    timestamp_100ns = (
        (parsed.time_low << 28)
        | (parsed.time_mid << 12)
        | (parsed.time_hi_version & 0x0FFF)
    )
    return (timestamp_100ns - _UUID_EPOCH_OFFSET_100NS) / 10_000_000


def _expired_thread_ids(conn: sqlite3.Connection, cutoff_unix: float) -> set[str]:
    """Return threads whose latest checkpoint is provably before ``cutoff``."""
    latest_by_thread: dict[str, float] = {}
    unverifiable: set[str] = set()
    rows = conn.execute("SELECT thread_id, checkpoint_id FROM checkpoints")
    for thread_id, checkpoint_id in rows:
        checkpoint_time = _uuid6_unix_seconds(checkpoint_id)
        if checkpoint_time is None:
            unverifiable.add(thread_id)
            continue
        latest_by_thread[thread_id] = max(
            checkpoint_time,
            latest_by_thread.get(thread_id, checkpoint_time),
        )
    return {
        thread_id
        for thread_id, latest in latest_by_thread.items()
        if thread_id not in unverifiable and latest < cutoff_unix
    }


def _excess_thread_ids(conn: sqlite3.Connection, keep: int = 100) -> set[str]:
    """Return threads outside the newest ``keep`` when the size cap fires."""
    rows = conn.execute(
        "SELECT thread_id FROM checkpoints "
        "GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC LIMIT -1 OFFSET ?",
        (keep,),
    )
    return {row[0] for row in rows}


def _vacuum_database(
    db_path: str,
    *,
    cutoff_unix: float,
    prune_to_recent_threads: bool = False,
) -> None:
    """Prune checkpoint threads and compact when rows or the size cap require it.

    The connection is always closed, including when ``VACUUM`` fails (for
    example inside an implicit transaction): a leaked handle keeps the file
    locked on Windows.
    """
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(db_path)
        thread_ids: set[str] = set()
        try:
            thread_ids.update(_expired_thread_ids(conn, cutoff_unix))
            if prune_to_recent_threads:
                thread_ids.update(_excess_thread_ids(conn))
            if thread_ids:
                parameters = [(thread_id,) for thread_id in sorted(thread_ids)]
                conn.executemany("DELETE FROM writes WHERE thread_id = ?", parameters)
                conn.executemany("DELETE FROM checkpoints WHERE thread_id = ?", parameters)
                conn.commit()
        except sqlite3.OperationalError:
            conn.rollback()
        if thread_ids or prune_to_recent_threads:
            conn.execute("VACUUM")
    finally:
        if conn is not None:
            conn.close()


__all__ = [
    "CheckpointerHandle",
    "build_checkpointer",
    "cleanup_old_checkpoints",
]
