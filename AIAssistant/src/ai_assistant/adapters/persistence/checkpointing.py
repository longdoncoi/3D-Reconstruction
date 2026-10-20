"""Selectable LangGraph checkpoint backend with safe local fallback."""
from __future__ import annotations

import logging
import os
import sqlite3
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


def _build_configured_saver(backend: str, url: str, path: str,
                            cipher: FieldCipher | None = None) -> Any | None:
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

        saver = PostgresSaver.from_conn_string(url)
        saver.setup()
        logger.info("Using Postgres LangGraph checkpointer")
        return saver
    if backend == "redis" and url:
        from langgraph.checkpoint.redis import RedisSaver

        saver = RedisSaver.from_conn_string(url)
        saver.setup()
        logger.info("Using Redis LangGraph checkpointer")
        return saver
    if backend in ("sqlite", "default"):
        from langgraph.checkpoint.sqlite import SqliteSaver

        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        saver = SqliteSaver(conn, serde=_serializer(cipher))
        saver.setup()
        logger.info("Using SQLite LangGraph checkpointer at %s", path)
        return saver
    return None


def build_checkpointer(cipher: FieldCipher | None = None) -> Any:
    """Return Postgres/Redis saver when explicitly configured, otherwise memory.

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
        saver = _build_configured_saver(
            backend, url, runtime_settings.checkpoint_path, cipher,
        )
        if saver is not None:
            return saver
    except Exception as error:
        if required:
            msg = (f"Checkpoint backend {backend!r} failed to initialize; "
                   "AGENT_CHECKPOINT_REQUIRED=1 refuses the memory fallback.")
            raise RuntimeError(msg) from error
        logger.exception("Checkpoint backend unavailable; using MemorySaver")
        return MemorySaver()
    if required:
        msg = (f"Checkpoint backend {backend!r} is not usable; "
               "AGENT_CHECKPOINT_REQUIRED=1 refuses the memory fallback.")
        raise RuntimeError(msg)
    if backend != "memory":
        logger.warning("Checkpoint backend %s is not configured; using MemorySaver", backend)
    return MemorySaver()


def cleanup_old_checkpoints(max_days: int = 30, max_size_mb: int = 50) -> None:
    """Clean up old checkpoints when using SQLite."""
    runtime_settings = load_agent_runtime_settings()
    backend = runtime_settings.checkpoint_backend
    if backend not in ("sqlite", "default"):
        return

    db_path = runtime_settings.checkpoint_path
    if not os.path.exists(db_path):
        return

    try:
        stat = os.stat(db_path)
        size_mb = stat.st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            logger.info("Checkpoint database size (%.1fMB) exceeds limit (%.1fMB); running VACUUM", size_mb, max_size_mb)
            _vacuum_database(db_path)
    except Exception as error:
        logger.warning("Could not clean up checkpoints: %s", error)


def _vacuum_database(db_path: str) -> None:
    """Prune old checkpoint rows and compact the SQLite file.

    The connection is always closed, including when ``VACUUM`` fails (for
    example inside an implicit transaction): a leaked handle keeps the file
    locked on Windows.
    """
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("DELETE FROM checkpoints WHERE thread_id NOT IN (SELECT thread_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 100)")
            conn.execute("DELETE FROM writes WHERE thread_id NOT IN (SELECT thread_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 100)")
            conn.commit()
        except sqlite3.OperationalError:
            conn.rollback()
        conn.execute("VACUUM")
    finally:
        if conn is not None:
            conn.close()


__all__ = [
    "build_checkpointer",
    "cleanup_old_checkpoints",
]
