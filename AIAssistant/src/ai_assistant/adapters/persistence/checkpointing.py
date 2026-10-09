"""Selectable LangGraph checkpoint backend with safe local fallback."""
from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from ai_assistant.settings import load_agent_runtime_settings

logger = logging.getLogger(__name__)


def build_checkpointer() -> Any:
    """Return Postgres/Redis saver when explicitly configured, otherwise memory."""
    runtime_settings = load_agent_runtime_settings()
    backend = runtime_settings.checkpoint_backend
    url = runtime_settings.checkpoint_url
    try:
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

            db_path = runtime_settings.checkpoint_path
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            conn = sqlite3.connect(db_path, check_same_thread=False)
            saver = SqliteSaver(conn)
            saver.setup()
            logger.info("Using SQLite LangGraph checkpointer at %s", db_path)
            return saver
        if backend != "memory":
            logger.warning("Checkpoint backend %s is not configured; using MemorySaver", backend)
    except Exception as error:  # noqa: BLE001
        logger.exception("Checkpoint backend unavailable; using MemorySaver: %s", error)
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
            with sqlite3.connect(db_path) as conn:
                try:
                    conn.execute("DELETE FROM checkpoints WHERE thread_id NOT IN (SELECT thread_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 100)")
                    conn.execute("DELETE FROM writes WHERE thread_id NOT IN (SELECT thread_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 100)")
                except sqlite3.OperationalError:
                    pass
                conn.execute("VACUUM")
    except Exception as error:  # noqa: BLE001
        logger.warning("Could not clean up checkpoints: %s", error)


__all__ = [
    "build_checkpointer",
    "cleanup_old_checkpoints",
]
