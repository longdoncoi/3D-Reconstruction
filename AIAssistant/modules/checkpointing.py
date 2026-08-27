"""Selectable LangGraph checkpoint backend with safe local fallback."""
from __future__ import annotations

import logging
import os
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


def build_checkpointer() -> Any:
    """Return Postgres/Redis saver when explicitly configured, otherwise memory.

    External services are opt-in so the desktop app remains self-contained.
    Required optional packages: langgraph-checkpoint-postgres or
    langgraph-checkpoint-redis.
    """
    backend = os.getenv("AGENT_CHECKPOINT_BACKEND", "memory").casefold()
    url = os.getenv("AGENT_CHECKPOINT_URL", "")
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
        if backend == "sqlite" or backend == "default":
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver
            db_path = os.getenv("AGENT_CHECKPOINT_PATH", "AIAssistant/Cache/checkpoints.sqlite")
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
    """Xóa các checkpoint cũ nếu dùng SQLite để tránh tràn bộ nhớ."""
    backend = os.getenv("AGENT_CHECKPOINT_BACKEND", "sqlite").casefold()
    if backend not in ("sqlite", "default"):
        return
        
    db_path = os.getenv("AGENT_CHECKPOINT_PATH", "AIAssistant/Cache/checkpoints.sqlite")
    if not os.path.exists(db_path):
        return
        
    import sqlite3
    try:
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        if size_mb > max_size_mb:
            logger.info("Checkpoint DB size (%.1fMB) exceeds limit %dMB, cleaning up...", size_mb, max_size_mb)
            with sqlite3.connect(db_path) as conn:
                # langgraph sqlite schema doesn't have an explicit created_at timestamp usually.
                # It uses thread_id and checkpoint_id (which is a timestamp-based UUID).
                # We can just vacuum or delete older records by limiting rows.
                # For simplicity, we just delete all but the last 100 threads if it gets too big.
                # For a rigorous implementation we can check the timestamp embedded in checkpoint_id.
                conn.execute("DELETE FROM checkpoints WHERE thread_id NOT IN (SELECT thread_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 100)")
                conn.execute("DELETE FROM writes WHERE thread_id NOT IN (SELECT thread_id FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 100)")
                conn.execute("VACUUM")
            logger.info("Cleanup done.")
    except Exception as e:
        logger.warning("Failed to cleanup checkpoints: %s", e)
