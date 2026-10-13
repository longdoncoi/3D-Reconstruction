"""Sprint 3 expanded: Postgres adapter for task/checkpoint durability (replica-ready)."""
from __future__ import annotations

import json
from typing import Any

try:
    import psycopg2
except ImportError:  # pragma: no cover
    psycopg2 = None  # type: ignore

class PostgresTaskStore:
    def __init__(self, dsn: str = "postgresql://localhost/agent") -> None:
        self.dsn = dsn
        if psycopg2 is None:
            raise ImportError("bad_param")
        self._conn = psycopg2.connect(dsn)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    id TEXT PRIMARY KEY,
                    capability TEXT,
                    message TEXT,
                    status TEXT,
                    result JSONB,
                    metadata JSONB,
                    classification TEXT,
                    created_at DOUBLE PRECISION,
                    updated_at DOUBLE PRECISION
                )
            """)
            self._conn.commit()

    def save(self, task: Any) -> None:
        with self._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO agent_tasks (id, capability, message, status, result, metadata, classification, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    capability = EXCLUDED.capability,
                    message = EXCLUDED.message,
                    status = EXCLUDED.status,
                    result = EXCLUDED.result,
                    metadata = EXCLUDED.metadata,
                    classification = EXCLUDED.classification,
                    updated_at = EXCLUDED.updated_at
            """, (
                task.id, task.capability, task.message, task.status,
                json.dumps(task.result) if task.result else None,
                json.dumps(task.metadata) if task.metadata else None,
                task.classification.value if hasattr(task.classification, "value") else str(task.classification),
                getattr(task, "created_at", 0), getattr(task, "updated_at", 0),
            ))
            self._conn.commit()

    def get(self, task_id: str) -> Any | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT id, capability, message, status, result, metadata FROM agent_tasks WHERE id = %s", (task_id,))
            row = cur.fetchone()
            return row
