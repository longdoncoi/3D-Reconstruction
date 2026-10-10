"""Unit tests for the LangGraph checkpoint adapter.

Postgres/Redis checkpointer imports are hidden behind fakes; the memory and
sqlite backends are exercised for real (both are in the offline CI dependency
set), so no external services are contacted.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_assistant.adapters.persistence import checkpointing as cp_module
from ai_assistant.settings import AgentRuntimeSettings


def _settings(**overrides) -> AgentRuntimeSettings:
    base = dict(
        sandbox_enabled=True,
        sandbox_runtime="local",
        sandbox_image="python:3.11-slim",
        max_write_bytes=10,
        checkpoint_backend="memory",
        checkpoint_url="",
        checkpoint_path="",
        native_tool_calls=False,
    )
    base.update(overrides)
    return AgentRuntimeSettings(**base)


class BuildCheckpointerTests(unittest.TestCase):
    def _patch_settings(self, settings):
        return mock.patch.object(
            cp_module, "load_agent_runtime_settings", return_value=settings,
        )

    def test_memory_backend(self):
        with self._patch_settings(_settings()):
            saver = cp_module.build_checkpointer()
        from langgraph.checkpoint.memory import MemorySaver
        self.assertIsInstance(saver, MemorySaver)

    def test_postgres_requires_url(self):
        with self._patch_settings(_settings(checkpoint_backend="postgres")):
            saver = cp_module.build_checkpointer()
        from langgraph.checkpoint.memory import MemorySaver
        self.assertIsInstance(saver, MemorySaver)

    def test_postgres_backend(self):
        created: list = []

        class _PostgresSaver:
            def __init__(self):
                created.append(self)

            @classmethod
            def from_conn_string(cls, url):
                created.append(url)
                return cls()

            def setup(self):
                return None

        fake = SimpleNamespace(PostgresSaver=_PostgresSaver)
        settings = _settings(checkpoint_backend="postgres", checkpoint_url="postgres://x")
        with self._patch_settings(settings):
            with mock.patch.dict(sys.modules, {"langgraph.checkpoint.postgres": fake}):
                saver = cp_module.build_checkpointer()
        self.assertIsInstance(saver, _PostgresSaver)

    def test_redis_backend(self):
        class _RedisSaver:
            @classmethod
            def from_conn_string(cls, url):
                return cls()

            def setup(self):
                return None

        fake = SimpleNamespace(RedisSaver=_RedisSaver)
        settings = _settings(checkpoint_backend="redis", checkpoint_url="redis://x")
        with self._patch_settings(settings):
            with mock.patch.dict(sys.modules, {"langgraph.checkpoint.redis": fake}):
                saver = cp_module.build_checkpointer()
        self.assertIsInstance(saver, _RedisSaver)

    def test_sqlite_backend_creates_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "checkpoints.sqlite"
            settings = _settings(checkpoint_backend="sqlite", checkpoint_path=str(db))
            with self._patch_settings(settings):
                saver = cp_module.build_checkpointer()
            self.assertTrue(db.exists())
            # Release the sqlite connection so the temp dir can be removed on
            # Windows (open file handles block deletion).
            conn = getattr(saver, "conn", None)
            if conn is not None:
                conn.close()

    def test_unknown_backend_falls_back_with_warning(self):
        with self._patch_settings(_settings(checkpoint_backend="bogus")):
            saver = cp_module.build_checkpointer()
        from langgraph.checkpoint.memory import MemorySaver
        self.assertIsInstance(saver, MemorySaver)

    def test_backend_failure_falls_back_to_memory(self):
        def boom():
            raise RuntimeError("connection refused")

        settings = _settings(checkpoint_backend="postgres", checkpoint_url="postgres://x")
        with self._patch_settings(settings):
            with mock.patch.dict(sys.modules, {"langgraph.checkpoint.postgres": SimpleNamespace(PostgresSaver=boom)}):
                saver = cp_module.build_checkpointer()
        from langgraph.checkpoint.memory import MemorySaver
        self.assertIsInstance(saver, MemorySaver)


class CleanupOldCheckpointsTests(unittest.TestCase):
    def _patch_settings(self, settings):
        return mock.patch.object(
            cp_module, "load_agent_runtime_settings", return_value=settings,
        )

    def test_non_sqlite_backend_is_noop(self):
        with self._patch_settings(_settings(checkpoint_backend="memory")):
            cp_module.cleanup_old_checkpoints()  # must not raise

    def test_missing_database_is_noop(self):
        with self._patch_settings(_settings(checkpoint_backend="sqlite", checkpoint_path="nope.sqlite")):
            cp_module.cleanup_old_checkpoints()  # must not raise

    def test_small_database_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "small.sqlite"
            db.write_bytes(b"tiny")
            with self._patch_settings(_settings(checkpoint_backend="sqlite", checkpoint_path=str(db))):
                cp_module.cleanup_old_checkpoints(max_size_mb=50)
            self.assertEqual(db.read_bytes(), b"tiny")

    def test_oversized_database_is_vacuumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "big.sqlite"
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_id TEXT)")
            conn.execute("CREATE TABLE writes (thread_id TEXT, checkpoint_id TEXT)")
            for i in range(150):
                conn.execute(
                    "INSERT INTO checkpoints VALUES (?, ?)",
                    (f"t{i:03d}", f"c{i:03d}"),
                )
                conn.execute(
                    "INSERT INTO writes VALUES (?, ?)",
                    (f"t{i:03d}", f"c{i:03d}"),
                )
            conn.commit()
            conn.close()
            with self._patch_settings(
                _settings(checkpoint_backend="sqlite", checkpoint_path=str(db))
            ):
                cp_module.cleanup_old_checkpoints(max_size_mb=0)
            # The VACUUM path prunes non-recent threads; the DB remains valid.
            check_conn = sqlite3.connect(db)
            try:
                remaining = check_conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
            finally:
                check_conn.close()
            self.assertEqual(remaining, 100)


if __name__ == "__main__":
    unittest.main()
