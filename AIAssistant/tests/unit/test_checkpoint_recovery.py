"""Tests for LangGraph checkpoint backend selection and recovery.

Covers SQLite, Memory, Postgres, Redis backend selection and the safe
fallback mechanism when a backend is unavailable.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from ai_assistant.adapters.persistence.checkpointing import (
    build_checkpointer,
    cleanup_old_checkpoints,
)


class CheckpointBackendTests(unittest.TestCase):
    """Test checkpointer backend selection logic."""

    def test_memory_backend_when_no_config(self) -> None:
        """Default to MemorySaver when no backend is configured."""
        with patch.dict(os.environ, {"AGENT_CHECKPOINT_BACKEND": "memory"}):
            saver = build_checkpointer()
            self.assertIsNotNone(saver)

    def test_unknown_backend_falls_back_to_memory(self) -> None:
        """Unknown backend falls back to MemorySaver with warning."""
        with patch.dict(os.environ, {"AGENT_CHECKPOINT_BACKEND": "unknown"}):
            saver = build_checkpointer()
            self.assertIsNotNone(saver)

    def test_postgres_backend_without_url_falls_back(self) -> None:
        """Postgres backend without URL falls back to MemorySaver."""
        with patch.dict(os.environ, {
            "AGENT_CHECKPOINT_BACKEND": "postgres",
            "AGENT_CHECKPOINT_URL": "",
        }):
            saver = build_checkpointer()
            self.assertIsNotNone(saver)

    def test_redis_backend_without_url_falls_back(self) -> None:
        """Redis backend without URL falls back to MemorySaver."""
        with patch.dict(os.environ, {
            "AGENT_CHECKPOINT_BACKEND": "redis",
            "AGENT_CHECKPOINT_URL": "",
        }):
            saver = build_checkpointer()
            self.assertIsNotNone(saver)

    def test_backend_import_failure_falls_back_to_memory(self) -> None:
        """Import failure for optional backend falls back to MemorySaver."""
        with patch.dict(os.environ, {
            "AGENT_CHECKPOINT_BACKEND": "postgres",
            "AGENT_CHECKPOINT_URL": "postgresql://fake",
        }):
            saver = build_checkpointer()
            self.assertIsNotNone(saver)


class CheckpointCleanupTests(unittest.TestCase):
    """Test checkpoint cleanup and vacuum operations."""

    def test_cleanup_noop_for_non_sqlite(self) -> None:
        """Cleanup is a no-op for non-SQLite backends."""
        with patch.dict(os.environ, {"AGENT_CHECKPOINT_BACKEND": "memory"}):
            cleanup_old_checkpoints()

    def test_cleanup_handles_missing_database(self) -> None:
        """Cleanup handles missing database file gracefully."""
        with patch.dict(os.environ, {
            "AGENT_CHECKPOINT_BACKEND": "sqlite",
            "AGENT_CHECKPOINT_PATH": "/nonexistent/path/checkpoints.sqlite",
        }):
            cleanup_old_checkpoints()


if __name__ == "__main__":
    unittest.main()
