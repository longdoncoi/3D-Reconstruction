"""Tests for concurrent approval handling and thread safety.

Covers edge cases where multiple approval requests arrive simultaneously,
ensuring the PendingActionStore and locking mechanisms work correctly.
"""
from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from ai_assistant.adapters.persistence import PendingActionStore
from ai_assistant.domain.security import Principal


class ConcurrentApprovalTests(unittest.TestCase):
    """Test thread-safe operations on PendingActionStore."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = PendingActionStore(str(Path(self.temp_dir.name) / "pending.json"))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_concurrent_pop_returns_unique_actions(self) -> None:
        """Two threads popping the same action_id must not both succeed."""
        self.store["action-1"] = {"tool": "read_file", "params": {}, "session_id": "s1"}
        self.store.save()

        results: list[dict | None] = []
        lock = threading.Lock()

        def pop_action() -> None:
            action = self.store.pop("action-1", None)
            with lock:
                results.append(action)

        threads = [threading.Thread(target=pop_action) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one thread should get the action
        non_none = [r for r in results if r is not None]
        self.assertEqual(len(non_none), 1, "Only one thread should pop the action")

    def test_concurrent_save_and_cleanup(self) -> None:
        """Cleanup during concurrent saves must not corrupt the store."""
        for i in range(10):
            self.store[f"action-{i}"] = {
                "tool": "read_file",
                "params": {},
                "session_id": f"s{i}",
                "created_at": time.time() - (i * 100),  # Varying ages
            }
        self.store.save()

        errors: list[Exception] = []

        def cleanup_worker() -> None:
            try:
                self.store.cleanup(time.time() - 250)  # Remove actions older than 250s
            except Exception as e:
                errors.append(e)

        def save_worker() -> None:
            try:
                self.store.save()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=cleanup_worker),
            threading.Thread(target=save_worker),
            threading.Thread(target=cleanup_worker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Concurrent operations raised errors: {errors}")

    def test_pop_nonexistent_action_raises_not_found(self) -> None:
        """Popping a non-existent action returns None (caller decides to raise)."""
        result = self.store.pop("nonexistent", None)
        self.assertIsNone(result)

    def test_concurrent_different_actions(self) -> None:
        """Concurrent pops of different action IDs must all succeed."""
        for i in range(5):
            self.store[f"action-{i}"] = {"tool": "read_file", "params": {}, "session_id": f"s{i}"}
        self.store.save()

        results: dict[str, dict] = {}
        lock = threading.Lock()

        def pop_action(action_id: str) -> None:
            action = self.store.pop(action_id, None)
            if action is not None:
                with lock:
                    results[action_id] = action

        threads = [threading.Thread(target=pop_action, args=(f"action-{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 5, "All 5 actions should be popped")


class PrincipalBindingTests(unittest.TestCase):
    """Test Principal.permits() with various scope configurations."""

    def test_principal_with_no_scopes_cannot_access_scoped_tool(self) -> None:
        principal = Principal(subject="test", scopes=frozenset())
        self.assertFalse(principal.permits("project.read"))

    def test_principal_with_matching_scope_can_access(self) -> None:
        principal = Principal(subject="test", scopes=frozenset({"project.read"}))
        self.assertTrue(principal.permits("project.read"))

    def test_principal_with_none_scope_always_permitted(self) -> None:
        principal = Principal(subject="test", scopes=frozenset())
        self.assertTrue(principal.permits(None))

    def test_principal_with_multiple_scopes(self) -> None:
        principal = Principal(
            subject="test",
            scopes=frozenset({"project.read", "project.write", "desktop.action"}),
        )
        self.assertTrue(principal.permits("project.read"))
        self.assertTrue(principal.permits("project.write"))
        self.assertTrue(principal.permits("desktop.action"))
        self.assertFalse(principal.permits("project.execute"))


if __name__ == "__main__":
    unittest.main()
