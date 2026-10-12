"""Guard-path coverage for the approval service.

``ApprovalService.approve`` is the only path that may mint a single-use
approval grant, so every rejection it can produce is a security boundary:
an unknown action, an unloaded model, a session mismatch, a missing session
binding (fail closed, action restored) and the explicit refusal that ends
the run. (Moved verbatim from ``test_coverage_boost.py`` during the
coverage-file split.)
"""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from ai_assistant.agents.approval_service import ApprovalService
from ai_assistant.agents.models import AgentApproveRequest
from ai_assistant.domain.errors import (
    AuthorizationError,
    ModelNotLoadedError,
    NotFoundError,
)


class ApprovalServiceTests(unittest.TestCase):
    """Test ApprovalService coverage."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        from ai_assistant.adapters.persistence import PendingActionStore
        self.store = PendingActionStore(str(Path(self.temp_dir.name) / "pending.json"))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_approve_missing_action_raises_not_found(self) -> None:
        """Approving a non-existent action raises NotFoundError."""
        from ai_assistant.llm.contracts import LLMBackend
        mock_llm = MagicMock(spec=LLMBackend)
        mock_llm.llm = MagicMock()  # LLM is loaded

        service = ApprovalService(
            pending_actions=self.store,
            llm_runtime=mock_llm,
        )
        with self.assertRaises(NotFoundError):
            service.approve(AgentApproveRequest(action_id="nonexistent", approved=True))

    def test_approve_without_llm_raises_model_not_loaded(self) -> None:
        """Approving without LLM loaded raises ModelNotLoadedError."""
        from ai_assistant.llm.contracts import LLMBackend
        mock_llm = MagicMock(spec=LLMBackend)
        mock_llm.llm = None  # LLM not loaded

        self.store["action-1"] = {
            "tool": "read_file",
            "params": {"path": "test.txt"},
            "session_id": "session-1",
            "task": "test",
            "messages": [],
            "steps": [],
            "iteration": 0,
            "temperature": 0.3,
            "language": "vi",
            "created_at": time.time(),
        }
        self.store.save()

        service = ApprovalService(
            pending_actions=self.store,
            llm_runtime=mock_llm,
        )
        with self.assertRaises(ModelNotLoadedError):
            service.approve(AgentApproveRequest(action_id="action-1", approved=True))

    def test_approve_session_mismatch_raises_authorization_error(self) -> None:
        """Approving with wrong session_id raises AuthorizationError."""
        from ai_assistant.llm.contracts import LLMBackend
        mock_llm = MagicMock(spec=LLMBackend)
        mock_llm.llm = MagicMock()  # LLM is loaded

        # Use a unique action_id, session_id, and fresh store to avoid test
        # pollution (other tests cancel "session-1" on the global coordinator).
        unique_id = f"action-mismatch-{time.time()}"
        unique_session = f"session-mismatch-{time.time()}"
        with tempfile.TemporaryDirectory() as tmpdir:
            from ai_assistant.adapters.persistence import PendingActionStore
            store = PendingActionStore(str(Path(tmpdir) / "pending.json"))
            store[unique_id] = {
                "tool": "read_file",
                "params": {"path": "test.txt"},
                "session_id": unique_session,
                "task": "test",
                "messages": [],
                "steps": [],
                "iteration": 0,
                "temperature": 0.3,
                "language": "vi",
                "created_at": time.time(),
            }
            store.save()

            service = ApprovalService(
                pending_actions=store,
                llm_runtime=mock_llm,
            )
            with self.assertRaises(AuthorizationError):
                service.approve(AgentApproveRequest(
                    action_id=unique_id,
                    approved=True,
                    session_id="wrong-session",
                ))

    def test_approve_rejected_returns_rejected_status(self) -> None:
        """Rejecting an action returns rejected status with steps."""
        from ai_assistant.llm.contracts import LLMBackend
        mock_llm = MagicMock(spec=LLMBackend)
        mock_llm.llm = MagicMock()

        unique_id = f"action-reject-{time.time()}"
        unique_session = f"session-reject-{time.time()}"
        with tempfile.TemporaryDirectory() as tmpdir:
            from ai_assistant.adapters.persistence import PendingActionStore
            store = PendingActionStore(str(Path(tmpdir) / "pending.json"))
            store[unique_id] = {
                "tool": "read_file",
                "params": {"path": "test.txt"},
                "session_id": unique_session,
                "task": "test",
                "messages": [],
                "steps": [{"type": "tool_call", "tool": "read_file", "iteration": 0}],
                "iteration": 0,
                "temperature": 0.3,
                "language": "vi",
                "created_at": time.time(),
            }
            store.save()

            service = ApprovalService(
                pending_actions=store,
                llm_runtime=mock_llm,
            )
            result = service.approve(AgentApproveRequest(
                action_id=unique_id,
                approved=False,
                session_id=unique_session,
            ))
            self.assertEqual(result["status"], "rejected")
            self.assertIn("steps", result)

    def test_approve_missing_session_id_raises_authorization_error(self) -> None:
        """An absent session id must fail closed, not bypass the binding."""
        from ai_assistant.llm.contracts import LLMBackend
        mock_llm = MagicMock(spec=LLMBackend)
        mock_llm.llm = MagicMock()

        unique_id = f"action-nosession-{time.time()}"
        unique_session = f"session-nosession-{time.time()}"
        with tempfile.TemporaryDirectory() as tmpdir:
            from ai_assistant.adapters.persistence import PendingActionStore
            store = PendingActionStore(str(Path(tmpdir) / "pending.json"))
            store[unique_id] = {
                "tool": "read_file",
                "params": {"path": "test.txt"},
                "session_id": unique_session,
                "task": "test",
                "messages": [],
                "steps": [],
                "iteration": 0,
                "temperature": 0.3,
                "language": "vi",
                "created_at": time.time(),
            }
            store.save()

            service = ApprovalService(
                pending_actions=store,
                llm_runtime=mock_llm,
            )
            with self.assertRaises(AuthorizationError):
                service.approve(AgentApproveRequest(action_id=unique_id, approved=True))
            # The rejected attempt must put the action back for its real owner.
            self.assertIn(unique_id, store)


if __name__ == "__main__":
    unittest.main()
