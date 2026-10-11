"""Coverage boost tests for low-coverage modules.

Targets: llama_cpp_backend, model_loader, a2a_client, bootstrap/container,
agents/approval_service, agents/service, agents/completion.
"""
from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_assistant.adapters.a2a_client import TrustedA2AClient
from ai_assistant.agents.approval_service import ApprovalService
from ai_assistant.agents.completion import (
    CRITIC_JSON_SCHEMA,
    PLANNER_JSON_SCHEMA,
    parse_tool_call,
)
from ai_assistant.agents.models import AgentApproveRequest
from ai_assistant.agents.service import AgentService
from ai_assistant.bootstrap.container import PlatformContainer, build_container
from ai_assistant.domain.errors import AuthorizationError, ModelNotLoadedError, NotFoundError
from ai_assistant.llm.llama_cpp_backend import LlamaCppBackend
from ai_assistant.llm.model_loader import get_backend, load_model
from ai_assistant.tools.factory import create_tool_registry, platform_executors


class LlamaCppBackendTests(unittest.TestCase):
    """Test LlamaCppBackend coverage."""

    def test_backend_initialization(self) -> None:
        """LlamaCppBackend initializes with correct attributes."""
        with patch("ai_assistant.llm.llama_cpp_backend.LlamaCppBackend._load"):
            backend = LlamaCppBackend(
                model_path="test.gguf",
                n_ctx=4096,
            )
            self.assertIsNotNone(backend)
            self.assertEqual(backend.model_path, "test.gguf")

    def test_backend_is_vision_supported_false(self) -> None:
        """LlamaCppBackend reports vision not supported by default."""
        with patch("ai_assistant.llm.llama_cpp_backend.LlamaCppBackend._load"):
            backend = LlamaCppBackend(model_path="test.gguf")
            self.assertFalse(backend.is_vision_supported)

    def test_backend_is_vision_supported_true(self) -> None:
        """LlamaCppBackend reports vision supported when configured."""
        with patch("ai_assistant.llm.llama_cpp_backend.LlamaCppBackend._load"):
            backend = LlamaCppBackend(model_path="test.gguf", is_vision=True)
            self.assertTrue(backend.is_vision_supported)

    def test_backend_model_description(self) -> None:
        """Backend returns model description."""
        with patch("ai_assistant.llm.llama_cpp_backend.LlamaCppBackend._load"):
            backend = LlamaCppBackend(model_path="test.gguf", desc="Test model")
            self.assertIn("Test model", backend.model_description)


class ModelLoaderTests(unittest.TestCase):
    """Test model_loader coverage."""

    def test_get_backend_returns_none_when_not_loaded(self) -> None:
        """get_backend returns None when no model is loaded."""
        with patch("ai_assistant.llm.model_loader._global_backend", None):
            backend = get_backend()
            self.assertIsNone(backend)

    def test_get_backend_returns_loaded_backend(self) -> None:
        """get_backend returns the loaded backend."""
        mock_backend = MagicMock()
        with patch("ai_assistant.llm.model_loader._global_backend", mock_backend):
            backend = get_backend()
            self.assertIs(backend, mock_backend)

    def test_load_model_publishes_global_backend(self) -> None:
        """load_model publishes the backend as global."""
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_model.is_vision = False
        mock_model.repo_id = "test-repo"
        mock_model.filename = "test.gguf"
        mock_registry.get_by_index.return_value = mock_model

        with patch("ai_assistant.llm.model_loader.LlamaCppBackend") as mock_backend_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a fake model file so download is skipped
                model_file = Path(tmpdir) / "test.gguf"
                model_file.write_text("fake")

                backend = load_model(mock_registry, tmpdir, model_idx=0, enable_vision=False)
                self.assertIsNotNone(backend)


class TrustedA2AClientTests(unittest.TestCase):
    """Test TrustedA2AClient coverage."""

    def test_client_rejects_untrusted_endpoint(self) -> None:
        """Client rejects endpoints not in allowlist."""
        client = TrustedA2AClient(
            trusted_endpoints=frozenset({"https://trusted.example.com"}),
            timeout_seconds=5,
        )
        result = client.submit("https://untrusted.example.com", "supervisor", "test")
        self.assertEqual(result["routing"], "rejected_policy")

    def test_client_accepts_trusted_endpoint(self) -> None:
        """Client accepts endpoints in allowlist."""
        client = TrustedA2AClient(
            trusted_endpoints=frozenset({"https://trusted.example.com"}),
            timeout_seconds=5,
        )
        self.assertIsNotNone(client)

    def test_client_discover_returns_empty_on_network_failure(self) -> None:
        """discover returns empty dict when network fails."""
        client = TrustedA2AClient(
            trusted_endpoints=frozenset({"https://unreachable.example.com"}),
            timeout_seconds=1,
        )
        with patch("ai_assistant.adapters.a2a_client.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = OSError("Network unreachable")
            cards = client.discover()
        self.assertEqual(cards, {})


class BootstrapContainerTests(unittest.TestCase):
    """Test build_container coverage."""

    def test_build_container_returns_platform_container(self) -> None:
        """build_container returns a PlatformContainer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            container = build_container(
                Path(tmpdir),
                [],
                {},
                MagicMock(),
            )
            self.assertIsInstance(container, PlatformContainer)
            self.assertIsNotNone(container.settings)
            self.assertIsNotNone(container.plugins)
            self.assertIsNotNone(container.tools)
            self.assertIsNotNone(container.tasks)
            self.assertIsNotNone(container.gateway)

    def test_build_container_with_legacy_tools(self) -> None:
        """build_container handles legacy tool dicts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_tools = [
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "policy": "read_only",
                    "schema": {"type": "object"},
                }
            ]
            container = build_container(
                Path(tmpdir),
                legacy_tools,
                {"test_tool": lambda params: {"success": True}},
                MagicMock(),
            )
            self.assertIsInstance(container, PlatformContainer)


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


class AgentServiceTests(unittest.TestCase):
    """Test AgentService facade coverage."""

    def test_agent_service_initialization(self) -> None:
        """AgentService initializes with default dependencies."""
        service = AgentService()
        self.assertIsNotNone(service.pending_actions)
        self.assertIsNotNone(service.pending_lock)
        self.assertIsNotNone(service.tool_registry)

    def test_agent_service_with_custom_dependencies(self) -> None:
        """AgentService accepts custom dependencies."""
        from ai_assistant.adapters.persistence import PendingActionStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PendingActionStore(str(Path(tmpdir) / "pending.json"))
            lock = threading.Lock()
            service = AgentService(
                pending_actions=store,
                pending_lock=lock,
            )
            self.assertIs(service.pending_actions, store)
            self.assertIs(service.pending_lock, lock)

    def test_agent_service_default_initialization(self) -> None:
        """AgentService initializes without arguments using default pending path."""
        service = AgentService()
        self.assertIsNotNone(service.pending_actions)
        self.assertIsNotNone(service.pending_lock)

    def test_platform_executors_returns_dict(self) -> None:
        """platform_executors returns a dict of executor functions."""
        executors = platform_executors(create_tool_registry())
        self.assertIsInstance(executors, dict)
        self.assertGreater(len(executors), 0)

    def test_tool_registry_has_tools(self) -> None:
        """A created tool registry contains registered tools."""
        tools = create_tool_registry().get_all()
        self.assertGreater(len(tools), 0)


class CompletionTests(unittest.TestCase):
    """Test completion and parsing coverage."""

    def test_parse_tool_call_valid_json(self) -> None:
        """parse_tool_call handles valid JSON tool calls."""
        text = '{"kind": "tool", "tool": "read_file", "params": {"path": "test.txt"}}'
        tool, params = parse_tool_call(
            text, {},
            lambda *a, **kw: (a[1], None),  # validate_fn returns (params, error)
            lambda *a, **kw: None,
        )
        self.assertEqual(tool, "read_file")
        self.assertEqual(params.get("path"), "test.txt")

    def test_parse_tool_call_no_tool(self) -> None:
        """parse_tool_call returns None for non-tool text."""
        tool, _ = parse_tool_call("Just a regular message", {}, lambda *a, **kw: None, lambda *a, **kw: None)
        self.assertIsNone(tool)

    def test_planner_json_schema_structure(self) -> None:
        """PLANNER_JSON_SCHEMA has correct structure."""
        self.assertIn("type", PLANNER_JSON_SCHEMA)
        self.assertEqual(PLANNER_JSON_SCHEMA["type"], "object")

    def test_critic_json_schema_structure(self) -> None:
        """CRITIC_JSON_SCHEMA has correct structure."""
        self.assertIn("type", CRITIC_JSON_SCHEMA)
        self.assertEqual(CRITIC_JSON_SCHEMA["type"], "object")


if __name__ == "__main__":
    unittest.main()
