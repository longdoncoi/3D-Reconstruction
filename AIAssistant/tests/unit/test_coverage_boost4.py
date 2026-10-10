"""Fourth coverage batch: model loader singleton lifecycle and A2A protocol routing.

Both modules touch process-global state, so tests snapshot and restore it to
keep the suite order-independent.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_assistant.adapters import a2a_protocol
from ai_assistant.adapters.a2a_protocol import (
    A2ARouter,
    AgentCard,
    AgentSkill,
    RemoteAgent,
    a2a_available,
    build_agent_card,
    discover_remote_agents,
    get_remote_registry,
)
from ai_assistant.config.model_registry import ModelDefinition, ModelRegistry
from ai_assistant.llm import model_loader


class ModelLoaderTests(unittest.TestCase):
    """Coverage for download/load/reload singleton lifecycle."""

    def setUp(self) -> None:
        self._original = model_loader._global_backend

    def tearDown(self) -> None:
        model_loader._global_backend = self._original

    def _registry(self, *, vision: bool = False) -> ModelRegistry:
        primary = ModelDefinition(
            "repo/primary", "primary.gguf", "Primary", is_vision=vision,
            mmproj_repo_id="repo/mmproj" if vision else None,
            mmproj_filename="mmproj.gguf" if vision else None,
        )
        fallback = ModelDefinition("repo/fallback", "fallback.gguf", "Fallback")
        return ModelRegistry(models=(primary,), fallback=fallback)

    def test_download_if_missing_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "m.gguf").write_text("weights", encoding="utf-8")
            path = model_loader.download_if_missing(ModelDefinition("repo", "m.gguf", "desc"), tmpdir)
        self.assertEqual(path, os.path.join(tmpdir, "m.gguf"))

    def test_download_if_missing_triggers_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "huggingface_hub.hf_hub_download"
        ) as mock_download:
            model_loader.download_if_missing(ModelDefinition("repo", "m.gguf", "desc"), tmpdir)
        mock_download.assert_called_once()

    def test_load_model_publishes_backend(self) -> None:
        registry = self._registry()
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            model_loader, "download_if_missing", return_value="primary.gguf"
        ), patch.object(model_loader, "LlamaCppBackend") as backend_cls, patch.object(
            model_loader, "release_ml_memory"
        ):
            backend = model_loader.load_model(registry, tmpdir, model_idx=0, enable_vision=True)

        self.assertIs(model_loader.get_backend(), backend)
        backend_cls.assert_called_once()

    def test_load_model_vision_disabled_uses_fallback(self) -> None:
        registry = self._registry(vision=True)
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            model_loader, "download_if_missing", return_value="fallback.gguf"
        ), patch.object(model_loader, "LlamaCppBackend") as backend_cls, patch.object(
            model_loader, "release_ml_memory"
        ):
            model_loader.load_model(registry, tmpdir, enable_vision=False)

        self.assertFalse(backend_cls.call_args.kwargs["is_vision"])
        self.assertEqual(backend_cls.call_args.kwargs["desc"], "Fallback")

    def test_load_model_replaces_existing_backend(self) -> None:
        model_loader._global_backend = MagicMock()
        registry = self._registry()
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            model_loader, "download_if_missing", return_value="primary.gguf"
        ), patch.object(model_loader, "LlamaCppBackend"), patch.object(
            model_loader, "release_ml_memory"
        ) as mock_release:
            model_loader.load_model(registry, tmpdir)
        mock_release.assert_called()

    def test_load_model_downloads_mmproj_for_vision(self) -> None:
        registry = self._registry(vision=True)
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            model_loader, "download_if_missing", return_value="primary.gguf"
        ), patch.object(model_loader, "LlamaCppBackend") as backend_cls, patch.object(
            model_loader, "release_ml_memory"
        ), patch("huggingface_hub.hf_hub_download") as mock_download:
            model_loader.load_model(registry, tmpdir, enable_vision=True)

        mock_download.assert_called_once()
        self.assertTrue(backend_cls.call_args.kwargs["is_vision"])

    def test_reload_model_without_backend_raises(self) -> None:
        model_loader._global_backend = None
        with self.assertRaises(RuntimeError):
            model_loader.reload_model(self._registry(), "models_dir")

    def test_reload_model_delegates(self) -> None:
        model_loader._global_backend = MagicMock()
        with patch.object(model_loader, "load_model", return_value="new-backend") as mock_load:
            result = model_loader.reload_model(self._registry(), "models_dir")
        self.assertEqual(result, "new-backend")
        mock_load.assert_called_once()

    def test_get_backend_initially_none(self) -> None:
        model_loader._global_backend = None
        self.assertIsNone(model_loader.get_backend())


class AgentCardTests(unittest.TestCase):
    """Coverage for Agent Card construction and serialisation."""

    def test_build_agent_card_has_skills(self) -> None:
        card = build_agent_card(url="https://example.com/a2a")
        self.assertIsInstance(card, AgentCard)
        self.assertEqual(card.url, "https://example.com/a2a")
        self.assertGreater(len(card.skills), 0)
        self.assertIsInstance(card.skills[0], AgentSkill)

    def test_agent_card_serialisation_round_trip(self) -> None:
        card = AgentCard(
            name="Test", description="d", url="https://x",
            skills=[AgentSkill(id="s", name="S", description="desc", tags=["t"])],
        )
        payload = json.loads(card.to_json())
        self.assertEqual(payload["name"], "Test")
        self.assertEqual(card.to_dict()["url"], "https://x")


class RemoteDiscoveryTests(unittest.TestCase):
    """Coverage for remote agent discovery."""

    def test_discover_no_urls_returns_registry(self) -> None:
        with patch.object(a2a_protocol, "_remote_registry", {}), patch.object(
            a2a_protocol, "a2a_remote_agent_urls", return_value=[]
        ):
            result = discover_remote_agents()
        self.assertEqual(result, {})

    def test_discover_success_populates_registry(self) -> None:
        payload = {
            "name": "Remote",
            "url": "https://remote.example.com",
            "skills": [{"id": "supervisor", "name": "Supervisor"}, {"id": "", "name": "ignored"}],
        }
        with patch.object(a2a_protocol, "_remote_registry", {}), patch(
            "urllib.request.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value.read.return_value = json.dumps(payload).encode("utf-8")
            mock_urlopen.return_value = mock_resp
            result = discover_remote_agents(["https://remote.example.com"])

        self.assertIn("https://remote.example.com", result)
        self.assertIn("supervisor", result["https://remote.example.com"].skills)

    def test_discover_failure_is_swallowed(self) -> None:
        with patch.object(a2a_protocol, "_remote_registry", {}), patch(
            "urllib.request.urlopen", side_effect=OSError("unreachable")
        ):
            result = discover_remote_agents(["https://remote.example.com"])
        self.assertEqual(result, {})

    def test_get_remote_registry_returns_copy(self) -> None:
        with patch.object(a2a_protocol, "_remote_registry", {"u": RemoteAgent(url="u", card={}, skills={})}):
            snapshot = get_remote_registry()
        self.assertIn("u", snapshot)

    def test_discover_uses_well_known_url(self) -> None:
        base = "https://remote.example.com"
        with patch.object(a2a_protocol, "_remote_registry", {}), patch(
            "urllib.request.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value.read.return_value = b'{"name":"R","skills":[]}'
            mock_urlopen.return_value = mock_resp
            discover_remote_agents([base])
        request = mock_urlopen.call_args.args[0]
        self.assertIn(".well-known/agent.json", request.full_url)


class A2ARouterTests(unittest.TestCase):
    """Coverage for the A2ARouter facade."""

    class _FakeClient:
        def __init__(self, outcome) -> None:
            self.outcome = outcome

        def submit(self, *args, **kwargs):
            if isinstance(self.outcome, Exception):
                raise self.outcome
            return self.outcome

    def _remote(self) -> RemoteAgent:
        return RemoteAgent(url="https://r.example.com", card={}, skills={"supervisor": {}})

    def test_a2a_available_reflects_sdk_and_setting(self) -> None:
        with patch.object(a2a_protocol, "a2a_enabled", return_value=True), patch.object(
            a2a_protocol, "_sdk_available", return_value=True
        ):
            self.assertTrue(a2a_available())
        with patch.object(a2a_protocol, "a2a_enabled", return_value=False), patch.object(
            a2a_protocol, "_sdk_available", return_value=True
        ):
            self.assertFalse(a2a_available())

    def test_route_unavailable_when_transport_disabled(self) -> None:
        with patch.object(a2a_protocol, "a2a_available", return_value=False):
            result = A2ARouter().route("supervisor", "hi")
        self.assertEqual(result["status"], "unavailable")

    def test_route_unavailable_when_no_remote(self) -> None:
        with patch.object(a2a_protocol, "a2a_available", return_value=True), patch.object(
            a2a_protocol, "_remote_registry", {}
        ):
            result = A2ARouter().route("supervisor", "hi")
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("No trusted remote agent", result["error"])

    def test_route_success(self) -> None:
        client = self._FakeClient({"routing": "remote_selected", "task": {"id": "t1"}})
        registry = {"https://r.example.com": self._remote()}
        with patch.object(a2a_protocol, "a2a_available", return_value=True), patch.object(
            a2a_protocol, "_remote_registry", registry
        ):
            result = A2ARouter(client=client).route("supervisor", "hi", {"k": "v"})
        self.assertEqual(result["source"], "remote")
        self.assertEqual(result["task"]["id"], "t1")

    def test_route_rejected_dispatch_becomes_failure(self) -> None:
        client = self._FakeClient({"routing": "rejected_policy", "error": "not allowed"})
        registry = {"https://r.example.com": self._remote()}
        with patch.object(a2a_protocol, "a2a_available", return_value=True), patch.object(
            a2a_protocol, "_remote_registry", registry
        ):
            result = A2ARouter(client=client).route("supervisor", "hi")
        self.assertEqual(result["status"], "failed")
        self.assertIn("not allowed", result["error"])

    def test_route_transport_exception_becomes_failure(self) -> None:
        client = self._FakeClient(RuntimeError("socket closed"))
        registry = {"https://r.example.com": self._remote()}
        with patch.object(a2a_protocol, "a2a_available", return_value=True), patch.object(
            a2a_protocol, "_remote_registry", registry
        ):
            result = A2ARouter(client=client).route("supervisor", "hi")
        self.assertEqual(result["status"], "failed")
        self.assertIn("socket closed", result["error"])

    def test_find_remote_none_when_skill_absent(self) -> None:
        registry = {"https://r.example.com": self._remote()}
        with patch.object(a2a_protocol, "_remote_registry", registry):
            self.assertIsNone(A2ARouter().find_remote("nonexistent"))


if __name__ == "__main__":
    unittest.main()
