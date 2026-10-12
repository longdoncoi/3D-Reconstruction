"""Final-edge coverage, part 1: storage, retrieval, registry and transport glue.

Each test below exists for exactly one uncovered line or arc: a task-store
miss, a context-filtered listing, an explicit ``result_k``, an undecodable
image chunk, a schemaless chunk, a TOML registry without a fallback, an
orchestrator-less run, a well-formed agent card, a live metrics registry, a
terminal record that refuses to run again, a non-mapping payload on load and
a reset with nothing on disk.
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai_assistant.adapters.persistence.sqlite_store import SqliteTaskStore
from ai_assistant.agents.service import AgentService
from ai_assistant.application.agent_runs import AgentRunService
from ai_assistant.application.coordination import TaskCoordinator
from ai_assistant.config.model_registry import ModelDefinition, ModelRegistry, load_model_registry
from ai_assistant.domain.tasks import AgentTask


def _task(**overrides) -> AgentTask:
    fields = {"capability": "chat", "message": "hi"}
    fields.update(overrides)
    return AgentTask.new(**fields)


class SqliteTaskStoreTests(unittest.TestCase):
    def _store(self, directory: str) -> SqliteTaskStore:
        return SqliteTaskStore(Path(directory) / "tasks.sqlite")

    def test_get_returns_none_for_an_unknown_task(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            try:
                self.assertIsNone(store.get("no-such-task"))
            finally:
                store.close()

    def test_list_filters_by_context_id(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            try:
                store.create(_task(context_id="c1"))
                store.create(_task(context_id="c2"))
                listed = list(store.list(context_id="c1"))
                self.assertEqual(len(listed), 1)
                self.assertEqual(listed[0].context_id, "c1")
                self.assertEqual(len(list(store.list())), 2)
            finally:
                store.close()


class RetrievalPipelineTests(unittest.TestCase):
    def _pipeline(self, **overrides):
        from ai_assistant.rag.retrieval import RetrievalPipeline

        kwargs = {
            "vector_retriever": None,
            "bm25_retriever": None,
            "reranker": None,
            "knowledge_chunks": [],
            "use_reranker": False,
        }
        kwargs.update(overrides)
        return RetrievalPipeline(**kwargs)

    def test_an_explicit_result_k_skips_the_default(self):
        with patch(
            "ai_assistant.rag.retrieval.hybrid_retrieve", return_value=[],
        ) as hybrid:
            doc, code, images = self._pipeline().get_context("query", result_k=5)
        hybrid.assert_called_once()
        self.assertEqual((doc, code, images), ("", "", []))

    def test_an_undecodable_image_chunk_is_skipped_not_crashed_on(self):
        chunk = SimpleNamespace(is_image=True, image_b64=None,
                                source_path="missing.png", text="")
        with patch(
            "ai_assistant.rag.retrieval.hybrid_retrieve", return_value=[chunk],
        ):
            with self.assertLogs("ai_assistant.rag.retrieval", level="WARNING") as logged:
                _, _, images = self._pipeline().get_context("query", result_k=5)
        self.assertEqual(images, [])
        self.assertTrue(any("missing.png" in line for line in logged.output))

    def test_a_chunk_without_text_falls_back_to_its_string_form(self):
        chunk = SimpleNamespace(is_image=False, source_path="s.py")
        with patch(
            "ai_assistant.rag.retrieval.hybrid_retrieve", return_value=[chunk],
        ):
            _, code_ctx, _ = self._pipeline().get_context("query", result_k=5)
        self.assertIn("PROJECT CODE EVIDENCE", code_ctx)


class ModelRegistryTests(unittest.TestCase):
    def test_a_registry_without_a_fallback_uses_the_builtin_default(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "models.toml"
            config.write_text(
                '[[models]]\nrepo_id = "r"\nfilename = "m.gguf"\ndesc = "M"\n',
                encoding="utf-8",
            )
            registry = load_model_registry(directory)
            self.assertEqual(len(registry.models), 1)
            self.assertIn("Fallback", registry.fallback.desc)


class AgentRunServiceTests(unittest.TestCase):
    def test_a_run_without_an_orchestrator_fails_closed(self):
        service = AgentRunService(orchestrator=None)
        result = service.run(_task())
        self.assertEqual(result["status"], "failed")
        self.assertIn("orchestrator", result["content"])


class AgentCardReadTests(unittest.TestCase):
    def test_a_well_formed_card_parses_its_capabilities(self):
        from ai_assistant.adapters.a2a_client import TrustedA2AClient

        payload = {
            "name": "Remote", "version": "2.0",
            "skills": [{"id": "code"}, {"id": 7}],
            "capabilities": {"streaming": True},
        }
        response = MagicMock()
        response.read.return_value = json.dumps(payload).encode("utf-8")
        response.__enter__.return_value = response
        client = TrustedA2AClient(
            trusted_endpoints=frozenset({"http://remote.test"}), timeout_seconds=1,
        )
        with patch("ai_assistant.adapters.a2a_client.urlopen", return_value=response):
            card = client._read_card("http://remote.test")
        self.assertEqual(card.skills, frozenset({"code"}))
        self.assertTrue(card.streaming)
        self.assertEqual(card.version, "2.0")


class MetricsTests(unittest.TestCase):
    def test_record_tool_with_a_live_registry_observes_latency(self):
        tool = MagicMock()
        latency = MagicMock()
        registry = {"tool": tool, "tool_latency": latency}
        with patch(
            "ai_assistant.observability.metrics.get_metrics", return_value=registry,
        ):
            from ai_assistant.observability.metrics import record_tool
            record_tool("read_file", True, 1.5)
        tool.labels.assert_called_once_with("read_file", "success")
        latency.labels.assert_called_once_with("read_file")

    def test_record_tool_without_a_registry_is_silent(self):
        with patch(
            "ai_assistant.observability.metrics.get_metrics", return_value={},
        ):
            from ai_assistant.observability.metrics import record_tool
            record_tool("read_file", False, None)

    def test_record_tool_without_a_duration_skips_the_histogram(self):
        tool = MagicMock()
        registry = {"tool": tool, "tool_latency": MagicMock()}
        with patch(
            "ai_assistant.observability.metrics.get_metrics", return_value=registry,
        ):
            from ai_assistant.observability.metrics import record_tool
            record_tool("read_file", False)
        tool.labels.assert_called_once_with("read_file", "error")


class CoordinationTests(unittest.TestCase):
    def test_a_terminal_record_refuses_to_run_again(self):
        coordinator = TaskCoordinator()
        coordinator.start("s1")
        coordinator.update("s1", status="succeeded")
        snapshot = coordinator.update("s1", status="running")
        self.assertEqual(snapshot["status"], "succeeded")


class PendingLoadTests(unittest.TestCase):
    def test_a_non_mapping_payload_loads_as_empty(self):
        from ai_assistant.agents.pending_store import PendingActionStore

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.json"
            path.write_text("[1, 2, 3]", encoding="utf-8")
            store = PendingActionStore(path)
            store.load()
            self.assertEqual(len(store), 0)


class ResetStateTests(unittest.TestCase):
    def test_reset_with_no_file_on_disk_still_clears_memory(self):
        from ai_assistant.agents.pending_store import PendingActionStore

        with tempfile.TemporaryDirectory() as directory:
            store = PendingActionStore(Path(directory) / "absent.json")
            store["a"] = {"v": 1}
            service = AgentService(
                pending_actions=store, pending_lock=threading.Lock(),
                task_coordinator=MagicMock(),
            )
            service.reset_state()
            self.assertEqual(len(store), 0)



# ── Registry index and TOML loading (moved from test_coverage_boost2) ──

class ModelRegistryFileTests(unittest.TestCase):
    """Coverage for registry index access and TOML loading."""

    def test_get_by_index_edges(self) -> None:
        fallback = ModelDefinition("fallback/repo", "f.gguf", "fallback")
        m0 = ModelDefinition("repo/a", "a.gguf", "a")
        m1 = ModelDefinition("repo/b", "b.gguf", "b")
        registry = ModelRegistry(models=(m0, m1), fallback=fallback)
        self.assertIs(registry.get_by_index(0), m0)
        self.assertIs(registry.get_by_index(1), m1)
        self.assertIs(registry.get_by_index(99), m0)   # out of bounds -> first
        self.assertIs(registry.get_by_index(-2), m0)   # negative -> first

    def test_get_by_index_empty_returns_fallback(self) -> None:
        fallback = ModelDefinition("fallback/repo", "f.gguf", "fallback")
        registry = ModelRegistry(models=(), fallback=fallback)
        self.assertIs(registry.get_by_index(0), fallback)

    def test_load_model_registry_missing_config_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = load_model_registry(tmpdir)
        self.assertGreater(len(registry.models), 0)
        self.assertTrue(registry.models[0].is_vision is False or registry.models[0].is_vision is True)

    def test_load_model_registry_from_toml(self) -> None:
        toml_text = (
            "[[models]]\n"
            'repo_id = "test/repo"\n'
            'filename = "m.gguf"\n'
            'desc = "Test"\n'
            "is_vision = true\n"
            "[[models]]\n"
            'repo_id = "test/repo2"\n'
            'filename = "m2.gguf"\n'
            'desc = "Test 2"\n'
            "[fallback]\n"
            'repo_id = "fallback/repo"\n'
            'filename = "f.gguf"\n'
            'desc = "Fallback"\n'
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "models.toml").write_text(toml_text, encoding="utf-8")
            registry = load_model_registry(tmpdir)
        self.assertEqual(len(registry.models), 2)
        self.assertTrue(registry.models[0].is_vision)
        self.assertEqual(registry.fallback.repo_id, "fallback/repo")

if __name__ == "__main__":
    unittest.main()
