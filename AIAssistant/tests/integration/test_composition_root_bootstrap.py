"""Composition-root bootstrap regression tests (P0).

The production server must reach a usable LLM after ``bootstrap_runtime()``.
A regression during the ``modules/`` removal replaced the functional legacy
bridges with no-op shims, so the entry point started while every agent call
failed with ``ModelNotLoadedError``. These tests pin the load path:

    bootstrap_runtime()
      -> legacy.rag_module.initialize_rag  -> real RAG bridge
      -> legacy.llm_module.load_model      -> ai_assistant.llm.load_model
      -> LlamaCppBackend                   -> llm_module.llm is set

Model weights and the embedding model are never loaded in tests; the backend
and the document scan are replaced with doubles.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from ai_assistant.legacy import llm_module, rag_module


class CompositionRootImportTests(unittest.TestCase):
    """The entry module must import and expose a consistent runtime shape."""

    @classmethod
    def setUpClass(cls) -> None:
        import StartChatbotServer as server_module

        cls.server = server_module

    def test_entry_module_exposes_server_start_time(self) -> None:
        """``StartChatbotServer`` imports ``_SERVER_START_TIME`` from legacy config."""
        self.assertTrue(hasattr(self.server, "_SERVER_START_TIME"))
        self.assertTrue(hasattr(self.server, "bootstrap_runtime"))

    def test_legacy_bridges_are_functional_not_noops(self) -> None:
        """The bridges must delegate to the real loaders, not log-and-return."""
        self.assertIsNotNone(llm_module.load_model)
        # A no-op shim logs "legacy shim" and never delegates; the real bridge
        # calls into ai_assistant.llm, which we can observe by patching below.
        import inspect
        source = inspect.getsource(llm_module.load_model)
        self.assertNotIn("legacy shim", source)
        self.assertTrue(
            any(name in source for name in ("_new_load", "load_model")),
            "load_model must delegate to the real model loader",
        )

    def test_rag_bridge_exposes_pipeline_state(self) -> None:
        """RAG consumers (admin/health/tools) read these attributes."""
        for attribute in (
            "knowledge_index",
            "knowledge_chunks",
            "bm25_index",
            "_reranker",
            "rag_lock",
            "initialize_rag",
            "get_context",
            "release_embedding_for_vision",
            "_release_ml_memory",
        ):
            self.assertTrue(hasattr(rag_module, attribute), f"missing {attribute}")


class BootstrapRuntimeLoadTests(unittest.TestCase):
    """``bootstrap_runtime()`` must publish a usable backend on the LLM runtime."""

    @classmethod
    def setUpClass(cls) -> None:
        import StartChatbotServer as server_module

        cls.server = server_module

    def setUp(self) -> None:
        from ai_assistant.llm import model_loader

        self._saved_backend = model_loader._global_backend
        model_loader._global_backend = None

    def tearDown(self) -> None:
        from ai_assistant.llm import model_loader

        model_loader._global_backend = self._saved_backend

    @staticmethod
    def _patch_backend(backend):
        """Isolate the load path: no weights, no downloads, no VRAM release."""
        return (
            patch(
                "ai_assistant.llm.model_loader.LlamaCppBackend",
                return_value=backend,
            ),
            patch(
                "ai_assistant.llm.model_loader.download_if_missing",
                return_value="Models/test.gguf",
            ),
            patch("ai_assistant.llm.model_loader.release_ml_memory"),
        )

    def test_bootstrap_publishes_llm_backend(self) -> None:
        backend = MagicMock()
        backend.is_vision_supported = False
        backend.model_description = "Test model (Q4_K_M)"

        p1, p2, p3 = self._patch_backend(backend)
        with p1, p2, p3, patch.object(rag_module, "initialize_rag", return_value=0):
            self.server.bootstrap_runtime()

        # The agent facade gates on llm_runtime.llm — this is the P0 signal.
        self.assertIsNotNone(llm_module.llm)
        self.assertFalse(llm_module.is_vision_model)
        self.assertEqual(llm_module.active_model_desc, "Test model (Q4_K_M)")

        # The whole facade must now accept an execution request.
        self.server.agent_service._require_llm()

    def test_agent_service_gated_before_load(self) -> None:
        """Without a backend the facade must raise, proving the gate is real."""
        from ai_assistant.domain.errors import ModelNotLoadedError

        saved = llm_module.llm
        try:
            llm_module.llm = None
            with self.assertRaises(ModelNotLoadedError):
                self.server.agent_service._require_llm()
        finally:
            llm_module.llm = saved

    def test_vision_model_releases_embedding(self) -> None:
        """Vision models must drop the embedding model after load."""
        backend = MagicMock()
        backend.is_vision_supported = True
        backend.model_description = "Vision model"

        p1, p2, p3 = self._patch_backend(backend)
        with p1, p2, p3, patch.object(
            rag_module, "initialize_rag", return_value=0
        ), patch.object(rag_module, "release_embedding_for_vision") as release:
            self.server.bootstrap_runtime()
            release.assert_called_once()

        self.assertTrue(llm_module.is_vision_model)


if __name__ == "__main__":
    unittest.main()
