"""Targeted coverage for remaining bootstrap/server.py and config/logging.py gaps."""
from __future__ import annotations

import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class BootstrapServerCoverageGaps(unittest.TestCase):
    """Pin defensive import-time branches in bootstrap/server.py."""

    def test_model_index_guard_negative_and_overrun(self) -> None:
        """The selector really is clamped by ``_resolve_model_idx``.

        The previous version of this test re-implemented the guard inline, so
        it asserted its own copy of the logic instead of the shipped one.
        """
        from ai_assistant.bootstrap import server as s

        models = s.MODELS
        cases = {
            ("server.py", "-1"): 0,
            ("server.py", "99999"): 0,
            ("server.py", "abc"): 0,
            ("server.py", "2.5"): 0,
        }
        for argv, expected in cases.items():
            with self.subTest(argv=argv):
                self.assertEqual(s._resolve_model_idx(list(argv), models), expected)
        self.assertEqual(s._resolve_model_idx(["server.py"], models), 0)
        self.assertEqual(s._resolve_model_idx(["server.py", "0"], models), 0)
        self.assertIn(s._resolve_model_idx(["server.py", "1"], models), (0, 1))

    def test_adopt_backend_refuses_a_missing_backend(self) -> None:
        """A failed load must be visible, not leave the old model advertised."""
        from ai_assistant.bootstrap.server import _LLMModule

        previous = _LLMModule.llm
        try:
            self.assertFalse(_LLMModule._adopt_backend(None))
            self.assertIs(_LLMModule.llm, previous)
            backend = MagicMock()
            backend.is_vision_supported = False
            backend.model_description = "adopted"
            self.assertTrue(_LLMModule._adopt_backend(backend))
            self.assertIs(_LLMModule.llm, backend)
            self.assertEqual(_LLMModule.active_model_desc, "adopted")
        finally:
            _LLMModule.llm = previous

    def test_llm_load_reload_and_estimate(self) -> None:
        """Cover _LLMModule load/reload/estimate (lines 132-164)."""
        from ai_assistant.bootstrap.server import _LLMModule, llm_module
        # Mock registry and backend so load/reload can execute without weights
        mock_backend = MagicMock()
        mock_backend.is_vision_supported = False
        mock_backend.model_description = "Test"
        with patch("ai_assistant.bootstrap.server.load_model") as mock_load:
            with patch("ai_assistant.bootstrap.server.get_backend", return_value=mock_backend):
                llm_module.load_model()
                mock_load.assert_called()
                # reload path — mock reload_model dependency
                with patch("ai_assistant.bootstrap.server.reload_model"):
                    llm_module.reload_model()
        # estimate defensive
        self.assertEqual(_LLMModule.estimate_tokens(""), 1)
        self.assertGreaterEqual(_LLMModule.estimate_tokens("hello"), 1)

    def test_rag_bridge_methods(self) -> None:
        """Cover _RAGModule legacy bridge branches (initialize_rag / get_context)."""
        import os

        from ai_assistant.bootstrap.server import _RAGModule, rag_module
        self.assertTrue(hasattr(_RAGModule.initialize_rag, "__call__"))
        self.assertTrue(hasattr(_RAGModule.get_context, "__call__"))
        self.assertTrue(hasattr(_RAGModule.release_embedding_for_vision, "__call__"))
        self.assertTrue(hasattr(_RAGModule._release_ml_memory, "__call__"))
        self.assertTrue(hasattr(_RAGModule._build_retrievers, "__call__"))
        # No pipeline wired yet: get_context is honestly empty, not a stub that
        # pretends it searched.
        result = rag_module.get_context("test", None, None)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, ("", "", []))
        # Disabled branch (AI_ENABLE_RAG=0): no I/O, deterministic.
        with patch.dict(os.environ, {"AI_ENABLE_RAG": "0"}):
            self.assertEqual(rag_module.initialize_rag(enable_reranker=False, force_rebuild=False), 0)
        # Cache-miss + empty-corpus branch: scan is stubbed so no full project
        # walk happens, and the module must report the empty state it would
        # expose on /health.
        with patch("ai_assistant.rag.index.is_cache_valid", return_value=False), \
                patch("ai_assistant.rag.index.scan_documents", return_value=[]):
            ret = rag_module.initialize_rag(enable_reranker=False, force_rebuild=False)
        self.assertEqual(ret, 0)
        self.assertIsNone(rag_module.knowledge_index)
        self.assertEqual(rag_module.get_context("test", None, None), ("", "", []))
        # Hard failure (I/O error) is logged with traceback and clears state.
        with patch("ai_assistant.rag.index.is_cache_valid", side_effect=RuntimeError("disk gone")), \
                patch("ai_assistant.bootstrap.server.logger") as fake_logger:
            self.assertEqual(rag_module.initialize_rag(), 0)
            self.assertTrue(fake_logger.exception.called)
        self.assertIsNone(rag_module._pipeline)
        # Retrieval failure path logs and returns the empty envelope, never raises.
        rag_module._pipeline = object()
        with patch("ai_assistant.bootstrap.server.logger") as fake_logger:
            self.assertEqual(rag_module.get_context("test", None, None), ("", "", []))
            self.assertTrue(fake_logger.exception.called)
        rag_module._pipeline = None
        # Direct release (release_embedding_for_vision → llm.hardware)
        rag_module.release_embedding_for_vision()

    def test_logging_devnull_branches(self) -> None:
        """Cover sys.stdout/sys.stderr None branches in logging.py (lines 33,35)."""
        import ai_assistant.config.logging as log_cfg
        # Force branches by temporarily nulling streams
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        new_stdout = None
        new_stderr = None
        try:
            sys.stdout = None  # type: ignore[assignment]
            sys.stderr = None  # type: ignore[assignment]
            with tempfile.TemporaryDirectory() as tmp_logs:
                log_cfg.setup_logging(tmp_logs)
                new_stdout = sys.stdout
                new_stderr = sys.stderr
                import logging
                logging.shutdown()
        finally:
            # Prevent ResourceWarning from devnull handles we opened
            if new_stdout is not None and new_stdout is not old_stdout and hasattr(new_stdout, "close"):
                new_stdout.close()
            if new_stderr is not None and new_stderr is not old_stderr and hasattr(new_stderr, "close"):
                new_stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class ConfigPathsCoverage(unittest.TestCase):
    def test_embed_cache_dir_property(self) -> None:
        import tempfile

        from ai_assistant.config.paths import AppPaths
        with tempfile.TemporaryDirectory() as tmp:
            p = AppPaths(
                project_root=__import__("pathlib").Path(tmp),
                base_dir=__import__("pathlib").Path(tmp),
                data_dir=__import__("pathlib").Path(tmp),
                models_dir=__import__("pathlib").Path(tmp),
                cache_dir=__import__("pathlib").Path(tmp),
                logs_dir=__import__("pathlib").Path(tmp),
                docs_dirs=(),
            )
            self.assertIn("embed_model", str(p.embed_cache_dir))


class BootstrapStabilizer(unittest.TestCase):
    """Stabilize mock-path state leakage across integration tests."""

    def tearDown(self) -> None:
        from ai_assistant.bootstrap.server import llm_module, rag_module
        llm_module.llm = None
        llm_module.is_vision_model = False
        llm_module.active_model_desc = ""
        rag_module.knowledge_index = None
        rag_module.knowledge_chunks = []
        rag_module.bm25_index = None
        rag_module.vector_retriever = None
        rag_module.bm25_retriever = None
