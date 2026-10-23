"""Targeted coverage for remaining bootstrap/server.py and config/logging.py gaps."""
from __future__ import annotations

import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class BootstrapServerCoverageGaps(unittest.TestCase):
    """Pin defensive branches in bootstrap/server.py and bootstrap/modules.py."""

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
        from ai_assistant.bootstrap.modules import LlmModule

        module = LlmModule(registry=MagicMock(), paths=MagicMock(), model_idx=0,
                           use_vision_llm=False)
        self.assertFalse(module._adopt_backend(None))
        self.assertIsNone(module.llm)
        backend = MagicMock()
        backend.is_vision_supported = False
        backend.model_description = "adopted"
        self.assertTrue(module._adopt_backend(backend))
        self.assertIs(module.llm, backend)
        self.assertEqual(module.active_model_desc, "adopted")

    def test_model_reload_waits_for_an_active_lease(self) -> None:
        import threading

        from ai_assistant.bootstrap.modules import LlmModule

        module = LlmModule(registry=MagicMock(), paths=MagicMock(), model_idx=0,
                           use_vision_llm=False)
        lease_entered = threading.Event()
        release_lease = threading.Event()
        reload_done = threading.Event()

        def hold_lease() -> None:
            with module.model_lease():
                lease_entered.set()
                release_lease.wait(5)

        def swap_model() -> None:
            with module.model_reload():
                reload_done.set()

        holder = threading.Thread(target=hold_lease)
        holder.start()
        self.assertTrue(lease_entered.wait(2))
        swapper = threading.Thread(target=swap_model)
        swapper.start()
        self.assertFalse(reload_done.wait(0.1), "reload started before the lease ended")
        release_lease.set()
        holder.join(2)
        self.assertTrue(reload_done.wait(2))
        swapper.join(2)

    def test_a_new_lease_waits_for_an_active_reload(self) -> None:
        import threading

        from ai_assistant.bootstrap.modules import LlmModule

        module = LlmModule(registry=MagicMock(), paths=MagicMock(), model_idx=0,
                           use_vision_llm=False)
        reload_entered = threading.Event()
        release_reload = threading.Event()
        lease_entered = threading.Event()

        def hold_reload() -> None:
            with module.model_reload():
                reload_entered.set()
                release_reload.wait(5)

        def take_lease() -> None:
            with module.model_lease():
                lease_entered.set()

        swapper = threading.Thread(target=hold_reload)
        swapper.start()
        self.assertTrue(reload_entered.wait(2))
        leaser = threading.Thread(target=take_lease)
        leaser.start()
        self.assertFalse(lease_entered.wait(0.1), "a lease entered during a reload")
        release_reload.set()
        swapper.join(2)
        self.assertTrue(lease_entered.wait(2))
        leaser.join(2)

    def test_llm_load_reload_and_estimate(self) -> None:
        """Cover LlmModule load/reload/estimate paths (single owner, ADR 0041)."""
        from ai_assistant.bootstrap.modules import LlmModule

        module = LlmModule(registry=MagicMock(), paths=MagicMock(),
                           model_idx=0, use_vision_llm=False)
        first_backend = MagicMock()
        first_backend.is_vision_supported = False
        first_backend.model_description = "Test"
        second_backend = MagicMock()
        second_backend.is_vision_supported = True
        second_backend.model_description = "Second"
        with (
            patch("ai_assistant.llm.load_backend", return_value=first_backend) as mock_load,
            patch("ai_assistant.llm.hardware.release_ml_memory") as release,
        ):
            module.load_model()
            mock_load.assert_called_once()
            # Nothing was resident, so there is nothing to release.
            release.assert_not_called()
        self.assertIs(module.llm, first_backend)
        self.assertEqual(module.active_model_desc, "Test")

        with (
            patch("ai_assistant.llm.load_backend", return_value=second_backend),
            patch("ai_assistant.llm.hardware.release_ml_memory") as release,
        ):
            # A second load replaces the resident model in the same owner.
            module.load_model()
            # The previous weights are freed before the new ones are adopted.
            release.assert_called_once()
        self.assertIs(module.llm, second_backend)
        self.assertTrue(module.is_vision_model)
        # estimate defensive
        self.assertEqual(LlmModule.estimate_tokens(""), 1)
        self.assertGreaterEqual(LlmModule.estimate_tokens("hello"), 1)

    def test_release_drops_the_only_backend_reference(self) -> None:
        """``/admin/release-vram`` is only honest if the object can be collected."""
        import gc
        import weakref

        from ai_assistant.bootstrap.modules import LlmModule

        class _Backend:
            is_vision_supported = True
            model_description = "loaded"

        module = LlmModule(registry=MagicMock(), paths=MagicMock(),
                           model_idx=0, use_vision_llm=False)
        backend = _Backend()
        module._adopt_backend(backend)
        reference = weakref.ref(backend)
        del backend

        with patch("ai_assistant.llm.hardware.release_ml_memory") as release:
            self.assertTrue(module.release())
            release.assert_called_once()
        gc.collect()

        self.assertIsNone(reference(), "a second owner kept the backend alive")
        self.assertIsNone(module.llm)
        self.assertEqual(module.active_model_desc, "")
        self.assertFalse(module.is_vision_model)
        # Releasing twice is a no-op, not a second "released" signal.
        self.assertFalse(module.release())

    def test_rag_bridge_methods(self) -> None:
        """Cover RagModule retrieval branches (initialize_rag / get_context)."""
        import os

        from ai_assistant.adapters.persistence.storage_crypto import AesGcmFieldCipher
        from ai_assistant.bootstrap.modules import RagModule

        rag = RagModule(cipher=AesGcmFieldCipher(b"g" * 32))
        self.assertTrue(hasattr(rag.initialize_rag, "__call__"))
        self.assertTrue(hasattr(rag.get_context, "__call__"))
        self.assertTrue(hasattr(rag.release_embedding_for_vision, "__call__"))
        self.assertTrue(hasattr(rag._release_ml_memory, "__call__"))
        self.assertTrue(hasattr(rag._build_retrievers, "__call__"))
        # No pipeline wired yet: get_context is honestly empty, not a stub that
        # pretends it searched.
        result = rag.get_context("test", None, None)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, ("", "", []))
        # Disabled branch (AI_ENABLE_RAG=0): no I/O, deterministic.
        with patch.dict(os.environ, {"AI_ENABLE_RAG": "0"}):
            self.assertEqual(rag.initialize_rag(enable_reranker=False, force_rebuild=False), 0)
        # Cache-miss + empty-corpus branch: scan is stubbed so no full project
        # walk happens, and the module must report the empty state it would
        # expose on /health.
        with patch("ai_assistant.rag.index.is_cache_valid", return_value=False), \
                patch("ai_assistant.rag.index.scan_documents", return_value=[]):
            ret = rag.initialize_rag(enable_reranker=False, force_rebuild=False)
        self.assertEqual(ret, 0)
        self.assertIsNone(rag.knowledge_index)
        self.assertEqual(rag.get_context("test", None, None), ("", "", []))
        # Hard failure (I/O error) is logged with traceback and clears state.
        with patch("ai_assistant.rag.index.is_cache_valid", side_effect=RuntimeError("disk gone")), \
                patch("ai_assistant.bootstrap.modules.logger") as fake_logger:
            self.assertEqual(rag.initialize_rag(), 0)
            self.assertTrue(fake_logger.exception.called)
        self.assertIsNone(rag._pipeline)
        # Retrieval failure path logs and returns the empty envelope, never raises.
        rag._pipeline = object()
        with patch("ai_assistant.bootstrap.modules.logger") as fake_logger:
            self.assertEqual(rag.get_context("test", None, None), ("", "", []))
            self.assertTrue(fake_logger.exception.called)
        rag._pipeline = None
        # Direct release (release_embedding_for_vision → llm.hardware)
        rag.release_embedding_for_vision()

    def test_logging_devnull_branches(self) -> None:
        """Cover sys.stdout/sys.stderr None branches in logging.py (lines 33,35)."""
        import logging

        import ai_assistant.config.logging as log_cfg

        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level

        def _release_new_handlers() -> None:
            """Detach and close only the handlers ``setup_logging`` added.

            A global ``logging.shutdown()`` here closed *every* handler in the
            process — including log files other tests still emit through — and
            left root pointing at a deleted temp directory, which surfaced later
            as ``ValueError: I/O operation on closed file`` and as
            ``FileNotFoundError: …server_*.log`` from unrelated tests.
            """
            for handler in list(root.handlers):
                if handler not in saved_handlers:
                    root.removeHandler(handler)
                    handler.close()

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
                # Both absent streams were given dummies, not left as None …
                self.assertIsNotNone(new_stdout)
                self.assertIsNotNone(new_stderr)
                # Release the temp log file before the directory goes away …
                _release_new_handlers()
                # … and each replacement really accepts output.
                new_stdout.write("stdout survived\n")
                new_stderr.write("stderr survived\n")
        finally:
            _release_new_handlers()
            root.setLevel(saved_level)
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
        from ai_assistant.bootstrap.server import runtime_modules
        runtime_modules.llm.llm = None
        runtime_modules.llm.is_vision_model = False
        runtime_modules.llm.active_model_desc = ""
        runtime_modules.rag.knowledge_index = None
        runtime_modules.rag.knowledge_chunks = []
        runtime_modules.rag.bm25_index = None
