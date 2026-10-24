"""Coverage for the RAG wiring the composition root actually performs.

``RagModule._wire_index`` is the one path that goes from an on-disk corpus to
a serving ``RetrievalPipeline``: scan → build → persist → wire, then the
cache-hit variant that only loads. The surrounding tests stub it out (they are
about the *observable* empty state), so this file runs it for real against a
mini corpus with a deterministic encoder.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_assistant.adapters.persistence.storage_crypto import AesGcmFieldCipher
from ai_assistant.bootstrap.modules import RagModule
from ai_assistant.config.paths import AppPaths
from ai_assistant.config.rag_config import RAGSettings
from ai_assistant.rag.embeddings import HashingEmbeddingEncoder

CORPUS = """# Roles
DevManager la ky su chu chot cua du an.
Teamlead quan ly nhom phat trien.
"""


def _mini_paths(root: Path) -> AppPaths:
    docs = root / "Docs"
    docs.mkdir(parents=True, exist_ok=True)
    return AppPaths(
        project_root=root, base_dir=root, data_dir=root,
        models_dir=root / "Models", cache_dir=root / "Cache",
        logs_dir=root / "logs", docs_dirs=(docs,),
    )


class WireIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "Docs").mkdir(parents=True, exist_ok=True)
        (self.root / "Docs" / "roles.txt").write_text(CORPUS, encoding="utf-8")
        self.paths = _mini_paths(self.root)
        self.settings = RAGSettings.from_env_and_dict({})
        encoder = HashingEmbeddingEncoder(self.settings.embedding_dimension)
        self._patches = [
            mock.patch("ai_assistant.config.paths.get_paths", return_value=self.paths),
            # Keep the encoder offline: the factory would otherwise load weights.
            mock.patch("ai_assistant.rag.embeddings.lazy_embedding_encoder",
                       return_value=encoder),
        ]
        for patcher in self._patches:
            patcher.start()
        self.rag = RagModule(cipher=AesGcmFieldCipher(b"w" * 32))
        self.addCleanup(self.rag.reset_index_state)

    def tearDown(self) -> None:
        for patcher in self._patches:
            patcher.stop()
        self._tmp.cleanup()

    def test_building_from_scratch_wires_a_serving_pipeline(self) -> None:
        count = self.rag.initialize_rag(enable_reranker=False, force_rebuild=True)

        self.assertGreater(count, 0)
        self.assertEqual(len(self.rag.knowledge_chunks), count)
        self.assertIsNotNone(self.rag.knowledge_index)
        self.assertIsNotNone(self.rag._pipeline)
        # The retriever pair lives on the pipeline that actually serves
        # queries (the write-only ``RagModule.vector_retriever`` field went
        # with the ADR 0015 dead-surface sweep).
        self.assertIsNotNone(self.rag._pipeline.vector_retriever)
        self.assertIsNotNone(self.rag._pipeline.bm25_retriever)

        doc_ctx, code_ctx, _images = self.rag.get_context("DevManager vai tro du an")
        self.assertTrue(doc_ctx or code_ctx)

    def test_a_second_run_loads_the_cache_instead_of_rebuilding(self) -> None:
        self.rag.initialize_rag(enable_reranker=False, force_rebuild=True)
        loaded = self.rag.initialize_rag(enable_reranker=False, force_rebuild=False)

        self.assertGreater(loaded, 0)
        self.assertIsNotNone(self.rag._pipeline)

    def test_an_unusable_index_is_reported_as_empty_not_as_a_success(self) -> None:
        """ADR 0015: a build that yields nothing degrades, it does not claim."""
        with mock.patch(
            "ai_assistant.rag.index.load_or_build_index",
            return_value=(None, [], None),
        ):
            self.assertEqual(self.rag.initialize_rag(force_rebuild=True), 0)
        self.assertEqual(self.rag.knowledge_chunks, [])
        self.assertIsNone(self.rag._pipeline)

    def test_retrievers_refuse_a_bare_state(self) -> None:
        self.assertEqual(self.rag._build_retrievers(None, None, None, None, None), (None, None))

    def test_releasing_ml_memory_delegates_to_the_hardware_helper(self) -> None:
        with mock.patch("ai_assistant.llm.hardware.release_ml_memory") as release:
            self.rag.release_ml_memory()
        release.assert_called_once_with()

    def test_a_disabled_rag_reports_zero_without_touching_the_corpus(self) -> None:
        with mock.patch.dict("os.environ", {"AI_ENABLE_RAG": "0"}):
            self.assertEqual(self.rag.initialize_rag(force_rebuild=False), 0)
        self.assertEqual(self.rag.knowledge_chunks, [])

    def test_the_reset_drops_every_retrieval_handle_together(self) -> None:
        """ADR 0032: the reset used to exist twice with two different field
        sets — one cleared the pipeline but not the encoder/reranker
        references, the other cleared those but not the pipeline. A reader
        could therefore observe ``/health`` reporting zero chunks while
        retrieval still served the previous corpus out of the stale
        pipeline."""
        self.rag.initialize_rag(enable_reranker=False, force_rebuild=True)
        self.assertIsNotNone(self.rag._pipeline)
        self.assertIsNotNone(self.rag.embed_model_ref)

        self.rag.reset_index_state()

        self.assertIsNone(self.rag.knowledge_index)
        self.assertEqual(self.rag.knowledge_chunks, [])
        self.assertIsNone(self.rag.bm25_index)
        self.assertIsNone(self.rag._pipeline)
        self.assertIsNone(self.rag.embed_model_ref)
        self.assertFalse(self.rag.reranker_active)
        # And nothing is served any more.
        self.assertEqual(self.rag.get_context("DevManager"), ("", "", []))

    def test_a_failed_rebuild_leaves_no_stale_corpus_behind(self) -> None:
        """The failure path resets through the same single owner, so the
        previous corpus cannot survive a failed rebuild."""
        self.rag.initialize_rag(enable_reranker=False, force_rebuild=True)
        with mock.patch(
            "ai_assistant.rag.index.load_or_build_index",
            side_effect=RuntimeError("index boom"),
        ):
            self.assertEqual(self.rag.initialize_rag(force_rebuild=True), 0)
        self.assertIsNone(self.rag._pipeline)
        self.assertEqual(self.rag.knowledge_chunks, [])
        self.assertEqual(self.rag.get_context("DevManager"), ("", "", []))


if __name__ == "__main__":
    unittest.main()
