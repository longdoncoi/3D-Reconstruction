"""Cache integration tests for the legacy RAG compatibility shim."""
from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from ai_assistant.legacy import rag_module
from ai_assistant.rag.domain import ChunkResult


class _SentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass


class LegacyRagCacheTests(unittest.TestCase):
    def setUp(self):
        self._state = {
            name: getattr(rag_module, name)
            for name in (
                "knowledge_index",
                "knowledge_chunks",
                "bm25_index",
                "vector_retriever",
                "bm25_retriever",
                "embed_model_ref",
                "_reranker",
                "_pipeline",
            )
        }
        self.addCleanup(self._restore_state)

    def _restore_state(self):
        for name, value in self._state.items():
            setattr(rag_module, name, value)

    def _run_with_fake_sentence_transformers(self, **patches):
        sentence_transformers = SimpleNamespace(
            SentenceTransformer=_SentenceTransformer,
            CrossEncoder=mock.Mock(),
        )
        defaults = {
            "ENABLE_RAG": True,
            "USE_RERANKER": False,
            **patches,
        }
        return mock.patch.multiple(rag_module, **defaults), mock.patch.dict(
            sys.modules, {"sentence_transformers": sentence_transformers}
        )

    def test_valid_cache_skips_scan_and_index_rebuild(self):
        chunks = [ChunkResult(text="cached", source_path="cached.md", loader_type="md")]
        index, bm25 = object(), object()
        dense, sparse = object(), object()
        module_patch, import_patch = self._run_with_fake_sentence_transformers()

        with module_patch, import_patch, \
             mock.patch.object(rag_module, "is_cache_valid", return_value=True), \
             mock.patch.object(rag_module, "load_cache", return_value=(index, chunks, bm25)), \
             mock.patch.object(rag_module, "scan_documents") as scan, \
             mock.patch.object(rag_module, "build_index_from_scratch") as build, \
             mock.patch.object(rag_module, "save_cache") as save, \
             mock.patch.object(rag_module, "_build_retrievers", return_value=(dense, sparse)):
            count = rag_module.initialize_rag()

        self.assertEqual(count, 1)
        scan.assert_not_called()
        build.assert_not_called()
        save.assert_not_called()
        self.assertIs(rag_module.knowledge_index, index)
        self.assertIs(rag_module.vector_retriever, dense)
        self.assertIs(rag_module.bm25_retriever, sparse)

    def test_cache_miss_builds_and_saves_index(self):
        chunks = [ChunkResult(text="fresh", source_path="fresh.md", loader_type="md")]
        index, bm25 = object(), object()
        module_patch, import_patch = self._run_with_fake_sentence_transformers()
        registry = object()

        with module_patch, import_patch, \
             mock.patch.object(rag_module, "is_cache_valid", return_value=False), \
             mock.patch.object(rag_module.DocumentLoaderRegistry, "create_default", return_value=registry), \
             mock.patch.object(rag_module, "scan_documents", return_value=chunks) as scan, \
             mock.patch.object(
                 rag_module, "build_index_from_scratch", return_value=(index, chunks, bm25)
             ) as build, \
             mock.patch.object(rag_module, "save_cache") as save, \
             mock.patch.object(rag_module, "_build_retrievers", return_value=(object(), object())):
            count = rag_module.initialize_rag()

        self.assertEqual(count, 1)
        scan.assert_called_once_with(rag_module._paths, registry)
        build.assert_called_once_with(chunks, mock.ANY, rag_module._rag_settings)
        save.assert_called_once_with(index, chunks, bm25, rag_module._paths, rag_module._rag_settings)

    def test_cached_index_retrievers_preserve_chunk_indices(self):
        encoder = mock.Mock()
        encoder.encode.return_value = [[0.1, 0.2]]
        index = SimpleNamespace(
            ntotal=2,
            search=mock.Mock(return_value=([[0.9, 0.6]], [[1, 0]])),
        )
        faiss = SimpleNamespace(normalize_L2=mock.Mock())
        bm25 = SimpleNamespace(get_scores=mock.Mock(return_value=[0.1, 0.8]))

        with mock.patch.dict(sys.modules, {"faiss": faiss}):
            dense, sparse = rag_module._build_retrievers(index, bm25, encoder)
            dense_items = dense.retrieve("find cached file")
        sparse_items = sparse.retrieve("find cached file")

        encoder.encode.assert_called_once()
        faiss.normalize_L2.assert_called_once()
        self.assertEqual(
            [item.node.metadata["chunk_index"] for item in dense_items], [1, 0]
        )
        self.assertEqual(
            [item.node.metadata["chunk_index"] for item in sparse_items], [1, 0]
        )


if __name__ == "__main__":
    unittest.main()
