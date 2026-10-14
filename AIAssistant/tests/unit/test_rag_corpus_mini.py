"""P0 #1 — RAG must really run end-to-end on a mini corpus (no mocks of the index).

The previous version of this test asserted ``len(loaded) >= 0`` (always true),
called ``scan_documents(paths, None)`` (registry ``None``) against the *parent*
of a temp directory, and never checked that anything was retrievable — the
"fake green" the roadmap called out.

Now: scan → build → persist → reload → retrieve, with a deterministic encoder
(``HashingEmbeddingEncoder``) so the whole path runs offline in milliseconds
and a real failure (empty index, empty context, dead pipeline) fails loudly.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_assistant.config.paths import AppPaths
from ai_assistant.config.rag_config import RAGSettings
from ai_assistant.rag.embeddings import HashingEmbeddingEncoder
from ai_assistant.rag.index import is_cache_valid, load_cache, load_or_build_index, scan_documents
from ai_assistant.rag.loaders import DocumentLoaderRegistry
from ai_assistant.rag.retrieval import RetrievalPipeline
from ai_assistant.rag.retrievers import BM25Retriever, FaissVectorRetriever, build_retrievers

CORPUS = """# Vai tro du an
DevManager la ky su chu chot cua du an 3D-Reconstruction.
Teamlead quan ly nhom phat trien va phan cong cong viec.
"""


def _mini_paths(root: Path) -> AppPaths:
    """Build AppPaths entirely inside ``root`` (no walking of parent dirs)."""
    docs = root / "Docs"
    docs.mkdir(parents=True, exist_ok=True)
    return AppPaths(
        project_root=root, base_dir=root, data_dir=root,
        models_dir=root / "Models", cache_dir=root / "Cache",
        logs_dir=root / "logs", docs_dirs=(docs,),
    )


class RagEndToEndTests(unittest.TestCase):
    """Corpus → chunks → FAISS/BM25 → context, then reload from cache."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        docs = self.root / "Docs"
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "roles.txt").write_text(CORPUS, encoding="utf-8")
        self.paths = _mini_paths(self.root)
        self.settings = RAGSettings.from_env_and_dict({})
        self.encoder = HashingEmbeddingEncoder(self.settings.embedding_dimension)
        self.registry = DocumentLoaderRegistry.create_default(
            self.settings.chunk_chars, self.settings.overlap_chars)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build(self):
        chunks = scan_documents(self.paths, self.registry)
        self.assertGreater(len(chunks), 0, "the corpus must produce chunks")
        return load_or_build_index(
            chunks, self.encoder, self.paths, self.settings, force_rebuild=True)

    def test_scan_build_and_retrieve_returns_context(self) -> None:
        index, chunks, bm25 = self._build()
        self.assertIsNotNone(index, "FAISS index must be built, not silently None")
        self.assertIsNotNone(bm25, "BM25 index must be built")
        self.assertEqual(len(chunks), len(scan_documents(self.paths, self.registry)))

        vector_retriever, bm25_retriever = build_retrievers(
            index, bm25, chunks, self.encoder, self.settings)
        self.assertIsInstance(vector_retriever, FaissVectorRetriever)
        self.assertIsInstance(bm25_retriever, BM25Retriever)

        # Both channels must actually hit the corpus document.
        dense_hits = vector_retriever.retrieve("DevManager vai tro du an")
        self.assertTrue(dense_hits, "dense retrieval must return a hit for its own corpus")
        self.assertEqual(dense_hits[0].node.metadata["chunk_index"], 0)
        sparse_hits = bm25_retriever.retrieve("DevManager vai tro du an")
        self.assertTrue(sparse_hits, "BM25 must rank the document for its own terms")
        self.assertEqual(sparse_hits[0].node.metadata["chunk_index"], 0)
        # Vocabulary miss is an explicit empty result, not a random top-k.
        self.assertEqual(bm25_retriever.retrieve("qqqzz unrelated"), [])

        pipeline = RetrievalPipeline(
            vector_retriever, bm25_retriever, reranker=None,
            knowledge_chunks=chunks,
            max_context_chars=self.settings.max_context_chars,
            similarity_threshold=self.settings.similarity_threshold,
            use_reranker=False,
        )
        doc_ctx, code_ctx, images = pipeline.get_context("DevManager vai tro du an")
        self.assertTrue(doc_ctx or code_ctx, "retrieval must return a non-empty context")
        self.assertIn("DevManager", doc_ctx + code_ctx)
        self.assertEqual(images, [])

        # Empty query is a hard no-op, not a "return everything" leak.
        self.assertEqual(pipeline.get_context("   "), ("", "", []))

    def test_cache_roundtrip_serves_without_rebuilding(self) -> None:
        index, chunks, _bm25 = self._build()
        self.assertTrue(is_cache_valid(self.paths, self.settings), "cache must be persisted")
        loaded_index, loaded_chunks, loaded_bm25 = load_cache(self.paths, self.settings)
        self.assertIsNotNone(loaded_index)
        self.assertEqual(len(loaded_chunks), len(chunks))
        self.assertIsNotNone(loaded_bm25)
        self.assertEqual(loaded_index.ntotal, index.ntotal)

        # Second build attempt must hit the cache (no rebuild of the corpus).
        cached_index, cached_chunks, cached_bm25 = load_or_build_index(
            [], None, self.paths, self.settings)
        self.assertIsNotNone(cached_index)
        self.assertEqual(len(cached_chunks), len(chunks))
        self.assertIsNotNone(cached_bm25)

    def test_build_retrievers_without_index_is_explicitly_none(self) -> None:
        """An absent index degrades to ``None`` retrievers, never to a fake hit."""
        vector_retriever, bm25_retriever = build_retrievers(
            None, None, [], self.encoder, self.settings)
        self.assertIsNone(vector_retriever)
        self.assertIsNone(bm25_retriever)
        pipeline = RetrievalPipeline(None, None, None, [])
        self.assertEqual(pipeline.get_context("DevManager"), ("", "", []))


class RetrieverGuardTests(unittest.TestCase):
    """Every ``return []`` in the two retrievers is a policy, not an accident.

    An empty index, a blank query, an unknown vocabulary and a failing
    encoder/BM25 table all have to fail closed (no context, ``rag_chunks`` on
    ``/health`` keeps telling the truth) instead of handing back a random top-k.
    """

    def _encoder(self, raise_error: bool = False):
        """Deterministic offline encoder double (or one that always fails)."""
        import numpy as np

        def encode(text, normalize_embeddings=True):
            if raise_error:
                msg = "encoder offline"
                raise RuntimeError(msg)
            return np.zeros(4, dtype="float32")

        return SimpleNamespace(encode=encode)

    def test_dense_retriever_is_empty_without_an_index_or_a_query(self) -> None:
        encoder = self._encoder()
        self.assertEqual(FaissVectorRetriever(None, encoder).retrieve("anything"), [])
        index = SimpleNamespace(ntotal=1)
        vector = FaissVectorRetriever(index, encoder)
        self.assertEqual(vector.retrieve("   "), [])
        self.assertEqual(vector.retrieve(""), [])

    def test_dense_retriever_skips_a_zero_length_index(self) -> None:
        vector = FaissVectorRetriever(SimpleNamespace(ntotal=0), self._encoder())
        self.assertEqual(vector.retrieve("query"), [])

    def test_dense_retriever_filters_padding_identifiers(self) -> None:
        """``search`` pads with ``-1``; those slots must not become chunks."""
        import numpy as np

        class _Index:
            ntotal = 2

            def search(self, vector, k):
                # 1-D query vector forces the reshape branch before the search.
                return np.asarray([[1.0, -1.0]], dtype="float32"), np.asarray([[0, -1]])

        hits = FaissVectorRetriever(_Index(), self._encoder()).retrieve("query")
        self.assertEqual([hit.node.metadata["chunk_index"] for hit in hits], [0])

    def test_dense_retriever_accepts_an_already_batched_encoding(self) -> None:
        """A batched (2-D) encoding skips the reshape branch entirely."""
        import numpy as np

        class _Encoder:
            def encode(self, text, normalize_embeddings=True):
                return np.zeros((1, 4), dtype="float32")

        seen: dict = {}

        class _Index:
            ntotal = 1

            def search(self, vector, k):
                seen["ndim"] = vector.ndim
                return np.asarray([[1.0]]), np.asarray([[0]])

        hits = FaissVectorRetriever(_Index(), _Encoder()).retrieve("query")
        self.assertEqual([hit.node.metadata["chunk_index"] for hit in hits], [0])
        self.assertEqual(seen["ndim"], 2, "a batched encoding must not be reshaped")

    def test_dense_retriever_swallows_an_encoder_failure(self) -> None:
        vector = FaissVectorRetriever(SimpleNamespace(ntotal=2), self._encoder(raise_error=True))
        self.assertEqual(vector.retrieve("query"), [])

    def test_bm25_retriever_is_empty_without_a_table_or_a_query(self) -> None:
        self.assertEqual(BM25Retriever(None).retrieve("anything"), [])
        table = SimpleNamespace(get_scores=lambda tokens: [1.0])
        retriever = BM25Retriever(table)
        self.assertEqual(retriever.retrieve("   "), [])

    def test_bm25_retriever_is_empty_when_the_query_has_no_vocabulary(self) -> None:
        table = SimpleNamespace(get_scores=lambda tokens: [1.0])
        self.assertEqual(BM25Retriever(table, tokenize=lambda text: []).retrieve("hi"), [])

    def test_bm25_retriever_swallows_a_table_failure(self) -> None:
        def boom(tokens):
            msg = "index corrupt"
            raise RuntimeError(msg)

        self.assertEqual(BM25Retriever(SimpleNamespace(get_scores=boom)).retrieve("hi"), [])

    def test_bm25_retriever_drops_an_all_zero_ranking(self) -> None:
        table = SimpleNamespace(get_scores=lambda tokens: [0.0, 0.0])
        self.assertEqual(BM25Retriever(table).retrieve("hi"), [])

    def test_bm25_retriever_drops_zero_scores_but_keeps_negative_ones(self) -> None:
        """A tiny corpus can rank with a negative idf; sign is not the rank."""
        table = SimpleNamespace(get_scores=lambda tokens: [0.0, 3.0, -2.0])
        hits = BM25Retriever(table).retrieve("hi")
        self.assertEqual(
            [hit.node.metadata["chunk_index"] for hit in hits],
            [1, 2],
        )

    def test_build_retrievers_refuses_to_invent_an_index(self) -> None:
        settings = RAGSettings.from_env_and_dict({})
        self.assertEqual(
            build_retrievers(None, None, [], self._encoder(), settings),
            (None, None),
        )


if __name__ == "__main__":
    unittest.main()
