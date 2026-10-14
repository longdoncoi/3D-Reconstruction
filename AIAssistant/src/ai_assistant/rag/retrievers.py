"""Production retrievers over the cached FAISS index and BM25 table.

``RetrievalPipeline`` (``rag/retrieval.py``) is transport-agnostic: it consumes
objects that expose ``.node.metadata["chunk_index"]`` and ``.score``. This
module provides the two concrete retrievers the composition root wires in
``bootstrap.server._RAGModule.initialize_rag`` — before this module existed the
pipeline was never constructed, so ``get_context`` always returned an empty
context and ``rag_search`` always reported ``found=False``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from .nlp_utils import tokenize_vn

logger = logging.getLogger("ai_assistant.rag.retrievers")


@dataclass(frozen=True)
class RetrievedNode:
    """Minimal node wrapper carrying the chunk identity used for fusion."""

    metadata: dict[str, Any]


@dataclass(frozen=True)
class ScoredNode:
    """Scored retrieval hit, shaped like the pipeline expects."""

    node: RetrievedNode
    score: float | None


class FaissVectorRetriever:
    """Dense top-k retrieval over the FAISS index with the active encoder."""

    def __init__(self, index: Any, encoder: Any, query_prefix: str = "", k: int = 30) -> None:
        self._index = index
        self._encoder = encoder
        self._query_prefix = query_prefix
        self._k = max(1, int(k))

    def retrieve(self, query: str) -> list[ScoredNode]:
        if self._index is None or not str(query).strip():
            return []
        try:
            import faiss as _faiss
            import numpy as np

            ntotal = int(getattr(self._index, "ntotal", 0))
            if ntotal == 0:
                return []
            encoded = self._encoder.encode(self._query_prefix + query, normalize_embeddings=True)
            vector = np.asarray(encoded, dtype="float32")
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            _faiss.normalize_L2(vector)
            scores, identifiers = self._index.search(vector, min(self._k, ntotal))
        except Exception:
            logger.exception("Dense retrieval failed")
            return []
        results: list[ScoredNode] = []
        for score, chunk_id in zip(scores[0], identifiers[0]):
            if int(chunk_id) < 0:
                continue
            results.append(ScoredNode(RetrievedNode({"chunk_index": int(chunk_id)}), float(score)))
        return results


class BM25Retriever:
    """Sparse top-k retrieval over the cached ``rank_bm25`` table."""

    def __init__(self, bm25: Any, k: int = 30,
                 tokenize: Callable[[str], list[str]] = tokenize_vn) -> None:
        self._bm25 = bm25
        self._k = max(1, int(k))
        self._tokenize = tokenize

    def retrieve(self, query: str) -> list[ScoredNode]:
        if self._bm25 is None or not str(query).strip():
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        try:
            import numpy as np

            scores = np.asarray(self._bm25.get_scores(tokens), dtype="float64")
        except Exception:
            logger.exception("BM25 retrieval failed")
            return []
        if scores.size == 0 or not bool(np.any(scores)):
            # No query term exists in the corpus vocabulary: nothing to rank.
            return []
        # Only zero entries are dropped: with a tiny corpus BM25Okapi can emit
        # negative scores (negative idf), and the rank — not the sign — is what
        # the fusion in ``hybrid_retrieve`` consumes.
        order = np.argsort(scores)[::-1]
        results: list[ScoredNode] = []
        for chunk_id in order[: self._k]:
            score = float(scores[chunk_id])
            if score == 0.0:
                continue
            results.append(ScoredNode(RetrievedNode({"chunk_index": int(chunk_id)}), score))
        return results


def build_retrievers(index: Any, bm25: Any, chunks: list[Any], encoder: Any,
                     settings: Any, k: int = 30) -> tuple[FaissVectorRetriever | None, BM25Retriever | None]:
    """Build the dense/sparse pair, or ``(None, None)`` when nothing is indexed.

    Returning ``None`` explicitly (instead of a retriever over an empty index)
    keeps :func:`ai_assistant.rag.retrieval.hybrid_retrieve` fail-closed: no
    index means no context, and ``/health`` reports ``rag_chunks == 0``.
    """
    if index is None or not chunks:
        logger.warning("RAG retrievers not built: index=%s chunks=%d",
                       "present" if index is not None else "missing", len(chunks))
        return None, None
    query_prefix = str(getattr(settings, "embedding_query_prefix", ""))
    vector_retriever = FaissVectorRetriever(index, encoder, query_prefix=query_prefix, k=k)
    sparse_retriever = BM25Retriever(bm25, k=k)
    return vector_retriever, sparse_retriever


__all__ = [
    "BM25Retriever",
    "FaissVectorRetriever",
    "RetrievedNode",
    "ScoredNode",
    "build_retrievers",
]
