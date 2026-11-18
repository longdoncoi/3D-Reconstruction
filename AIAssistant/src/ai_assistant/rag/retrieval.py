"""RAG Retrieval Pipelines and formatting."""
from __future__ import annotations

import logging
from typing import Any

from ..image_utils import image_to_data_uri
from .domain import ChunkResult, RagHit, RetrievalResult
from .nlp_utils import expand_project_role_query

logger = logging.getLogger("ai_assistant.rag.retrieval")


def rrf_fuse(rankings: list[list[int]], weights: list[float], k: int = 60) -> list[tuple[int, float]]:
    """Fuse ranked lists using Reciprocal Rank Fusion."""
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, chunk_id in enumerate(ranking, 1):
            if chunk_id >= 0:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def hybrid_retrieve(
    query: str,
    query_image_b64: str | None,
    vector_retriever: Any,
    bm25_retriever: Any,
    knowledge_chunks: list[ChunkResult],
    similarity_threshold: float,
    k: int = 30,
    final_k: int = 12
) -> list[ChunkResult]:
    """Hybrid semantic (text/image) + BM25 (text only) retrieval."""
    ranked = hybrid_retrieve_scored(
        query, query_image_b64, vector_retriever, bm25_retriever,
        knowledge_chunks, similarity_threshold, k=k, final_k=final_k,
    )
    return [knowledge_chunks[chunk_id] for chunk_id, _score in ranked]


def hybrid_retrieve_scored(
    query: str,
    query_image_b64: str | None,
    vector_retriever: Any,
    bm25_retriever: Any,
    knowledge_chunks: list[ChunkResult],
    similarity_threshold: float,
    k: int = 30,
    final_k: int = 12,
) -> list[tuple[int, float]]:
    """Return fused chunk identities without discarding their RRF score."""
    if vector_retriever is None or bm25_retriever is None or not knowledge_chunks or (not query.strip() and not query_image_b64):
        return []

    candidate_limit = max(1, min(int(k), 30))
    retrieval_query = expand_project_role_query(query)
    
    dense = vector_retriever.retrieve(retrieval_query)[:candidate_limit]
    sparse = bm25_retriever.retrieve(retrieval_query)[:candidate_limit]
    
    dense_ids = [item.node.metadata["chunk_index"] for item in dense
                 if item.score is None or item.score >= similarity_threshold]
    sparse_ids = [item.node.metadata["chunk_index"] for item in sparse]
    
    combined = rrf_fuse([dense_ids, sparse_ids], [0.55, 0.45])

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:final_k]


def rerank_chunks(query: str, chunks: list[ChunkResult], reranker: Any) -> list[ChunkResult]:
    """Cross-encoder re-ranking for better precision."""
    if reranker is None or not chunks:
        return chunks
    try:
        pairs = [(query, getattr(c, "text", str(c))[:600]) for c in chunks]
        scores = reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, chunks, strict=True), key=lambda x: x[0], reverse=True)
        logger.debug("Rerank scores: %s", [f"{s:.3f}" for s, _ in ranked])
        return [c for _, c in ranked]
    except Exception as error:
        logger.warning("Reranker failed, fallback: %s", error)
        return chunks


def _rerank_scored(
    query: str,
    candidates: list[tuple[int, float]],
    chunks: list[ChunkResult],
    reranker: Any,
) -> list[tuple[int, float, float | None]]:
    """Rerank while retaining chunk identity and both ranking signals."""
    if reranker is None or not candidates:
        return [(chunk_id, fusion_score, None) for chunk_id, fusion_score in candidates]
    try:
        pairs = [(query, chunks[chunk_id].text[:600]) for chunk_id, _ in candidates]
        scores = reranker.predict(pairs, show_progress_bar=False)
        ranked: list[tuple[int, float, float | None]] = sorted(
            (
                (chunk_id, fusion_score, float(rerank_score))
                for rerank_score, (chunk_id, fusion_score) in zip(scores, candidates, strict=True)
            ),
            key=lambda item: item[2] if item[2] is not None else float("-inf"),
            reverse=True,
        )
        logger.debug("Rerank scores: %s", [f"{score:.3f}" for _, _, score in ranked])
    except Exception as error:
        logger.warning("Reranker failed, fallback: %s", error)
        return [(chunk_id, fusion_score, None) for chunk_id, fusion_score in candidates]
    else:
        return ranked


def dedup_by_source(chunks: list[ChunkResult], max_per_source: int = 2) -> list[ChunkResult]:
    """Source deduplication to ensure diverse context."""
    seen: dict[str, int] = {}
    result: list[ChunkResult] = []
    for chunk in chunks:
        src = getattr(chunk, "source_path", str(chunk)[:80])
        if seen.get(src, 0) < max_per_source:
            result.append(chunk)
            seen[src] = seen.get(src, 0) + 1
    logger.debug("Dedup: %d → %d chunks (%d sources)", len(chunks), len(result), len(seen))
    return result


def expand_with_parent(chunks: list[ChunkResult], budget_chars: int) -> list[ChunkResult]:
    """Hierarchical context expansion: append parent_text snippets."""
    seen_texts: set[str] = {getattr(c, "text", "")[:80] for c in chunks}
    expanded: list[ChunkResult] = []
    used_chars = sum(len(getattr(c, "text", "")) for c in chunks)

    for chunk in chunks:
        expanded.append(chunk)
        parent = getattr(chunk, "parent_text", None)
        if not parent:
            continue
        key = parent[:80]
        if key in seen_texts:
            continue
        remaining = budget_chars - used_chars
        if remaining < 200:
            break
            
        parent_snippet = parent[:min(len(parent), remaining, 1200)]
        from copy import copy
        parent_chunk = copy(chunk)
        parent_chunk.text = f"[Parent context]\n{parent_snippet}"
        parent_chunk.parent_text = None
        parent_chunk.hierarchy_level = getattr(chunk, "hierarchy_level", 0) - 1
        expanded.append(parent_chunk)
        seen_texts.add(key)
        used_chars += len(parent_snippet)

    return expanded


def format_context_block(chunks: list[ChunkResult], section_title: str) -> str:
    """Format chunks into a numbered block for the LLM prompt."""
    if not chunks:
        return ""
    lines = [f"=== {section_title} ==="]
    for i, chunk in enumerate(chunks, 1):
        text = getattr(chunk, "text", str(chunk))
        body = text.split("\n", 1)[-1].strip()
        lines.append(f"\n--- Evidence {i} ---\n{body}")
    return "\n".join(lines)


class RetrievalPipeline:
    def __init__(
        self,
        vector_retriever: Any,
        bm25_retriever: Any,
        reranker: Any,
        knowledge_chunks: list[ChunkResult],
        max_context_chars: int = 9000,
        similarity_threshold: float = 0.30,
        use_reranker: bool = True,
        reranker_top_k: int = 8,
        answerability_threshold: float = 0.40,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.knowledge_chunks = knowledge_chunks
        self.max_context_chars = max_context_chars
        self.similarity_threshold = similarity_threshold
        self.use_reranker = use_reranker
        self.reranker_top_k = reranker_top_k
        self.answerability_threshold = max(0.0, min(float(answerability_threshold), 1.0))

    @staticmethod
    def _confidence(fusion_score: float) -> float:
        """Normalise RRF against the best possible rank-one two-channel hit."""
        best_possible = 1.0 / 61.0
        return max(0.0, min(1.0, fusion_score / best_possible))

    def search(
        self,
        query: str,
        query_image_b64: str | None = None,
        result_k: int | None = None,
    ) -> RetrievalResult:
        """Retrieve structured, attributable evidence with a no-answer gate."""
        if not query.strip() and not query_image_b64:
            return RetrievalResult.empty(query, "empty_query")
        if self.vector_retriever is None or self.bm25_retriever is None or not self.knowledge_chunks:
            return RetrievalResult.empty(query, "index_unavailable")
        if result_k is None:
            result_k = self.reranker_top_k

        retrieval_query = expand_project_role_query(query)
        candidate_pool = min(30, max(12, int(result_k) * 3))
        fused = hybrid_retrieve_scored(
            query,
            query_image_b64,
            self.vector_retriever,
            self.bm25_retriever,
            self.knowledge_chunks,
            self.similarity_threshold,
            k=candidate_pool,
            final_k=candidate_pool,
        )
        ranked = _rerank_scored(
            retrieval_query,
            fused,
            self.knowledge_chunks,
            self.reranker if self.use_reranker and query else None,
        )

        selected: list[tuple[int, float, float | None]] = []
        per_source: dict[str, int] = {}
        limit = max(1, min(int(result_k), self.reranker_top_k))
        for chunk_id, fusion_score, rerank_score in ranked:
            chunk = self.knowledge_chunks[chunk_id]
            source = chunk.source_path
            if per_source.get(source, 0) >= 2:
                continue
            per_source[source] = per_source.get(source, 0) + 1
            selected.append((chunk_id, fusion_score, rerank_score))
            if len(selected) >= limit:
                break

        hits: list[RagHit] = []
        for rank, (chunk_id, fusion_score, rerank_score) in enumerate(selected, 1):
            chunk = self.knowledge_chunks[chunk_id]
            metadata = chunk.metadata or {}
            hits.append(RagHit(
                chunk_id=chunk_id,
                text=chunk.text,
                source_path=str(metadata.get("relative_path") or chunk.source_path),
                loader_type=chunk.loader_type,
                rank=rank,
                fusion_score=float(fusion_score),
                confidence=self._confidence(float(fusion_score)),
                rerank_score=rerank_score,
                symbol=str(metadata["symbol"]) if metadata.get("symbol") else None,
                heading=str(metadata["heading"]) if metadata.get("heading") else None,
                line_start=int(metadata["line_start"]) if metadata.get("line_start") is not None else None,
                line_end=int(metadata["line_end"]) if metadata.get("line_end") is not None else None,
                is_image=chunk.is_image,
                parent_text=getattr(chunk, "parent_text", None),
            ))

        confidence = hits[0].confidence if hits else 0.0
        answerable = bool(hits) and confidence >= self.answerability_threshold
        reason = "evidence_found" if answerable else ("low_confidence" if hits else "no_hits")
        return RetrievalResult(
            query=query,
            hits=tuple(hits),
            confidence=confidence,
            answerable=answerable,
            reason=reason,
        )

    def get_context(self, query: str, query_image_b64: str | None = None, result_k: int | None = None) -> tuple[str, str, list[str]]:
        """Compatibility formatter for callers that still consume context blocks.

        New agent/tool callers use :meth:`search` and its answerability gate.
        Keeping this adapter on the legacy retrieval seam also preserves
        injected/third-party retrievers that return chunk objects directly.
        """
        if result_k is None:
            result_k = self.reranker_top_k

        retrieval_query = expand_project_role_query(query)
        candidate_pool = min(30, max(12, int(result_k) * 3))
        candidates = hybrid_retrieve(
            query, query_image_b64,
            self.vector_retriever, self.bm25_retriever,
            self.knowledge_chunks, self.similarity_threshold,
            k=candidate_pool, final_k=candidate_pool,
        )

        if self.use_reranker and query:
            candidates = rerank_chunks(retrieval_query, candidates, self.reranker)
            candidates = candidates[:self.reranker_top_k]
        else:
            candidates = candidates[:12]

        candidates = dedup_by_source(candidates, max_per_source=2)
        candidates = candidates[:max(1, min(result_k, self.reranker_top_k))]

        candidates = expand_with_parent(candidates, budget_chars=self.max_context_chars // 2)

        doc_chunks = []
        code_chunks = []
        image_chunks = []
        total = 0

        for chunk in candidates:
            if getattr(chunk, "is_image", False):
                try:
                    image_chunks.append(chunk.image_b64 or image_to_data_uri(chunk.source_path))
                except Exception as e:
                    logger.warning("Failed to prepare image context %s: %s",
                                   getattr(chunk, "source_path", "?"), e)
                continue

            text_content = getattr(chunk, "text", str(chunk))
            remaining = self.max_context_chars - total
            if remaining < 150:
                break
                
            trimmed_text = text_content[:remaining] if len(text_content) > remaining else text_content
            
            from copy import copy
            trimmed_chunk = copy(chunk) if hasattr(chunk, "text") else chunk
            if hasattr(trimmed_chunk, "text"):
                trimmed_chunk.text = trimmed_text

            if text_content.startswith("[Tai lieu") or text_content.startswith("[Parent context]\n[Tai lieu"):
                doc_chunks.append(trimmed_chunk)
            else:
                code_chunks.append(trimmed_chunk)
            total += len(trimmed_text)

        doc_ctx = format_context_block(doc_chunks, "PROJECT DOCUMENT EVIDENCE")
        code_ctx = format_context_block(code_chunks, "PROJECT CODE EVIDENCE")
        return doc_ctx, code_ctx, image_chunks
