"""Unit tests for the RAG retrieval pipeline.

These cover the framework-free fusion/rerank/dedup/expansion logic in
``ai_assistant.rag.retrieval`` with fakes, so they run in the offline CI
environment without faiss, sentence-transformers or llama_index.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass

from ai_assistant.rag.domain import ChunkResult
from ai_assistant.rag.retrieval import (
    RetrievalPipeline,
    dedup_by_source,
    expand_with_parent,
    format_context_block,
    hybrid_retrieve,
    rerank_chunks,
    rrf_fuse,
)


def _chunk(text: str, source: str = "src/a.py", **kwargs) -> ChunkResult:
    return ChunkResult(text=text, source_path=source, loader_type="markdown", **kwargs)


@dataclass
class _Node:
    metadata: dict


@dataclass
class _Item:
    node: _Node
    score: float | None = None


class _FakeRetriever:
    def __init__(self, items: list[_Item]) -> None:
        self._items = items
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[_Item]:
        self.queries.append(query)
        return list(self._items)


class _FakeReranker:
    def __init__(self, scores: list[float] | None = None, error: bool = False) -> None:
        self._scores = scores or []
        self._error = error
        self.pairs: list = []

    def predict(self, pairs, show_progress_bar=False):  # noqa: ARG002
        self.pairs = pairs
        if self._error:
            raise RuntimeError("reranker boom")
        return self._scores


class RrfFuseTests(unittest.TestCase):
    def test_fuses_and_orders_by_combined_score(self):
        result = rrf_fuse([[0, 1], [1, 2]], [1.0, 1.0], k=0)
        self.assertEqual(result[0][0], 1)  # appears in both lists
        self.assertEqual({cid for cid, _ in result}, {0, 1, 2})

    def test_ignores_negative_ids(self):
        result = rrf_fuse([[-1, 0]], [1.0])
        self.assertEqual([cid for cid, _ in result], [0])

    def test_weights_affect_ranking(self):
        result = rrf_fuse([[0], [1]], [10.0, 0.1])
        self.assertEqual(result[0][0], 0)


class HybridRetrieveTests(unittest.TestCase):
    def test_returns_chunks_above_threshold(self):
        chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
        dense = [_Item(_Node({"chunk_index": 0}), 0.9), _Item(_Node({"chunk_index": 1}), 0.2)]
        sparse = [_Item(_Node({"chunk_index": 2}), None)]
        result = hybrid_retrieve(
            "query", None,
            _FakeRetriever(dense), _FakeRetriever(sparse),
            chunks, 0.3, k=10, final_k=5,
        )
        returned = {id(chunk) for chunk in result}
        self.assertIn(id(chunks[0]), returned)
        self.assertIn(id(chunks[2]), returned)
        self.assertNotIn(id(chunks[1]), returned)

    def test_keeps_dense_items_with_none_score(self):
        chunks = [_chunk("a")]
        dense = [_Item(_Node({"chunk_index": 0}), None)]
        result = hybrid_retrieve(
            "q", "data:image/png;base64,AAA",
            _FakeRetriever(dense), _FakeRetriever([]), chunks, 0.9,
        )
        self.assertEqual(result, chunks)

    def test_empty_when_missing_retrievers_or_chunks(self):
        self.assertEqual(hybrid_retrieve("q", None, None, None, [], 0.3), [])

    def test_empty_for_blank_query_without_image(self):
        chunks = [_chunk("a")]
        self.assertEqual(
            hybrid_retrieve("   ", None, _FakeRetriever([]), _FakeRetriever([]), chunks, 0.3),
            [],
        )


class RerankChunksTests(unittest.TestCase):
    def test_orders_by_score(self):
        chunks = [_chunk("a"), _chunk("b")]
        out = rerank_chunks("q", chunks, _FakeReranker([0.1, 0.9]))
        self.assertEqual(out, [chunks[1], chunks[0]])

    def test_without_reranker_returns_input(self):
        chunks = [_chunk("a")]
        self.assertIs(rerank_chunks("q", chunks, None), chunks)

    def test_failure_falls_back_to_input(self):
        chunks = [_chunk("a")]
        self.assertEqual(rerank_chunks("q", chunks, _FakeReranker(error=True)), chunks)


class DedupTests(unittest.TestCase):
    def test_limits_chunks_per_source(self):
        chunks = [
            _chunk("a", "s"), _chunk("b", "s"), _chunk("c", "s"), _chunk("d", "t"),
        ]
        out = dedup_by_source(chunks, max_per_source=2)
        self.assertEqual([c.source_path for c in out], ["s", "s", "t"])


class ExpandWithParentTests(unittest.TestCase):
    def test_appends_parent_snippet(self):
        chunks = [_chunk("child", parent_text="P" * 500, hierarchy_level=2)]
        out = expand_with_parent(chunks, budget_chars=1000)
        self.assertEqual(len(out), 2)
        self.assertIn("[Parent context]", out[1].text)
        self.assertEqual(out[1].hierarchy_level, 1)
        self.assertIsNone(out[1].parent_text)

    def test_skips_duplicate_parent(self):
        parent = "P" * 300
        chunks = [_chunk("a", parent_text=parent), _chunk("b", parent_text=parent)]
        out = expand_with_parent(chunks, budget_chars=5000)
        self.assertEqual(len(out), 3)

    def test_stops_when_budget_exhausted(self):
        chunks = [_chunk("x" * 1000, parent_text="P" * 500)]
        out = expand_with_parent(chunks, budget_chars=1100)
        self.assertEqual(len(out), 1)


class FormatContextBlockTests(unittest.TestCase):
    def test_formats_numbered_evidence(self):
        out = format_context_block([_chunk("title\nbody line")], "TITLE")
        self.assertIn("=== TITLE ===", out)
        self.assertIn("--- Evidence 1 ---", out)
        self.assertIn("body line", out)

    def test_empty_returns_blank(self):
        self.assertEqual(format_context_block([], "X"), "")


class RetrievalPipelineTests(unittest.TestCase):
    def test_missing_retrievers_returns_empty(self):
        pipeline = RetrievalPipeline(None, None, None, [])
        self.assertEqual(pipeline.get_context("q"), ("", "", []))

    def test_splits_doc_and_code_evidence(self):
        doc = _chunk("[Tai lieu A]\nnoi dung tai lieu", "docs/a.md")
        code = _chunk("def foo():\n    return 1", "src/a.py")
        dense = [
            _Item(_Node({"chunk_index": 0}), 0.9),
            _Item(_Node({"chunk_index": 1}), 0.8),
        ]
        pipeline = RetrievalPipeline(
            _FakeRetriever(dense), _FakeRetriever([]), None, [doc, code],
            use_reranker=False, reranker_top_k=8,
        )
        doc_ctx, code_ctx, images = pipeline.get_context("query")
        self.assertIn("PROJECT DOCUMENT EVIDENCE", doc_ctx)
        self.assertIn("PROJECT CODE EVIDENCE", code_ctx)
        self.assertEqual(images, [])

    def test_returns_image_data_uri(self):
        image = ChunkResult(
            text="", source_path="img.png", loader_type="image",
            is_image=True, image_b64="data:image/png;base64,AAA",
        )
        dense = [_Item(_Node({"chunk_index": 0}), 0.9)]
        pipeline = RetrievalPipeline(
            _FakeRetriever(dense), _FakeRetriever([]), None, [image],
            use_reranker=False,
        )
        doc_ctx, code_ctx, images = pipeline.get_context("query")
        self.assertEqual(images, ["data:image/png;base64,AAA"])
        self.assertEqual((doc_ctx, code_ctx), ("", ""))

    def test_uses_reranker_when_enabled(self):
        chunks = [_chunk("a"), _chunk("b")]
        dense = [
            _Item(_Node({"chunk_index": 0}), 0.9),
            _Item(_Node({"chunk_index": 1}), 0.9),
        ]
        pipeline = RetrievalPipeline(
            _FakeRetriever(dense), _FakeRetriever([]), _FakeReranker([0.1, 0.9]), chunks,
            use_reranker=True, reranker_top_k=8,
        )
        _doc_ctx, code_ctx, _images = pipeline.get_context("query")
        self.assertIn("b", code_ctx)

    def test_respects_small_char_budget(self):
        chunks = [_chunk("z" * 500, "src/big.py")]
        dense = [_Item(_Node({"chunk_index": 0}), 0.9)]
        pipeline = RetrievalPipeline(
            _FakeRetriever(dense), _FakeRetriever([]), None, chunks,
            max_context_chars=100, use_reranker=False,
        )
        doc_ctx, code_ctx, _images = pipeline.get_context("q")
        self.assertEqual((doc_ctx, code_ctx), ("", ""))


if __name__ == "__main__":
    unittest.main()
