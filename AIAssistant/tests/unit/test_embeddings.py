"""Unit tests for the E5 embedding adapter.

``llama-index`` is not part of the offline CI dependency set and is imported
inside ``E5LlamaEmbedding.__init__``, so the tests inject a fake
``llama_index.core.embeddings.BaseEmbedding`` through ``sys.modules``.
"""
from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest import mock

from ai_assistant.rag.embeddings import E5LlamaEmbedding, EmbeddingEncoder


class _Vector:
    def tolist(self) -> list[float]:
        return [1.0, 2.0]


class _FakeModel:
    def __init__(self) -> None:
        self.last_input = None

    def encode(self, text, normalize_embeddings: bool = False):
        self.last_input = text
        return _Vector()


class _FakeBaseEmbedding:
    def __init__(self, **kwargs) -> None:
        self.model = kwargs["model"]


def _install_fake_llama_index():
    fake_root = types.ModuleType("llama_index")
    fake_core = types.ModuleType("llama_index.core")
    fake_embeddings = types.ModuleType("llama_index.core.embeddings")
    fake_embeddings.BaseEmbedding = _FakeBaseEmbedding
    return mock.patch.dict(
        sys.modules,
        {
            "llama_index": fake_root,
            "llama_index.core": fake_core,
            "llama_index.core.embeddings": fake_embeddings,
        },
    )


class E5LlamaEmbeddingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._patch = _install_fake_llama_index()
        self._patch.start()
        self.model = _FakeModel()
        self.adapter = E5LlamaEmbedding(self.model, query_prefix="q:", passage_prefix="p:")
        self._patch.stop()

    def test_query_embedding_uses_query_prefix(self) -> None:
        result = self.adapter.value._get_query_embedding("hello")
        self.assertEqual(result, [1.0, 2.0])
        self.assertEqual(self.model.last_input, "q:hello")

    def test_text_embedding_uses_passage_prefix(self) -> None:
        result = self.adapter.value._get_text_embedding("doc")
        self.assertEqual(result, [1.0, 2.0])
        self.assertEqual(self.model.last_input, "p:doc")

    def test_text_embeddings_prefix_each_item(self) -> None:
        result = self.adapter.value._get_text_embeddings(["a", "b"])
        self.assertEqual(result, [1.0, 2.0])
        self.assertEqual(self.model.last_input, ["p:a", "p:b"])

    def test_async_query_embedding_matches_sync(self) -> None:
        async def _run():
            return await self.adapter.value._aget_query_embedding("hi")

        self.assertEqual(asyncio.run(_run()), [1.0, 2.0])

    def test_encoder_protocol_is_runtime_checkable(self) -> None:
        """pydantic builds ``isinstance(model, EmbeddingEncoder)``; a plain
        Protocol would raise ``TypeError`` and abort server startup."""
        self.assertTrue(isinstance(self.model, EmbeddingEncoder))
        self.assertFalse(isinstance(object(), EmbeddingEncoder))


if __name__ == "__main__":
    unittest.main()
