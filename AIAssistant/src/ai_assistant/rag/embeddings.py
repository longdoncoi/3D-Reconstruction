"""Embedding model adapters."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingEncoder(Protocol):
    """The minimal encoding surface the adapter needs from any encoder model.

    Both the E5 ONNX runtime wrapper and the sentence-transformers models expose
    this shape, so the LlamaIndex adapter depends on an interface instead of a
    concrete class.

    ``@runtime_checkable`` is required because pydantic builds an ``isinstance``
    validator for the ``model`` field of the local ``E5Embedding`` model. A plain
    Protocol cannot be the second argument to ``isinstance``, which made the
    pydantic schema build fail and aborted server startup (RAG index step).
    """

    def encode(
        self, texts: str | list[str], normalize_embeddings: bool = False
    ) -> Any: ...


class E5LlamaEmbedding:
    """Adapter that lets LlamaIndex use the application's E5 encoder/prefixes."""

    def __init__(self, model, query_prefix: str, passage_prefix: str):
        from llama_index.core.embeddings import BaseEmbedding
        
        # Capture prefixes in the closure so we don't depend on global config
        _q_prefix = query_prefix
        _p_prefix = passage_prefix

        class E5Embedding(BaseEmbedding):
            model: EmbeddingEncoder

            def _get_query_embedding(self, query: str):
                return self.model.encode(_q_prefix + query, normalize_embeddings=True).tolist()

            async def _aget_query_embedding(self, query: str):
                return self._get_query_embedding(query)

            def _get_text_embedding(self, text: str):
                return self.model.encode(_p_prefix + text, normalize_embeddings=True).tolist()

            def _get_text_embeddings(self, texts: list[str]):
                return self.model.encode(
                    [_p_prefix + text for text in texts], 
                    normalize_embeddings=True
                ).tolist()

        self.value = E5Embedding(model=model)
