"""Embedding model adapters."""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger("ai_assistant.rag.embeddings")


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

    def __init__(self, model: EmbeddingEncoder, query_prefix: str, passage_prefix: str):
        from llama_index.core.embeddings import BaseEmbedding
        
        # Capture prefixes in the closure so we don't depend on global config
        _q_prefix = query_prefix
        _p_prefix = passage_prefix

        class E5Embedding(BaseEmbedding):
            model: EmbeddingEncoder

            def _get_query_embedding(self, query: str) -> list[float]:
                embedding: list[float] = self.model.encode(
                    _q_prefix + query, normalize_embeddings=True
                ).tolist()
                return embedding

            async def _aget_query_embedding(self, query: str) -> list[float]:
                return self._get_query_embedding(query)

            def _get_text_embedding(self, text: str) -> list[float]:
                embedding: list[float] = self.model.encode(
                    _p_prefix + text, normalize_embeddings=True
                ).tolist()
                return embedding

            def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
                embeddings: list[list[float]] = self.model.encode(
                    [_p_prefix + text for text in texts], 
                    normalize_embeddings=True
                ).tolist()
                return embeddings

        self.value = E5Embedding(model=model)


class HashingEmbeddingEncoder:
    """Deterministic feature-hashing encoder used only as an offline fallback.

    This is **not** a semantic encoder: it exists so the retrieval contract stays
    testable with a fixed vector space and so a machine without
    ``sentence-transformers`` degrades *visibly* (warning log + ``is_fallback``
    flag surfaced by the caller) instead of the whole RAG pipeline silently
    dying. Chunks and queries are hashed into the configured dimension with a
    signed accumulator, then L2-normalized when asked.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = max(8, int(dimension))

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: str | list[str],
        normalize_embeddings: bool = False,
        **_: Any,
    ) -> Any:
        import numpy as np

        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vectors = []
        for text in items:
            vector = np.zeros(self._dimension, dtype="float32")
            for token in re.findall(r"\w+", str(text).casefold(), flags=re.UNICODE):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self._dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[bucket] += sign
            if normalize_embeddings:
                norm = float(np.linalg.norm(vector))
                if norm > 0.0:
                    vector /= norm
            vectors.append(vector)
        array = np.asarray(vectors, dtype="float32")
        return array[0] if single else array


def load_embedding_model(settings: Any) -> tuple[Any, bool]:
    """Resolve the encoder for ``settings``; returns ``(encoder, is_fallback)``.

    Prefers the configured ``sentence-transformers`` model and falls back to
    :class:`HashingEmbeddingEncoder` (same dimension) when the package or the
    weights are unavailable, logging the reason so an operator can tell a real
    embedding model from the deterministic fallback.
    """
    dimension = int(getattr(settings, "embedding_dimension", 384))
    model_name = str(getattr(settings, "embed_model_name", ""))
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers unavailable; using deterministic hashing encoder (dim=%d)", dimension)
        return HashingEmbeddingEncoder(dimension), True
    try:
        model = SentenceTransformer(model_name)
    except Exception:
        logger.exception("Cannot load embedding model %r; using deterministic hashing encoder", model_name)
        return HashingEmbeddingEncoder(dimension), True
    return model, False


class LazyEncoder:
    """Defer the (possibly heavy) encoder load until the first encode call.

    A cache hit only needs the FAISS index for serving; loading a transformer
    model at startup for it would be wasted work. Queries trigger the load on
    first use, through the same :func:`load_embedding_model` factory the build
    path uses, so both paths resolve to one encoder instance.

    The load is **guarded** (ADR 0044 §9): the tool node fans independent
    ``rag_search`` branches out on worker threads, and the first query against
    a warm cache is exactly where two branches reach this check-then-set at
    the same time. Without the lock both threads call the factory, so the
    process pays for two transformer instances (double RAM/VRAM) and one of
    them is leaked the moment the second assignment overwrites it. The lock
    makes "load once" a guarantee; the second thread waits for the in-flight
    load instead of repeating it. The fast path stays lock-free because a
    completed load never changes again.
    """

    def __init__(self, factory: Callable[[], tuple[Any, bool]]) -> None:
        self._factory = factory
        self._encoder: Any | None = None
        self._loaded = False
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Whether the encoder has been resolved (a hint, never a gate)."""
        return self._loaded

    @property
    def encoder(self) -> Any:
        """Force the underlying load and return the resolved encoder."""
        if self._loaded:
            return self._encoder
        with self._load_lock:
            if not self._loaded:
                self._encoder, _ = self._factory()
                self._loaded = True
        return self._encoder

    def encode(self, texts: str | list[str], **kwargs: Any) -> Any:
        return self.encoder.encode(texts, **kwargs)


def lazy_embedding_encoder(settings: Any) -> LazyEncoder:
    """Create the shared deferred encoder for the given RAG settings."""
    return LazyEncoder(lambda: load_embedding_model(settings))
