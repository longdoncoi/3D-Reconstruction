"""
Compatibility shim for the legacy rag_module.
New code should use ai_assistant.rag package directly.
"""

import threading
from types import SimpleNamespace

# --- NEW PLATFORM INTEGRATION ---
from ai_assistant.llm.hardware import release_ml_memory as _release_ml_memory
from ai_assistant.rag import (
    ChunkResult,  # noqa: F401
    DocumentLoaderRegistry,
    RetrievalPipeline,
    build_index_from_scratch,
    is_cache_valid,
    load_cache,
    save_cache,
    scan_documents,
)
from ai_assistant.rag import (
    image_to_data_uri as _image_to_data_uri,  # noqa: F401
)
from ai_assistant.rag import (
    is_image_file as _is_image_file,  # noqa: F401
)
from ai_assistant.rag.nlp_utils import tokenize_vn as _tokenize_vn

# Import configurations
from .config import (
    CHUNK_CHARS,  # noqa: F401
    EMBED_CACHE,
    EMBED_MODEL_NAME,
    ENABLE_RAG,
    MAX_CONTEXT_CHARS,  # noqa: F401
    RERANKER_MODEL,
    RERANKER_TOP_K,
    SIMILARITY_THRESHOLD,
    USE_RERANKER,
    _paths,
    _rag_settings,
    logger,
    startup_step,
)

knowledge_index = None
rag_lock = threading.RLock()
vector_retriever = None
bm25_retriever = None
knowledge_chunks: list = []
bm25_index = None
embed_model_ref = None
_reranker = None

_pipeline: RetrievalPipeline | None = None

class _FaissRetriever:
    """Expose a cached FAISS index through the retriever protocol."""

    def __init__(self, index, encoder, query_prefix: str, top_k: int = 30):
        self._index = index
        self._encoder = encoder
        self._query_prefix = query_prefix
        self._top_k = top_k

    def retrieve(self, query: str) -> list[SimpleNamespace]:
        import faiss
        import numpy as np

        vector = np.asarray(
            self._encoder.encode(
                [self._query_prefix + query],
                show_progress_bar=False,
                normalize_embeddings=True,
            ),
            dtype="float32",
        )
        faiss.normalize_L2(vector)
        scores, identifiers = self._index.search(
            vector, min(self._top_k, self._index.ntotal)
        )
        return [
            SimpleNamespace(
                node=SimpleNamespace(metadata={"chunk_index": int(identifier)}),
                score=float(score),
            )
            for score, identifier in zip(scores[0], identifiers[0])
            if identifier >= 0
        ]


class _Bm25Retriever:
    """Expose a cached rank_bm25 index through the retriever protocol."""

    def __init__(self, index, top_k: int = 30):
        self._index = index
        self._top_k = top_k

    def retrieve(self, query: str) -> list[SimpleNamespace]:
        scores = self._index.get_scores(_tokenize_vn(query))
        ranked = sorted(
            enumerate(scores), key=lambda item: item[1], reverse=True
        )[:self._top_k]
        return [
            SimpleNamespace(
                node=SimpleNamespace(metadata={"chunk_index": chunk_index}),
                score=float(score),
            )
            for chunk_index, score in ranked
        ]


def _build_retrievers(index, bm25, encoder):
    return (
        _FaissRetriever(index, encoder, _rag_settings.embedding_query_prefix),
        _Bm25Retriever(bm25),
    )

def initialize_rag(force_rebuild: bool = False, enable_reranker: bool = True) -> int:
    global knowledge_index, knowledge_chunks, bm25_index, embed_model_ref, _reranker, vector_retriever, bm25_retriever, _pipeline

    if not ENABLE_RAG:
        with rag_lock:
            knowledge_index, knowledge_chunks, bm25_index = None, [], None
            vector_retriever, bm25_retriever = None, None
            embed_model_ref, _reranker = None, None
            _pipeline = None
        return 0

    from sentence_transformers import CrossEncoder, SentenceTransformer
    with startup_step(f"Loading embedding model ({EMBED_MODEL_NAME})"):
        embed_model = SentenceTransformer(EMBED_MODEL_NAME, cache_folder=EMBED_CACHE)
        
    reranker = None
    if USE_RERANKER and enable_reranker:
        try:
            with startup_step(f"Loading reranker ({RERANKER_MODEL})"):
                reranker = CrossEncoder(RERANKER_MODEL, max_length=512, cache_folder=EMBED_CACHE)
        except Exception as error:
            logger.warning("Reranker load failed (%s); continuing without it", error)

    index, chunks, bm25 = None, [], None
    if not force_rebuild and is_cache_valid(_paths, _rag_settings):
        with startup_step("Loading RAG index cache"):
            index, chunks, bm25 = load_cache(_paths, _rag_settings)

    if index is None:
        with startup_step("Scanning documents"):
            registry = DocumentLoaderRegistry.create_default(
                chunk_chars=_rag_settings.chunk_chars,
                overlap_chars=_rag_settings.overlap_chars,
            )
            raw_chunks = scan_documents(_paths, registry)

        if raw_chunks:
            step_name = "Rebuilding RAG index" if force_rebuild else "Building RAG index"
            with startup_step(step_name):
                index, chunks, bm25 = build_index_from_scratch(
                    raw_chunks, embed_model, _rag_settings
                )
                save_cache(index, chunks, bm25, _paths, _rag_settings)
        else:
            logger.warning("No documents found; RAG context will be empty")

    if index is not None:
        dense, sparse = _build_retrievers(index, bm25, embed_model)
    else:
        dense, sparse = None, None

    with rag_lock:
        knowledge_index, knowledge_chunks, bm25_index = index, chunks, bm25
        vector_retriever, bm25_retriever = dense, sparse
        embed_model_ref, _reranker = embed_model, reranker
        
        _pipeline = RetrievalPipeline(
            vector_retriever=dense,
            bm25_retriever=sparse,
            reranker=reranker,
            knowledge_chunks=chunks,
            max_context_chars=_rag_settings.max_context_chars,
            similarity_threshold=SIMILARITY_THRESHOLD,
            use_reranker=USE_RERANKER,
            reranker_top_k=RERANKER_TOP_K
        )
        
    return len(chunks)


def release_embedding_for_vision() -> None:
    global embed_model_ref
    with rag_lock:
        embed_model_ref = None
    _release_ml_memory()


def get_context(query: str, query_image_b64: str | None = None, result_k: int = RERANKER_TOP_K) -> tuple:
    if not ENABLE_RAG or _pipeline is None:
        return "", "", []
    return _pipeline.get_context(query, query_image_b64, result_k)


# Additional exports for backward compatibility
def build_registry():
    return DocumentLoaderRegistry.create_default(
        chunk_chars=_rag_settings.chunk_chars,
        overlap_chars=_rag_settings.overlap_chars
    )
