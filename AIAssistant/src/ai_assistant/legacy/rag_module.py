# ruff: noqa: F401
"""
Compatibility shim for the legacy rag_module.
New code should use ai_assistant.rag package directly.
"""

import os
import threading

# --- NEW PLATFORM INTEGRATION ---
from ai_assistant.llm.hardware import release_ml_memory as _release_ml_memory
from ai_assistant.rag import (
    ChunkResult,
    DocumentLoaderRegistry,
    RetrievalPipeline,
    scan_documents,
)
from ai_assistant.rag import (
    E5LlamaEmbedding as _E5LlamaEmbedding,
)
from ai_assistant.rag import (
    image_to_data_uri as _image_to_data_uri,
)
from ai_assistant.rag import (
    is_image_file as _is_image_file,
)
from ai_assistant.rag.nlp_utils import tokenize_vn as _tokenize_vn

# Import configurations
from .config import (
    CACHE_DIR,
    CHUNK_CHARS,
    EMBED_CACHE,
    EMBED_MODEL_NAME,
    EMBEDDING_DIMENSION,
    ENABLE_RAG,
    MAX_CONTEXT_CHARS,
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
is_vision_model = False

_pipeline: RetrievalPipeline | None = None

def build_llamaindex_runtime(chunks: list, encoder):
    import faiss
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.schema import TextNode
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.vector_stores.faiss import FaissVectorStore

    nodes = [TextNode(text=chunk.text, metadata={"chunk_index": index})
             for index, chunk in enumerate(chunks) if not chunk.is_image]
    vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(EMBEDDING_DIMENSION))
    storage = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage,
                             embed_model=_E5LlamaEmbedding(
                                 encoder,
                                 _rag_settings.embedding_query_prefix,
                                 _rag_settings.embedding_passage_prefix
                             ).value,
                             show_progress=False)
    storage.persist(persist_dir=os.path.join(CACHE_DIR, "llamaindex"))
    return index, index.as_retriever(similarity_top_k=30), BM25Retriever.from_defaults(
        docstore=index.docstore, similarity_top_k=30, tokenizer=_tokenize_vn, skip_stemming=True)

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

    with startup_step("Scanning documents"):
        registry = DocumentLoaderRegistry.create_default(
            chunk_chars=_rag_settings.chunk_chars,
            overlap_chars=_rag_settings.overlap_chars
        )
        raw_chunks = scan_documents(_paths, registry)
        
    if raw_chunks:
        with startup_step("Rebuilding RAG index" if force_rebuild else "Loading/building RAG index"):
            # During migration, we still use the legacy llama_index integration for index building
            # to preserve compatibility with existing disk caches. This can be completely replaced later.
            index, dense, sparse = build_llamaindex_runtime(raw_chunks, embed_model)
            chunks, bm25 = raw_chunks, sparse
    else:
        index, chunks, bm25, dense, sparse = None, [], None, None, None
        logger.warning("No documents found; RAG context will be empty")

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
