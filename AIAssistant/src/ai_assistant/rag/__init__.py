"""RAG (Retrieval-Augmented Generation) package."""

from .domain import ChunkResult
from .embeddings import E5LlamaEmbedding
from .image_utils import IMAGE_EXTS, image_to_data_uri, is_image_file, load_image_for_embedding
from .index import build_index_from_scratch, is_cache_valid, load_cache, load_or_build_index, save_cache, scan_documents
from .loaders import BaseDocumentLoader, DocumentLoaderRegistry
from .nlp_utils import expand_project_role_query, tokenize_vn
from .retrieval import (
    RetrievalPipeline,
    dedup_by_source,
    expand_with_parent,
    format_context_block,
    hybrid_retrieve,
    rerank_chunks,
    rrf_fuse,
)
from .retrievers import BM25Retriever, FaissVectorRetriever, build_retrievers

__all__ = [
    "IMAGE_EXTS",
    "BM25Retriever",
    "BaseDocumentLoader",
    "ChunkResult",
    "DocumentLoaderRegistry",
    "E5LlamaEmbedding",
    "FaissVectorRetriever",
    "RetrievalPipeline",
    "build_index_from_scratch",
    "build_retrievers",
    "dedup_by_source",
    "expand_project_role_query",
    "expand_with_parent",
    "format_context_block",
    "hybrid_retrieve",
    "image_to_data_uri",
    "is_cache_valid",
    "is_image_file",
    "load_cache",
    "load_image_for_embedding",
    "load_or_build_index",
    "rerank_chunks",
    "rrf_fuse",
    "save_cache",
    "scan_documents",
    "tokenize_vn",
]
