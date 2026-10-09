"""Index building, caching, and persistence."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from datetime import datetime
from typing import Any

from ai_assistant.config.paths import AppPaths
from ai_assistant.config.rag_config import RAGSettings
from ai_assistant.llm.hardware import release_ml_memory

from .domain import ChunkResult
from .image_utils import load_image_for_embedding
from .nlp_utils import tokenize_vn

logger = logging.getLogger("ai_assistant.rag.index")

EXCLUDED_DIRS = {
    ".git", "build", "__pycache__", ".qtcreator", ".cache", "Cache",
    "runs", "Dicom", "Predict", "3DModels", "Dataset", "logs",
    ".github", ".prompts", ".review", ".tasks", "scripts"
}
SCANNABLE_EXTS = {".cpp", ".h", ".py", ".md", ".cmake", ".jpg", ".jpeg", ".png", ".webp"}
DOC_EXTS = {".docx", ".pdf", ".txt", ".eml", ".jpg", ".jpeg", ".png", ".webp"}


def get_file_system_hash(paths: AppPaths, settings: RAGSettings) -> str:
    """Compute a hash of all scanned files to determine if cache is valid."""
    entries = []
    
    for docs_dir in dict.fromkeys(paths.docs_dirs):
        if not docs_dir.is_dir():
            continue
        for root, _, files in os.walk(docs_dir):
            for f in sorted(files):
                path = os.path.join(root, f)
                try:
                    st = os.stat(path)
                    entries.append(f"{path}:{st.st_mtime:.3f}:{st.st_size}")
                except OSError:
                    pass

    for root, dirs, files in os.walk(paths.project_root):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() not in SCANNABLE_EXTS:
                continue
            path = os.path.join(root, f)
            try:
                st = os.stat(path)
                entries.append(f"{path}:{st.st_mtime:.3f}:{st.st_size}")
            except OSError:
                pass

    entries.append(f"embed_model={settings.embed_model_name}")
    entries.append(f"chunk_chars={settings.chunk_chars}")
    entries.append(f"cache_version={settings.cache_version}")

    combined = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def is_cache_valid(paths: AppPaths, settings: RAGSettings) -> bool:
    required = [paths.cache_index, paths.cache_chunks, paths.cache_bm25, paths.cache_metadata]
    if not all(p.exists() for p in required):
        logger.debug("Cache miss: files missing")
        return False
        
    try:
        with paths.cache_metadata.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        current_hash = get_file_system_hash(paths, settings)
        valid = (meta.get("fs_hash") == current_hash
                 and meta.get("embedding_dimension") == settings.embedding_dimension)
                 
        if not valid:
            logger.info("Cache stale (built_at=%s)", meta.get("built_at", "?"))
        else:
            logger.info("Cache valid: built_at=%s, chunks=%d",
                        meta.get("built_at", "?"), meta.get("chunk_count", 0))
        return valid
    except Exception as e:
        logger.warning("Cache read error: %s", e)
        return False


def save_cache(index: Any, chunks: list[ChunkResult], bm25: Any, paths: AppPaths, settings: RAGSettings, model_idx: int = 0) -> None:
    import faiss as _faiss
    t = time.monotonic()
    
    _faiss.write_index(index, str(paths.cache_index))
    
    with paths.cache_chunks.open("wb") as f:
        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    with paths.cache_bm25.open("wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    meta = {
        "fs_hash": get_file_system_hash(paths, settings),
        "chunk_count": len(chunks),
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_idx": model_idx,
        "embed_model": settings.embed_model_name,
        "embedding_dimension": settings.embedding_dimension,
        "chunk_chars": settings.chunk_chars,
        "cache_version": settings.cache_version,
    }
    
    with paths.cache_metadata.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        
    total_mb = sum(p.stat().st_size for p in [paths.cache_index, paths.cache_chunks, paths.cache_bm25]) / 1024**2
    logger.info("Cache saved: %.1fs | %.1fMB | %d chunks", time.monotonic()-t, total_mb, len(chunks))


def load_cache(paths: AppPaths, settings: RAGSettings) -> tuple[Any, list[ChunkResult], Any]:
    import faiss as _faiss
    t = time.monotonic()
    try:
        index = _faiss.read_index(str(paths.cache_index))
        if index.d != settings.embedding_dimension:
            raise ValueError(f"FAISS dimension {index.d} does not match {settings.embedding_dimension}")
            
        with paths.cache_chunks.open("rb") as f:
            chunks = pickle.load(f)  # noqa: S301 - trusted local cache
            
        with paths.cache_bm25.open("rb") as f:
            bm25 = pickle.load(f)  # noqa: S301 - trusted local cache
            
        logger.info("Cache loaded: %.1fs | chunks=%d", time.monotonic()-t, len(chunks))
        logger.debug(f"       chunks={len(chunks)}  (from Cache/)")
        return index, chunks, bm25
    except Exception as e:
        logger.error("Cache load failed: %s", e)
        return None, [], None


def build_index_from_scratch(chunks: list[ChunkResult], embed_model: Any, settings: RAGSettings) -> tuple[Any, list[ChunkResult], Any]:
    import faiss as _faiss
    import numpy as np
    from rank_bm25 import BM25Okapi

    t = time.monotonic()
    logger.info("Building index: %d chunks", len(chunks))

    t_enc = time.monotonic()
    text_indices = []
    texts = []
    image_items = []

    for i, c in enumerate(chunks):
        if getattr(c, "is_image", False) and settings.embedding_supports_images:
            image_items.append((i, getattr(c, "source_path", "")))
        else:
            texts.append(settings.embedding_passage_prefix + getattr(c, "text", str(c)))
            text_indices.append(i)

    final_embeddings = [None] * len(chunks)
    
    if texts:
        logger.debug(f"Encoding {len(texts)} text chunks")
        text_embs = embed_model.encode(texts, show_progress_bar=False, normalize_embeddings=True, batch_size=64)
        for idx, emb in zip(text_indices, text_embs):
            final_embeddings[idx] = emb
            
    if image_items:
        image_batch_size = 4
        logger.debug(f"Encoding {len(image_items)} image chunks")
        for start in range(0, len(image_items), image_batch_size):
            batch_items = image_items[start:start + image_batch_size]
            batch_indices = []
            batch_images = []
            for idx, image_path in batch_items:
                try:
                    batch_images.append(load_image_for_embedding(image_path))
                    batch_indices.append(idx)
                except Exception as e:
                    logger.warning("Image load skipped for embedding: %s | %s", image_path, e)

            if not batch_images:
                continue

            try:
                img_embs = embed_model.encode(
                    batch_images,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=len(batch_images)
                )
                for idx, emb in zip(batch_indices, img_embs):
                    final_embeddings[idx] = emb
            except Exception as e:
                logger.warning(
                    "Image embedding batch failed (%d-%d): %s; retrying one by one",
                    start + 1, start + len(batch_items), e
                )
                for idx, img in zip(batch_indices, batch_images):
                    try:
                        emb = embed_model.encode(
                            [img],
                            show_progress_bar=False,
                            normalize_embeddings=True,
                            batch_size=1
                        )[0]
                        final_embeddings[idx] = emb
                    except Exception as single_e:
                        logger.warning(
                            "Image embedding skipped for %s: %s",
                            getattr(chunks[idx], "source_path", idx), single_e
                        )
            finally:
                for img in batch_images:
                    try:
                        img.close()
                    except Exception:
                        pass
                release_ml_memory()

    missing_indices = [i for i, emb in enumerate(final_embeddings) if emb is None]
    if missing_indices:
        logger.warning("Falling back to text embeddings for %d chunks", len(missing_indices))
        fallback_texts = [getattr(chunks[i], "text", str(chunks[i])) for i in missing_indices]
        fallback_embs = embed_model.encode(
            fallback_texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=32
        )
        for idx, emb in zip(missing_indices, fallback_embs):
            final_embeddings[idx] = emb

    embeddings = final_embeddings
    logger.debug(f"\n ✓ Encoding complete ({time.monotonic()-t_enc:.1f}s)")

    emb = np.array(embeddings, dtype="float32")
    dim, n = emb.shape[1], len(emb)
    if dim != settings.embedding_dimension:
        raise ValueError(f"Embedding model returned {dim} dimensions; expected {settings.embedding_dimension}")

    if n < 1000:
        index = _faiss.IndexFlatIP(dim)
        _faiss.normalize_L2(emb)
        index.add(emb)
    else:
        nlist = min(int(n**0.5), 256)
        quantizer = _faiss.IndexFlatIP(dim)
        index = _faiss.IndexIVFFlat(quantizer, dim, nlist, _faiss.METRIC_INNER_PRODUCT)
        normed = emb.copy()
        _faiss.normalize_L2(normed)
        index.train(normed)
        index.add(normed)
        index.nprobe = min(16, nlist)

    logger.info("FAISS built: ntotal=%d dim=%d", index.ntotal, dim)

    tokenized = [tokenize_vn(getattr(c, "text", str(c))) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    logger.info("Index built: %.1fs total", time.monotonic()-t)
    return index, chunks, bm25


def load_or_build_index(chunks: list[ChunkResult], embed_model: Any, paths: AppPaths, settings: RAGSettings, force_rebuild: bool = False, model_idx: int = 0) -> tuple[Any, list[ChunkResult], Any]:
    if not force_rebuild and is_cache_valid(paths, settings):
        logger.debug("       [CACHE HIT]")
        index, chunks_loaded, bm25 = load_cache(paths, settings)
        if index is not None:
            return index, chunks_loaded, bm25
        logger.debug(" load failed, rebuilding...")
    logger.debug("       [CACHE MISS] Building index...")
    logger.info("Cache miss — rebuilding")
    
    index, chunks_loaded, bm25 = build_index_from_scratch(chunks, embed_model, settings)
    save_cache(index, chunks_loaded, bm25, paths, settings, model_idx)
    return index, chunks_loaded, bm25


def scan_documents(paths: AppPaths, registry: Any) -> list[ChunkResult]:
    """Scan all project paths and return chunked documents."""
    all_chunks: list[ChunkResult] = []
    stats = {"docx": 0, "pdf": 0, "txt": 0, "eml": 0, "md": 0, "source": 0, "errors": 0, "files": 0}

    for docs_dir in paths.docs_dirs:
        if docs_dir.is_dir():
            for root, _, files in os.walk(docs_dir):
                for filename in sorted(files):
                    if os.path.splitext(filename)[1].lower() not in DOC_EXTS:
                        continue
                    fp = os.path.join(root, filename)
                    stats["files"] += 1
                    results = registry.load_file(fp, str(paths.project_root))
                    if not results:
                        stats["errors"] += 1
                        continue
                    for r in results:
                        all_chunks.append(r)
                        ext = os.path.splitext(fp)[1].lower().lstrip(".")
                        if ext in stats:
                            stats[ext] += 1

    for root, dirs, files in os.walk(paths.project_root):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for filename in sorted(files):
            if filename.startswith("~") or filename.endswith(".user"):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SCANNABLE_EXTS:
                continue
            fp = os.path.join(root, filename)
            stats["files"] += 1
            for r in registry.load_file(fp, str(paths.project_root)):
                all_chunks.append(r)
                stats["md" if r.loader_type == "md" else "source"] += 1

    logger.info(
        "Scanned: %d files → %d chunks (docx=%d pdf=%d txt=%d md=%d src=%d err=%d)",
        stats["files"], len(all_chunks),
        stats["docx"], stats["pdf"], stats["txt"],
        stats["md"], stats["source"], stats["errors"]
    )
    logger.debug(f"       files={stats['files']}  chunks={len(all_chunks)}"
          f"  (docx={stats['docx']} pdf={stats['pdf']} txt={stats['txt']} eml={stats['eml']}"
          f" md={stats['md']} src={stats['source']} err={stats['errors']})")
    return all_chunks
