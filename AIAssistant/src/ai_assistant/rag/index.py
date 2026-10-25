"""Index building, caching, and persistence."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_assistant.config.paths import AppPaths
from ai_assistant.config.rag_config import RAGSettings
from ai_assistant.image_utils import load_image_for_embedding
from ai_assistant.llm.hardware import release_ml_memory
from ai_assistant.path_policy import canonical_path, is_blocked_dir, is_secret_name
from ai_assistant.ports import FieldCipher, StorageCryptoError

from .cache_seal import (
    BM25_CONTEXT,
    CHUNKS_CONTEXT,
    INDEX_CONTEXT,
    is_sealed_metadata,
    read_metadata,
    seal_metadata,
)
from .domain import ChunkResult
from .nlp_utils import tokenize_vn

logger = logging.getLogger("ai_assistant.rag.index")

#: The directories the corpus adds to the **shared** blocked-directory rule
#: (``path_policy.BLOCKED_DIR_NAMES``): they hold artifacts — build output,
#: datasets, run logs — rather than documents. The shared entries are not
#: repeated here: the corpus builder is a second reader of the same tree as
#: the agent's read tools, and it used to carry its own list that *omitted*
#: two of them (``node_modules`` and ``.vs``), so the two readers disagreed
#: about what may be read.
EXCLUDED_DIRS = {
    ".qtcreator", ".cache", "Cache", "runs", "Dicom", "Predict", "3DModels",
    "Dataset", "logs", ".github", ".prompts", ".review", ".tasks", "scripts",
    "3DRecontruction", "3drecontruction", "2DImages", "2dimages",
    "Upload", "upload", "Thumbnails", "thumbnails",
    "tests", "evals",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".agents", ".codex",
}
#: The source-ish files the project walk indexes.
SCANNABLE_EXTS = {".cpp", ".h", ".py", ".md", ".cmake", ".jpg", ".jpeg", ".png", ".webp"}
#: What a docs directory may hold: everything the project walk would index,
#: **plus** the document types. The project walk *skips* the docs directories
#: (they have their own walk, because a docs directory is not necessarily
#: inside the project root), so the docs walk has to cover the same files —
#: otherwise a ``.md`` under ``Docs/`` would simply stop being indexed.
#:
#: The two sets used to overlap on the image extensions *and* the docs
#: directories were walked twice, so an image under ``Docs/`` was chunked
#: twice: the corpus (and the sealed cache) held it twice, and a retrieval
#: could return it twice.
DOC_EXTS = SCANNABLE_EXTS | {".docx", ".pdf", ".txt", ".eml"}


def _is_excluded_dir(name: str) -> bool:
    """Whether the walk must not descend into this directory.

    The shared rule (``path_policy.is_blocked_dir``) owns "never descend into
    this"; the corpus adds the artifact directories. Neither list may omit
    the other's entries.
    """
    return is_blocked_dir(name) or name in EXCLUDED_DIRS


def _stays_inside(root: Path, candidate: Path) -> bool:
    """Whether the file's **real** target stays inside ``root``.

    A junction (on Windows: created without any privilege) or a symbolic link
    inside the tree points somewhere else. The corpus builder followed it and
    indexed the target as if it were part of the project — so a private
    document, or a ``secrets.py`` with an API key, reached the model and the
    sealed cache. The agent read gate has refused this since ADR 0033; the
    corpus builder is a second reader and needs the same rule.
    """
    try:
        canonical_path(candidate).relative_to(canonical_path(root))
    except (OSError, ValueError):
        return False
    return True


def _indexable_files(
    root: Path,
    extensions: set[str],
    *,
    exclude: tuple[Path, ...] = (),
) -> Iterator[str]:
    """Yield the files under ``root`` the corpus may read.

    One rule, one owner: the *agent* read gate already answers "may this file
    be read" (:mod:`ai_assistant.path_policy`), and the corpus builder is a
    second reader of the same filesystem. The extension list, the directory
    list, the secret names and the link rule all come from one place now —
    and the **same** walk feeds the cache hash, so "has the corpus changed?"
    is asked over exactly the files that were indexed.

    ``exclude`` names the roots another walk owns. Every file is walked
    **once**: the docs directories are inside the project root in a source
    checkout, and walking them twice indexed every image under them twice.
    """
    excluded = tuple(canonical_path(path) for path in exclude)
    for dirpath, dirnames, filenames in os.walk(root):
        kept = []
        for name in sorted(dirnames):
            if _is_excluded_dir(name):
                continue
            candidate = Path(dirpath) / name
            if excluded and canonical_path(candidate) in excluded:
                continue
            # Do not descend into a directory that points out of the tree: a
            # junction (creatable on Windows without any privilege) would
            # otherwise make the walk traverse an entire volume — every file
            # it finds is refused afterwards, but the *walk* already paid
            # for it.
            if not _stays_inside(root, candidate):
                continue
            kept.append(name)
        dirnames[:] = kept
        for filename in sorted(filenames):
            if os.path.splitext(filename)[1].lower() not in extensions:
                continue
            if is_secret_name(filename):
                continue
            path = Path(dirpath) / filename
            if not _stays_inside(root, path):
                continue
            yield str(path)


def get_file_system_hash(paths: AppPaths, settings: RAGSettings) -> str:
    """Compute a hash of all scanned files to determine if cache is valid."""
    entries = []
    docs_dirs = tuple(dict.fromkeys(paths.docs_dirs))

    for docs_dir in docs_dirs:
        if not docs_dir.is_dir():
            continue
        for path in _indexable_files(docs_dir, DOC_EXTS):
            try:
                st = os.stat(path)
                entries.append(f"{path}:{st.st_mtime:.3f}:{st.st_size}")
            except OSError:
                pass

    for path in _indexable_files(paths.project_root, SCANNABLE_EXTS, exclude=docs_dirs):
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
        meta = read_metadata(paths.cache_metadata)
        current_hash = get_file_system_hash(paths, settings)
        valid: bool = (meta.get("fs_hash") == current_hash
                       and meta.get("embedding_dimension") == settings.embedding_dimension
                       and is_sealed_metadata(meta))

        if not valid:
            logger.info("Cache stale (built_at=%s)", meta.get("built_at", "?"))
        else:
            logger.info("Cache valid: built_at=%s, chunks=%d",
                        meta.get("built_at", "?"), meta.get("chunk_count", 0))
    except Exception as e:
        logger.warning("Cache read error: %s", e)
        return False
    else:
        return valid


def _serialize_index(faiss_module: Any, index: Any) -> bytes:
    """The index as bytes, so it can be sealed before it reaches the disk.

    ``faiss.write_index`` would put the vectors on disk in the clear first;
    serializing in memory keeps the plaintext index out of the file system
    entirely.
    """
    return bytes(faiss_module.serialize_index(index))


def _deserialize_index(faiss_module: Any, payload: bytes) -> Any:
    """Rebuild the index from sealed bytes."""
    import numpy as np

    return faiss_module.deserialize_index(np.frombuffer(payload, dtype="uint8"))


def save_cache(index: Any, chunks: list[ChunkResult], bm25: Any, paths: AppPaths,
               settings: RAGSettings, model_idx: int = 0, *,
               cipher: FieldCipher) -> None:
    import faiss as _faiss
    t = time.monotonic()

    # The cache directory may not exist yet (first run on a fresh data dir, or
    # a hand-built AppPaths in tests) — create it before writing.
    paths.cache_index.parent.mkdir(parents=True, exist_ok=True)
    # Every artifact is sealed before it is written (ADR 0028): the chunk
    # corpus and the BM25 model hold the user's document text, so a copy of
    # the data volume must not hand over a readable copy of it.
    artifacts = (
        (paths.cache_index, _serialize_index(_faiss, index), INDEX_CONTEXT),
        (paths.cache_chunks, pickle.dumps(chunks, protocol=pickle.HIGHEST_PROTOCOL), CHUNKS_CONTEXT),
        (paths.cache_bm25, pickle.dumps(bm25, protocol=pickle.HIGHEST_PROTOCOL), BM25_CONTEXT),
    )
    for path, payload, context in artifacts:
        path.write_bytes(cipher.seal_bytes(payload, context=context))

    meta = {
        "fs_hash": get_file_system_hash(paths, settings),
        "chunk_count": len(chunks),
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_idx": model_idx,
        "embed_model": settings.embed_model_name,
        "embedding_dimension": settings.embedding_dimension,
        "chunk_chars": settings.chunk_chars,
        "cache_version": settings.cache_version,
        # Records that the artifacts on disk are sealed; a cache without this
        # block is a miss (see ``is_sealed_metadata``).
        "sealed": seal_metadata(),
    }
    with paths.cache_metadata.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    total_mb = sum(p.stat().st_size for p in [paths.cache_index, paths.cache_chunks, paths.cache_bm25]) / 1024**2
    logger.info("Cache saved: %.1fs | %.1fMB | %d chunks", time.monotonic()-t, total_mb, len(chunks))


def load_cache(paths: AppPaths, settings: RAGSettings, *,
               cipher: FieldCipher) -> tuple[Any, list[ChunkResult], Any]:
    import faiss as _faiss
    t = time.monotonic()

    def _read() -> tuple[Any, list[ChunkResult], Any]:
        meta = read_metadata(paths.cache_metadata)
        if not is_sealed_metadata(meta):
            # Written before sealing existed, or by another version: the
            # artifacts cannot be trusted, so the cache is a miss and the
            # caller rebuilds it (the corpus is derived, so this is cheap).
            logger.warning("RAG cache is not sealed — refusing to load %s",
                           paths.cache_chunks)
            return None, [], None
        try:
            index_payload = cipher.open_bytes(
                paths.cache_index.read_bytes(), context=INDEX_CONTEXT)
            chunks_payload = cipher.open_bytes(
                paths.cache_chunks.read_bytes(), context=CHUNKS_CONTEXT)
            bm25_payload = cipher.open_bytes(
                paths.cache_bm25.read_bytes(), context=BM25_CONTEXT)
        except (OSError, StorageCryptoError) as error:
            # Unreadable or not authentic (wrong key, tampered, planted): a
            # miss — never an unpickled payload.
            logger.warning("RAG cache could not be opened: %s", error)
            return None, [], None

        index = _deserialize_index(_faiss, index_payload)
        if index.d != settings.embedding_dimension:
            msg = f"FAISS dimension {index.d} does not match {settings.embedding_dimension}"
            raise ValueError(msg)

        # The bytes were authenticated by the AEAD above: a payload that does
        # not verify is never unpickled, so this cannot run planted code.
        chunks = pickle.loads(chunks_payload)  # noqa: S301
        bm25 = pickle.loads(bm25_payload)  # noqa: S301

        logger.info("Cache loaded: %.1fs | chunks=%d", time.monotonic()-t, len(chunks))
        logger.debug(f"       chunks={len(chunks)}  (from Cache/)")
        return index, chunks, bm25

    try:
        loaded = _read()
    except Exception:
        logger.exception("Cache load failed")
        return None, [], None
    else:
        return loaded


def _embed_image_batches(embed_model: Any, chunks: list[ChunkResult],
                         image_items: list[tuple[int, str]],
                         final_embeddings: list[Any]) -> None:
    """Encode image chunks in small batches with per-image fallback.

    A failed batch retries image by image; an image that fails alone keeps a
    slot the text fallback fills later. Opened images are always closed so no
    handle survives on Windows.
    """
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
            for idx, emb in zip(batch_indices, img_embs, strict=False):
                final_embeddings[idx] = emb
        except Exception as e:
            logger.warning(
                "Image embedding batch failed (%d-%d): %s; retrying one by one",
                start + 1, start + len(batch_items), e
            )
            for idx, img in zip(batch_indices, batch_images, strict=False):
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
        for idx, emb in zip(text_indices, text_embs, strict=False):
            final_embeddings[idx] = emb
            
    if image_items:
        _embed_image_batches(embed_model, chunks, image_items, final_embeddings)

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
        for idx, emb in zip(missing_indices, fallback_embs, strict=False):
            final_embeddings[idx] = emb

    embeddings = final_embeddings
    logger.debug(f"\n ✓ Encoding complete ({time.monotonic()-t_enc:.1f}s)")

    emb = np.array(embeddings, dtype="float32")
    dim, n = emb.shape[1], len(emb)
    if dim != settings.embedding_dimension:
        msg = f"Embedding model returned {dim} dimensions; expected {settings.embedding_dimension}"
        raise ValueError(msg)

    # Annotated as the faiss base class: below 1000 vectors the index is a
    # flat one, above it an IVF one — with faiss >= 1.14 shipping type stubs,
    # mypy sees the two concrete classes as incompatible unless the shared
    # variable carries the common base type.
    index: _faiss.Index
    if n < 1000:
        index = _faiss.IndexFlatIP(dim)
        _faiss.normalize_L2(emb)
        index.add(emb)
    else:
        nlist = min(int(n**0.5), 256)
        quantizer = _faiss.IndexFlatIP(dim)
        ivf = _faiss.IndexIVFFlat(quantizer, dim, nlist, _faiss.METRIC_INNER_PRODUCT)
        normed = emb.copy()
        _faiss.normalize_L2(normed)
        ivf.train(normed)
        ivf.add(normed)
        # nprobe is IVF-only, hence the concrete variable rather than the
        # base-typed one.
        ivf.nprobe = min(16, nlist)
        index = ivf

    logger.info("FAISS built: ntotal=%d dim=%d", index.ntotal, dim)

    tokenized = [tokenize_vn(getattr(c, "text", str(c))) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    logger.info("Index built: %.1fs total", time.monotonic()-t)
    return index, chunks, bm25


def load_or_build_index(chunks: list[ChunkResult], embed_model: Any, paths: AppPaths, settings: RAGSettings, force_rebuild: bool = False, model_idx: int = 0, *, cipher: FieldCipher) -> tuple[Any, list[ChunkResult], Any]:
    if not force_rebuild and is_cache_valid(paths, settings):
        logger.debug("       [CACHE HIT]")
        index, chunks_loaded, bm25 = load_cache(paths, settings, cipher=cipher)
        if index is not None:
            return index, chunks_loaded, bm25
        logger.debug(" load failed, rebuilding...")
    logger.debug("       [CACHE MISS] Building index...")
    logger.info("Cache miss — rebuilding")

    index, chunks_loaded, bm25 = build_index_from_scratch(chunks, embed_model, settings)
    save_cache(index, chunks_loaded, bm25, paths, settings, model_idx, cipher=cipher)
    return index, chunks_loaded, bm25


def scan_documents(paths: AppPaths, registry: Any) -> list[ChunkResult]:
    """Scan all project paths and return chunked documents."""
    all_chunks: list[ChunkResult] = []
    stats = {"docx": 0, "pdf": 0, "txt": 0, "eml": 0, "md": 0, "source": 0, "errors": 0, "files": 0}

    for docs_dir in paths.docs_dirs:
        if docs_dir.is_dir():
            for fp in _indexable_files(docs_dir, DOC_EXTS):
                if os.path.basename(fp).startswith("~") or fp.endswith(".user"):
                    continue
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

    # The docs directories have their own walk (they may live outside the
    # project root), so the project walk skips them: every file is walked
    # once, and an image under ``Docs/`` is not chunked twice.
    docs_dirs = tuple(dict.fromkeys(paths.docs_dirs))
    for fp in _indexable_files(paths.project_root, SCANNABLE_EXTS, exclude=docs_dirs):
        if os.path.basename(fp).startswith("~") or fp.endswith(".user"):
            continue
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
