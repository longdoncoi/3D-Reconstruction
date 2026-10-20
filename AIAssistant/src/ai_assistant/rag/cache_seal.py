"""Sealed RAG cache artifacts (ADR 0028).

The cache directory holds a second copy of the user's corpus: ``chunks.pkl``
carries the full text of every indexed document plus the base64 of every
image, ``bm25.pkl`` carries the tokenized corpus and ``faiss_index.bin``
carries the embeddings. All three are *derived* artifacts — they can always be
rebuilt from the source documents — but they were written as plaintext
pickles, so a copy of the data volume also handed over a readable copy of
everything the user had indexed.

They are now sealed with the same AES-256-GCM field cipher the task and
pending stores use (ADR 0027): one key, one AEAD, and the cache is both
authenticated and confidential. The HMAC that used to guard it is gone — GCM
authenticates the bytes themselves, so a planted, truncated or foreign-key
artifact is a cache *miss* (rebuilt) instead of an unpickled payload.

The failure mode is deliberately the cheap one: a cache that cannot be
authenticated is discarded and rebuilt, never partially trusted. The cache is
a derived artifact, so throwing it away costs time, not data.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("ai_assistant.rag.cache_seal")

#: Associated-data context per artifact. The context is part of the AEAD
#: authentication, so an envelope moved to another artifact fails to open
#: instead of being read as if it belonged there.
INDEX_CONTEXT = "rag.index"
CHUNKS_CONTEXT = "rag.chunks"
BM25_CONTEXT = "rag.bm25"

#: Bumped when the on-disk format changes. A cache written by another version
#: is a miss, not something to guess at.
SEALED_CACHE_VERSION = 2
_ALGO = "aes-256-gcm"


def seal_metadata() -> dict[str, str]:
    """The metadata fragment ``save_cache`` records for the files it wrote."""
    return {"algo": _ALGO, "version": str(SEALED_CACHE_VERSION)}


def is_sealed_metadata(meta: dict[str, Any]) -> bool:
    """Whether the recorded metadata says the artifacts were sealed.

    A cache written before sealing existed carries no such block: it is
    unverifiable, so it is a miss and the next build replaces it with a
    sealed one.
    """
    sealed = meta.get("sealed")
    if not isinstance(sealed, dict):
        return False
    return (
        sealed.get("algo") == _ALGO
        and sealed.get("version") == str(SEALED_CACHE_VERSION)
    )


def read_metadata(path: Path) -> dict[str, Any]:
    """The cache metadata, or an empty mapping when it cannot be read."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}
