"""HMAC integrity for the on-disk RAG cache (pickle + FAISS artifacts).

``load_cache`` unpickles ``chunks.pkl`` and ``bm25.pkl`` and hands the result
straight to the retrieval pipeline, so whatever is in those files runs
in-process. Nothing checked that the bytes were the ones this installation
wrote: a truncated copy, a cache from another profile or a deliberately
planted file was simply unpickled. The cache is now sealed with an HMAC and a
cache whose digest does not match is treated as a miss (rebuilt), never read.

Threat model, stated honestly (ADR 0015): the key lives *outside* the cache
dir, in ``<data_dir>/.secrets/cache_hmac_key`` (mode 0600). A process running
as this same user can usually read that key too, so this is not a defence
against a local attacker with full account access — it is a defence against
tampering *without* read access and against silent corruption (bad sector,
half-copied cache, files dropped in from another install). It turns "tampered
cache executed code" into "cache miss and a rebuild".

A cache written before this existed carries no ``integrity`` record: it is
unverifiable, so it is refused once and re-sealed by the next build.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from pathlib import Path

from ai_assistant.config.paths import AppPaths

logger = logging.getLogger("ai_assistant.rag.cache_integrity")

#: Relative to ``AppPaths.data_dir`` — deliberately not in the cache dir,
#: which is the thing being protected and may be wiped independently.
KEY_FILE = Path(".secrets") / "cache_hmac_key"
ALGO = "hmac-sha256"
_KEY_BYTES = 32
_BLOCK_BYTES = 1 << 20

__all__ = ["ALGO", "cache_digest", "recorded_digest", "seal_payload", "signing_key", "verify_cache"]


def signing_key(paths: AppPaths) -> bytes | None:
    """Read the machine-local key, minting it on first use.

    ``None`` means the data dir cannot hold a key (read-only or full disk);
    the caller then refuses to trust any cache rather than reading it blind.
    """
    key_path = paths.data_dir / KEY_FILE
    try:
        existing = key_path.read_bytes()
    except OSError:
        existing = b""
    if len(existing) >= _KEY_BYTES:
        return existing[:_KEY_BYTES]

    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key = os.urandom(_KEY_BYTES)
        key_path.write_bytes(key)
    except OSError:
        logger.warning("RAG cache key unavailable at %s — caches cannot be verified", key_path)
        return None
    try:
        # Best effort: Windows and some filesystems have no POSIX modes.
        os.chmod(key_path, 0o600)
    except OSError:
        pass
    logger.info("RAG cache signing key created at %s", key_path)
    return key


def cache_digest(key: bytes, paths: AppPaths) -> str:
    """HMAC-SHA256 over the three cache artifacts, in a fixed order.

    A missing artifact contributes only its name, so a digest computed while a
    file was absent can never equal one recorded while it existed.
    """
    digest = hmac.new(key, digestmod=hashlib.sha256)
    for artifact in (paths.cache_index, paths.cache_chunks, paths.cache_bm25):
        digest.update(artifact.name.encode("utf-8"))
        try:
            with artifact.open("rb") as handle:
                for block in iter(lambda: handle.read(_BLOCK_BYTES), b""):
                    digest.update(block)
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()


def seal_payload(key: bytes, paths: AppPaths) -> dict[str, str]:
    """The metadata fragment ``save_cache`` records for the files it wrote."""
    return {"algo": ALGO, "digest": cache_digest(key, paths)}


def recorded_digest(paths: AppPaths) -> str | None:
    """The digest stored with the cache, or ``None`` when it cannot be read."""
    try:
        with paths.cache_metadata.open("r", encoding="utf-8") as handle:
            integrity = json.load(handle).get("integrity")
    except (OSError, ValueError):
        return None
    if not isinstance(integrity, dict):
        return None
    return str(integrity.get("digest") or "") or None


def verify_cache(key: bytes | None, paths: AppPaths, recorded: str | None) -> bool:
    """Whether the on-disk cache still matches the digest written with it.

    An unverifiable cache (no key, or no digest recorded) fails closed: the
    caller rebuilds instead of trusting it.
    """
    if key is None or not recorded:
        return False
    return hmac.compare_digest(cache_digest(key, paths), recorded)
