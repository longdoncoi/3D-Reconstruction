"""Unit tests for the RAG cache seal (``rag/cache_integrity.py``).

``load_cache`` unpickles whatever sits in the cache dir, so the seal is the
only thing standing between a planted file and the retrieval pipeline. These
tests pin the parts the index tests cannot: how the key is minted and reused
(including when the data dir refuses to hold one), how the digest reacts to a
changed or missing artifact, and every way the recorded digest can be absent
or unverifiable — each of which has to fail closed.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_assistant.config.paths import AppPaths
from ai_assistant.rag.cache_integrity import (
    cache_digest,
    recorded_digest,
    seal_payload,
    signing_key,
    verify_cache,
)


def _paths(tmp: str) -> AppPaths:
    base = Path(tmp)
    cache = base / "Cache"
    cache.mkdir(parents=True, exist_ok=True)
    return AppPaths(
        project_root=base / "project",
        base_dir=base,
        data_dir=base,
        models_dir=base / "Models",
        cache_dir=cache,
        logs_dir=base / "logs",
        docs_dirs=(base / "docs",),
    )


class SigningKeyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.paths = _paths(self._tmp.name)
        self.key_path = self.paths.data_dir / ".secrets" / "cache_hmac_key"

    def test_the_key_is_minted_once_and_then_reused(self) -> None:
        first = signing_key(self.paths)
        second = signing_key(self.paths)

        self.assertEqual(len(first), 32)
        self.assertEqual(first, second)
        self.assertEqual(self.key_path.read_bytes(), first)

    def test_a_truncated_key_is_replaced(self) -> None:
        """A half-written key (crash mid-write) must not become the key."""
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path.write_bytes(b"short")

        key = signing_key(self.paths)

        self.assertEqual(len(key), 32)
        self.assertEqual(self.key_path.read_bytes(), key)

    def test_a_data_dir_that_cannot_hold_a_key_returns_none(self) -> None:
        """Read-only disk: the caller then refuses to trust any cache."""
        with mock.patch.object(Path, "write_bytes", side_effect=OSError("read-only")):
            self.assertIsNone(signing_key(self.paths))

    def test_a_filesystem_without_modes_still_gets_a_key(self) -> None:
        with mock.patch(
            "ai_assistant.rag.cache_integrity.os.chmod", side_effect=OSError("no modes")
        ):
            key = signing_key(self.paths)

        self.assertEqual(len(key), 32)


class CacheDigestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.paths = _paths(self._tmp.name)
        self.key = b"k" * 32

    def _write(self, name: str, payload: bytes) -> None:
        getattr(self.paths, name).write_bytes(payload)

    def test_the_digest_is_stable_and_covers_every_artifact(self) -> None:
        self._write("cache_index", b"index")
        self._write("cache_chunks", b"chunks")
        self._write("cache_bm25", b"bm25")

        first = cache_digest(self.key, self.paths)
        self.assertEqual(first, cache_digest(self.key, self.paths))
        self.assertEqual(len(first), 64)

        self.paths.cache_bm25.write_bytes(b"other")
        self.assertNotEqual(first, cache_digest(self.key, self.paths))

    def test_a_different_key_gives_a_different_digest(self) -> None:
        self._write("cache_chunks", b"chunks")

        self.assertNotEqual(
            cache_digest(b"k" * 32, self.paths), cache_digest(b"z" * 32, self.paths)
        )

    def test_a_missing_artifact_still_digests_but_never_matches_an_existing_one(self) -> None:
        """Name-only coverage for a missing file cannot equal a full digest."""
        self._write("cache_chunks", b"chunks")
        without_index = cache_digest(self.key, self.paths)

        self._write("cache_index", b"index")
        with_index = cache_digest(self.key, self.paths)

        self.assertEqual(len(without_index), 64)
        self.assertNotEqual(without_index, with_index)


class RecordedDigestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.paths = _paths(self._tmp.name)

    def _write_meta(self, payload: object) -> None:
        self.paths.cache_metadata.write_text(json.dumps(payload), encoding="utf-8")

    def test_the_sealed_digest_round_trips(self) -> None:
        key = signing_key(self.paths)
        self._write_meta({"integrity": seal_payload(key, self.paths)})

        self.assertEqual(recorded_digest(self.paths), cache_digest(key, self.paths))

    def test_a_missing_metadata_file_has_no_digest(self) -> None:
        self.assertIsNone(recorded_digest(self.paths))

    def test_unparseable_metadata_has_no_digest(self) -> None:
        self.paths.cache_metadata.write_text("{not json", encoding="utf-8")
        self.assertIsNone(recorded_digest(self.paths))

    def test_metadata_without_an_integrity_block_has_no_digest(self) -> None:
        self._write_meta({"chunk_count": 3})
        self.assertIsNone(recorded_digest(self.paths))

    def test_an_empty_digest_is_treated_as_missing(self) -> None:
        self._write_meta({"integrity": {"algo": "hmac-sha256", "digest": ""}})
        self.assertIsNone(recorded_digest(self.paths))


class VerifyCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.paths = _paths(self._tmp.name)

    def test_an_unsealed_or_unkeyed_cache_fails_closed(self) -> None:
        key = signing_key(self.paths)

        self.assertFalse(verify_cache(None, self.paths, "digest"))
        self.assertFalse(verify_cache(key, self.paths, None))
        self.assertFalse(verify_cache(key, self.paths, ""))

    def test_a_matching_digest_passes_and_a_changed_one_does_not(self) -> None:
        key = signing_key(self.paths)
        self.paths.cache_chunks.write_bytes(b"chunks")
        recorded = cache_digest(key, self.paths)

        self.assertTrue(verify_cache(key, self.paths, recorded))

        self.paths.cache_chunks.write_bytes(b"tampered")
        self.assertFalse(verify_cache(key, self.paths, recorded))


if __name__ == "__main__":
    unittest.main()
