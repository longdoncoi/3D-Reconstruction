"""Unit tests for the sealed RAG cache format (``rag/cache_seal.py``).

The cache on disk holds a second copy of the user's document corpus, so it is
sealed with the at-rest cipher (ADR 0028). These tests pin the format
contract: which metadata marks a cache as sealed, and which metadata does not.
A cache without the marker is a miss — the corpus is derived, so rebuilding
it costs time, not data.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_assistant.rag.cache_seal import (
    SEALED_CACHE_VERSION,
    is_sealed_metadata,
    read_metadata,
    seal_metadata,
)


class SealMetadataTests(unittest.TestCase):
    def test_the_marker_names_the_algorithm_and_the_version(self) -> None:
        marker = seal_metadata()

        self.assertEqual(marker["algo"], "aes-256-gcm")
        self.assertEqual(marker["version"], str(SEALED_CACHE_VERSION))

    def test_a_fresh_marker_is_recognised(self) -> None:
        self.assertTrue(is_sealed_metadata({"sealed": seal_metadata()}))

    def test_metadata_without_a_seal_block_is_not_sealed(self) -> None:
        """A cache written before sealing existed carries no marker."""
        self.assertFalse(is_sealed_metadata({}))
        self.assertFalse(is_sealed_metadata({"chunk_count": 3}))

    def test_a_seal_block_of_another_shape_is_not_sealed(self) -> None:
        self.assertFalse(is_sealed_metadata({"sealed": "aes-256-gcm"}))
        self.assertFalse(is_sealed_metadata({"sealed": ["aes-256-gcm"]}))
        self.assertFalse(
            is_sealed_metadata({"sealed": {"algo": "hmac-sha256", "version": "2"}}),
        )
        self.assertFalse(
            is_sealed_metadata({"sealed": {"algo": "aes-256-gcm", "version": "1"}}),
        )


class ReadMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "metadata.json"

    def test_a_readable_metadata_is_returned(self) -> None:
        self.path.write_text(json.dumps({"chunk_count": 2}), encoding="utf-8")

        self.assertEqual(read_metadata(self.path), {"chunk_count": 2})

    def test_a_missing_file_reads_as_empty(self) -> None:
        self.assertEqual(read_metadata(self.path), {})

    def test_an_unparseable_file_reads_as_empty(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")

        self.assertEqual(read_metadata(self.path), {})

    def test_a_json_document_that_is_not_an_object_reads_as_empty(self) -> None:
        self.path.write_text("[1, 2, 3]", encoding="utf-8")

        self.assertEqual(read_metadata(self.path), {})


if __name__ == "__main__":
    unittest.main()
