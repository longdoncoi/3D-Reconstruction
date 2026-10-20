"""Unit tests for ``ai_assistant.rag.index``.

The FAISS / numpy / rank_bm25 imports are function-local and hidden behind
fakes, so the caching, hashing, scanning and build orchestration logic can be
covered in the offline CI environment.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_assistant.adapters.persistence.storage_crypto import AesGcmFieldCipher
from ai_assistant.config.paths import AppPaths
from ai_assistant.config.rag_config import RAGSettings
from ai_assistant.rag.cache_seal import (
    BM25_CONTEXT,
    CHUNKS_CONTEXT,
    INDEX_CONTEXT,
    seal_metadata,
)
from ai_assistant.rag.domain import ChunkResult
from ai_assistant.rag.index import (
    build_index_from_scratch,
    get_file_system_hash,
    is_cache_valid,
    load_cache,
    load_or_build_index,
    save_cache,
    scan_documents,
)


def _settings(**overrides) -> RAGSettings:
    base = dict(
        embed_model_name="m",
        embedding_passage_prefix="passage: ",
        embedding_supports_images=False,
        embedding_dimension=3,
        chunk_chars=100,
        cache_version=2,
    )
    base.update(overrides)
    return RAGSettings(**base)


def _paths(tmp: str, with_docs: bool = True) -> AppPaths:
    base = Path(tmp)
    project = base / "project"
    project.mkdir(exist_ok=True)
    docs = base / "docs"
    if with_docs:
        docs.mkdir(exist_ok=True)
    for d in ("Models", "Cache"):
        (base / d).mkdir(exist_ok=True)
    cache = base / "Cache"
    return AppPaths(
        project_root=project,
        base_dir=base,
        data_dir=base,
        models_dir=base / "Models",
        cache_dir=cache,
        logs_dir=base / "logs",
        docs_dirs=(docs,),
    )


def _chunk(text: str, source: str) -> ChunkResult:
    return ChunkResult(text=text, source_path=source, loader_type="md")


def _cipher() -> AesGcmFieldCipher:
    return AesGcmFieldCipher(b"c" * 32)


def _write_sealed_cache(paths: AppPaths, chunks, bm25, *,
                        cipher: AesGcmFieldCipher | None = None) -> None:
    """Write the artifacts the way ``save_cache`` does, already sealed.

    The load guards below are about dimension/pickle failures, so they must
    get past the seal first instead of tripping on it.
    """
    key = cipher or _cipher()
    paths.cache_index.write_bytes(key.seal_bytes(b"index", context=INDEX_CONTEXT))
    paths.cache_chunks.write_bytes(
        key.seal_bytes(pickle.dumps(chunks), context=CHUNKS_CONTEXT))
    paths.cache_bm25.write_bytes(
        key.seal_bytes(pickle.dumps(bm25), context=BM25_CONTEXT))
    paths.cache_metadata.write_text(
        json.dumps({"sealed": seal_metadata(), "embedding_dimension": 3}),
        encoding="utf-8",
    )


class _FakeArr:
    def __init__(self, data):
        rows = list(data) if not isinstance(data, list) else data
        self.data = rows
        self.shape = self._shape(rows)

    @staticmethod
    def _shape(rows):
        if not rows:
            return (0,)
        if isinstance(rows[0], (list, tuple)):
            return (len(rows), len(rows[0]))
        return (len(rows),)

    def copy(self):
        return _FakeArr([list(row) for row in self.data])

    def __len__(self):
        return self.shape[0]


class _FakeNp:
    def array(self, data, dtype=None):
        return _FakeArr(data)

    def frombuffer(self, buffer, dtype=None):
        # The cache path rebuilds the FAISS index from sealed bytes; the
        # double only has to hand the fake faiss something index-shaped.
        return _FakeArr(list(buffer))

    # faiss API names, mirrored so the doubles stay faithful.
    def normalize_L2(self, arr):  # noqa: N802
        return None


class _FakeBytes:
    """Stand-in for the ``uint8`` array ``faiss.serialize_index`` returns."""

    def __init__(self, data: bytes):
        self.data = data

    def tobytes(self) -> bytes:
        return self.data

    def __bytes__(self) -> bytes:
        return self.data


class _FakeFaissIndex:
    def __init__(self, dim):
        self.d = dim
        self.ntotal = 0

    def add(self, emb):
        self.ntotal = emb.shape[0]

    def train(self, emb):
        self.ntotal = emb.shape[0]


class _FakeFaiss:
    METRIC_INNER_PRODUCT = 0

    def __init__(self, index_d=3):
        self.index_d = index_d
        self.written: list[str] = []
        self.serialized: list[object] = []

    def write_index(self, index, path):
        self.written.append(str(path))
        Path(path).write_bytes(b"index")

    def read_index(self, path):
        return _FakeFaissIndex(self.index_d)

    def serialize_index(self, index):
        self.serialized.append(index)
        return _FakeBytes(b"serialized-index")

    def deserialize_index(self, array):
        return _FakeFaissIndex(self.index_d)

    def IndexFlatIP(self, dim):  # noqa: N802
        return _FakeFaissIndex(dim)

    def IndexIVFFlat(self, quantizer, dim, nlist, metric):  # noqa: N802
        return _FakeFaissIndex(dim)

    def normalize_L2(self, arr):  # noqa: N802
        return None


class _FakeBm25:
    class BM25Okapi:
        def __init__(self, tokenized):
            self.tokenized = tokenized


class _FakeEmbed:
    def __init__(self, dim=3):
        self.dim = dim
        self.calls: list[list] = []

    def encode(self, data, **kwargs):
        self.calls.append((list(data), kwargs))
        count = len(data) if isinstance(data, (list, tuple)) else 1
        return [[float(i) for i in range(self.dim)] for _ in range(count)]


def _patch_heavy(faiss=None, numpy=None, bm25=None):
    """Install the heavy-dependency doubles in ``sys.modules``.

    A single ``patch.dict`` covers every requested module: stacking one
    ``patch.dict(sys.modules, ...)`` per dependency made the restore order
    significant (stopping in start order re-instates the fake that the last
    patch captured as "original"), which leaked ``numpy``/``faiss`` fakes into
    later tests).
    """
    modules: dict[str, object] = {}
    if faiss is not None:
        modules["faiss"] = faiss
    if numpy is not None:
        modules["numpy"] = numpy
    if bm25 is not None:
        modules["rank_bm25"] = bm25
    patcher = mock.patch.dict(sys.modules, modules)
    patcher.start()
    return [patcher]


class FileSystemHashTests(unittest.TestCase):
    def test_hash_is_stable_and_skips_non_scannable(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            (paths.docs_dirs[0] / "doc.md").write_text("doc", encoding="utf-8")
            (paths.project_root / "a.cpp").write_text("cpp", encoding="utf-8")
            (paths.project_root / "b.py").write_text("py", encoding="utf-8")
            (paths.project_root / "junk.txt").write_text("txt", encoding="utf-8")
            build = paths.project_root / "build"
            build.mkdir()
            (build / "x.cpp").write_text("skip", encoding="utf-8")
            settings = _settings()
            first = get_file_system_hash(paths, settings)
            second = get_file_system_hash(paths, settings)
        self.assertEqual(first, second)

    def test_settings_change_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            base = get_file_system_hash(paths, _settings())
            changed = get_file_system_hash(paths, _settings(embed_model_name="other"))
            different = get_file_system_hash(paths, _settings(cache_version=9))
        self.assertNotEqual(base, changed)
        self.assertNotEqual(base, different)

    def test_missing_docs_dir_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp, with_docs=False)
            first = get_file_system_hash(paths, _settings())
            second = get_file_system_hash(paths, _settings())
            # The absent docs tree is skipped rather than fatal: the project
            # tree still hashes, and the digest is stable across calls.
            self.assertRegex(first, r"^[0-9a-f]{64}$")
            self.assertEqual(first, second)

    def test_hash_survives_stat_errors(self):
        """OSError during stat of any scanned file is tolerated (both trees)."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            (paths.docs_dirs[0] / "doc.md").write_text("doc", encoding="utf-8")
            (paths.docs_dirs[0] / "doc.txt").write_text("doc", encoding="utf-8")
            (paths.project_root / "a.cpp").write_text("cpp", encoding="utf-8")
            (paths.project_root / "boom.cpp").write_text("cpp", encoding="utf-8")

            real_stat = os.stat

            def raising_stat(path, *args, **kwargs):
                name = os.fspath(path).replace("\\", "/")
                if name.endswith("boom.cpp") or name.endswith("doc.txt"):
                    raise OSError("boom")
                return real_stat(path, *args, **kwargs)

            readable = get_file_system_hash(paths, _settings())
            with mock.patch("ai_assistant.rag.index.os.stat", side_effect=raising_stat):
                failing = get_file_system_hash(paths, _settings())
                again = get_file_system_hash(paths, _settings())
            # The files whose stat raised are dropped from the entry list, so
            # the digest moves away from the fully-readable one — and it stays
            # deterministic while the failure persists.
            self.assertNotEqual(failing, readable)
            self.assertEqual(failing, again)


class CacheValidityTests(unittest.TestCase):
    def _write_all(self, paths: AppPaths, settings: RAGSettings) -> None:
        paths.cache_index.write_bytes(b"i")
        paths.cache_chunks.write_bytes(b"c")
        paths.cache_bm25.write_bytes(b"b")
        paths.cache_metadata.write_text(json.dumps({
            "fs_hash": get_file_system_hash(paths, settings),
            "embedding_dimension": settings.embedding_dimension,
            "sealed": seal_metadata(),
        }), encoding="utf-8")

    def test_missing_files_are_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            self.assertFalse(is_cache_valid(paths, _settings()))

    def test_a_hash_error_is_a_cache_miss_not_a_crash(self):
        """A filesystem error while hashing the tree must not break the caller."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            settings = _settings()
            self._write_all(paths, settings)
            with mock.patch("ai_assistant.rag.index.get_file_system_hash",
                            side_effect=OSError("disk gone")):
                with self.assertLogs("ai_assistant.rag.index", level="WARNING") as caught:
                    self.assertFalse(is_cache_valid(paths, settings))
        self.assertIn("Cache read error", caught.output[0])

    def test_valid_when_metadata_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            settings = _settings()
            self._write_all(paths, settings)
            self.assertTrue(is_cache_valid(paths, settings))

    def test_stale_hash_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            settings = _settings()
            self._write_all(paths, settings)
            other = _settings(embedding_dimension=5)
            self.assertFalse(is_cache_valid(paths, other))

    def test_dimension_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            settings = _settings()
            self._write_all(paths, settings)
            self.assertFalse(is_cache_valid(paths, _settings(cache_version=3)))

    def test_corrupt_metadata_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            paths.cache_index.write_bytes(b"i")
            paths.cache_chunks.write_bytes(b"c")
            paths.cache_bm25.write_bytes(b"b")
            paths.cache_metadata.write_text("{not json", encoding="utf-8")
            self.assertFalse(is_cache_valid(paths, _settings()))


class SaveLoadCacheTests(unittest.TestCase):
    def setUp(self):
        self.faiss = _FakeFaiss()
        self._patchers = _patch_heavy(faiss=self.faiss, numpy=_FakeNp())
        self.addCleanup(lambda: [p.stop() for p in self._patchers])
        self.cipher = _cipher()

    def test_save_cache_writes_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            settings = _settings()
            chunks = [_chunk("a", "a.md")]
            save_cache("index", chunks, "bm25", paths, settings, cipher=self.cipher)
            meta = json.loads(paths.cache_metadata.read_text(encoding="utf-8"))
        self.assertEqual(meta["chunk_count"], 1)
        self.assertEqual(meta["embedding_dimension"], 3)
        self.assertEqual(meta["sealed"], seal_metadata())
        # The index is serialized in memory, never written in the clear.
        self.assertEqual(self.faiss.serialized, ["index"])
        self.assertEqual(self.faiss.written, [])

    def test_the_artifacts_on_disk_are_sealed_not_plaintext_pickles(self):
        """The chunk corpus holds the user's document text — it must not be readable."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            chunks = [_chunk("confidential prompt", "secret.md")]
            save_cache("index", chunks, "bm25", paths, _settings(), cipher=self.cipher)

            on_disk = paths.cache_chunks.read_bytes()
            metadata = paths.cache_metadata.read_text(encoding="utf-8")
        self.assertTrue(on_disk.startswith(b"enc:v1:"))
        self.assertNotIn(b"confidential prompt", on_disk)
        self.assertNotIn("confidential prompt", metadata)

    def test_load_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            chunks = [_chunk("a", "a.md")]
            _write_sealed_cache(paths, chunks, "bm25")
            self.faiss.index_d = 3
            index, loaded, bm25 = load_cache(paths, _settings(), cipher=self.cipher)
        self.assertEqual(index.d, 3)
        self.assertEqual(loaded, chunks)
        self.assertEqual(bm25, "bm25")

    def test_load_cache_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            _write_sealed_cache(paths, [], "bm25")
            self.faiss.index_d = 99
            index, loaded, bm25 = load_cache(paths, _settings(), cipher=self.cipher)
        self.assertIsNone(index)
        self.assertEqual(loaded, [])
        self.assertIsNone(bm25)

    def test_load_cache_corrupt_pickle_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            _write_sealed_cache(paths, [], "bm25")
            # The seal is valid, the payload behind it is not a pickle.
            paths.cache_chunks.write_bytes(
                self.cipher.seal_bytes(b"not pickle", context=CHUNKS_CONTEXT))
            index, loaded, _bm25 = load_cache(paths, _settings(), cipher=self.cipher)
        self.assertIsNone(index)
        self.assertEqual(loaded, [])

    def test_a_tampered_chunk_file_is_refused_before_unpickling(self) -> None:
        """One flipped byte turns the cache into a miss, not into code."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            _write_sealed_cache(paths, [_chunk("a", "a.md")], "bm25")
            self.faiss.index_d = 3
            # The envelope stays structurally valid — only the authentication
            # tag moved.
            #
            # The tamper lands in the *middle* of the base64 text on purpose.
            # The last character of a base64 quantum contributes as few as two
            # bits, so replacing it can decode to the very same bytes — the
            # seal would then still verify and this test would pass without
            # having tampered anything (one run in four, depending on the
            # random nonce). A middle character is part of a full quantum, so
            # the decoded ciphertext always changes.
            raw = paths.cache_chunks.read_bytes()
            middle = len(raw) // 2
            replacement = b"Y" if raw[middle:middle + 1] != b"Y" else b"Z"
            paths.cache_chunks.write_bytes(
                raw[:middle] + replacement + raw[middle + 1:]
            )

            index, loaded, bm25 = load_cache(paths, _settings(), cipher=self.cipher)
        self.assertIsNone(index)
        self.assertEqual(loaded, [])
        self.assertIsNone(bm25)

    def test_a_legacy_plaintext_cache_is_refused_and_rebuilt(self) -> None:
        """A cache written before sealing existed is a miss, not a value to load."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            with paths.cache_chunks.open("wb") as handle:
                pickle.dump([_chunk("a", "a.md")], handle)
            with paths.cache_bm25.open("wb") as handle:
                pickle.dump("bm25", handle)
            paths.cache_metadata.write_text(
                json.dumps({"chunk_count": 1}), encoding="utf-8")

            with self.assertLogs("ai_assistant.rag.index", level="WARNING") as caught:
                index, loaded, _bm25 = load_cache(
                    paths, _settings(), cipher=self.cipher)
        self.assertIsNone(index)
        self.assertEqual(loaded, [])
        # The operator is told *why* the cache was dropped: an older, unsealed
        # cache is a different situation from a corrupt one.
        self.assertIn("not sealed", caught.output[0])

    def test_a_cache_written_by_another_key_is_refused(self) -> None:
        """Same files, different machine key (e.g. a copied data dir)."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            _write_sealed_cache(paths, [_chunk("a", "a.md")], "bm25")
            other = AesGcmFieldCipher(b"y" * 32)
            self.faiss.index_d = 3

            index, loaded, _bm25 = load_cache(paths, _settings(), cipher=other)
        self.assertIsNone(index)
        self.assertEqual(loaded, [])

    def test_save_cache_seals_what_it_wrote(self) -> None:
        """What save writes is what load reads back."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            save_cache("index", [_chunk("a", "a.md")], "bm25", paths, _settings(),
                       cipher=self.cipher)
            meta = json.loads(paths.cache_metadata.read_text(encoding="utf-8"))

            self.assertEqual(meta["sealed"]["algo"], "aes-256-gcm")
            self.assertEqual(meta["sealed"]["version"], "2")

            self.faiss.index_d = meta["embedding_dimension"]
            index, loaded, bm25 = load_cache(paths, _settings(), cipher=self.cipher)
        self.assertIsNotNone(index)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(bm25, "bm25")

    def test_the_seal_marker_is_what_makes_a_cache_valid(self) -> None:
        """A cache without the marker is stale even when its hash matches."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            settings = _settings()
            _write_sealed_cache(paths, [], "bm25")
            fs_hash = get_file_system_hash(paths, settings)
            paths.cache_metadata.write_text(
                json.dumps({"fs_hash": fs_hash, "embedding_dimension": 3}),
                encoding="utf-8",
            )
            self.assertFalse(is_cache_valid(paths, settings))

            paths.cache_metadata.write_text(
                json.dumps({"fs_hash": fs_hash, "embedding_dimension": 3,
                            "sealed": seal_metadata()}),
                encoding="utf-8",
            )
            self.assertTrue(is_cache_valid(paths, settings))


class BuildIndexTests(unittest.TestCase):
    def setUp(self):
        self.faiss = _FakeFaiss()
        self.bm25 = _FakeBm25()
        self._patchers = _patch_heavy(
            faiss=self.faiss, numpy=_FakeNp(), bm25=self.bm25,
        )
        self.addCleanup(lambda: [p.stop() for p in self._patchers])

    def test_builds_flat_index_for_small_corpus(self):
        chunks = [_chunk(f"text {i}", f"f{i}.md") for i in range(3)]
        index, loaded, bm25 = build_index_from_scratch(chunks, _FakeEmbed(dim=3), _settings())
        self.assertEqual(index.ntotal, 3)
        self.assertEqual(loaded, chunks)
        self.assertIsInstance(bm25, _FakeBm25.BM25Okapi)

    def test_dimension_mismatch_raises(self):
        chunks = [_chunk("x", "a.md")]
        with self.assertRaises(ValueError):
            build_index_from_scratch(chunks, _FakeEmbed(dim=7), _settings())

    def test_builds_ivf_index_for_large_corpus(self):
        chunks = [_chunk(f"text chunk number {i} with body", f"f{i}.md") for i in range(1000)]
        index, loaded, bm25 = build_index_from_scratch(chunks, _FakeEmbed(dim=3), _settings())
        self.assertEqual(index.ntotal, 1000)
        self.assertEqual(index.nprobe, 16)
        self.assertEqual(loaded, chunks)
        self.assertIsInstance(bm25, _FakeBm25.BM25Okapi)


class _FakeImg:
    """Stand-in for a loaded image object that must be closed after encoding."""

    def __init__(self, name: str = "img") -> None:
        self.name = name
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ImageFailEmbed:
    """Embed model whose ``encode`` refuses non-string (image) inputs."""

    def __init__(self, dim: int = 3) -> None:
        self.dim = dim

    def encode(self, data, **kwargs):
        tagged = list(data)
        if any(not isinstance(item, str) for item in tagged):
            msg = "image encode failed"
            raise RuntimeError(msg)
        return [[float(i) for i in range(self.dim)] for _ in tagged]


class ImageIndexTests(unittest.TestCase):
    """Coverage of the image-aware embedding paths in build_index_from_scratch."""

    def setUp(self):
        self.faiss = _FakeFaiss()
        self.bm25 = _FakeBm25()
        self._patchers = _patch_heavy(
            faiss=self.faiss, numpy=_FakeNp(), bm25=self.bm25,
        )
        self.addCleanup(lambda: [p.stop() for p in self._patchers])
        self.images = [
            ChunkResult(text=f"image {i}", source_path=f"i{i}.png",
                        loader_type="image", is_image=True)
            for i in range(2)
        ]

    def test_image_chunks_encode_with_batched_close(self):
        chunks = [_chunk("txt", "t.md"), *self.images]
        with mock.patch("ai_assistant.rag.index.load_image_for_embedding",
                        return_value=_FakeImg()) as load, \
             mock.patch("ai_assistant.rag.index.release_ml_memory") as release:
            index, _, _ = build_index_from_scratch(
                chunks, _FakeEmbed(dim=3), _settings(embedding_supports_images=True))
        self.assertEqual(load.call_count, 2)
        self.assertEqual(index.ntotal, 3)
        self.assertEqual(release.call_count, 1)

    def test_image_batch_failure_retries_one_by_one(self):
        chunk = _chunk("txt", "t.md")

        class _BatchOnlyFailEmbed:
            dim = 3

            def __init__(self):
                self.image_encodes = 0

            def encode(self, data, **kwargs):
                data = list(data)
                if any(not isinstance(d, str) for d in data):
                    if len(data) > 1:
                        msg = "batch failed"
                        raise RuntimeError(msg)
                    self.image_encodes += 1
                return [[0.0, 1.0, 2.0]]

        embed = _BatchOnlyFailEmbed()
        with mock.patch("ai_assistant.rag.index.load_image_for_embedding",
                        return_value=_FakeImg()), \
             mock.patch("ai_assistant.rag.index.release_ml_memory"):
            index, _, _ = build_index_from_scratch(
                [chunk, *self.images], embed, _settings(embedding_supports_images=True))
        self.assertEqual(index.ntotal, 3)
        self.assertEqual(embed.image_encodes, 2)

    def test_all_image_loads_failing_falls_back_to_text(self):
        chunks = [_chunk("txt", "t.md"), *self.images]
        with mock.patch("ai_assistant.rag.index.load_image_for_embedding",
                        side_effect=ValueError("no image file")), \
             mock.patch("ai_assistant.rag.index.release_ml_memory"):
            index, _, _ = build_index_from_scratch(
                chunks, _FakeEmbed(dim=3), _settings(embedding_supports_images=True))
        self.assertEqual(index.ntotal, 3)

    def test_image_encode_failure_skips_then_falls_back_to_text(self):
        chunks = [_chunk("txt", "t.md"), *self.images]
        with mock.patch("ai_assistant.rag.index.load_image_for_embedding",
                        return_value=_FakeImg()), \
             mock.patch("ai_assistant.rag.index.release_ml_memory"):
            index, _, _ = build_index_from_scratch(
                chunks, _ImageFailEmbed(dim=3), _settings(embedding_supports_images=True))
        self.assertEqual(index.ntotal, 3)

    def test_image_close_failure_is_tolerated(self):
        exploding = _FakeImg("boom")
        exploding.close = mock.Mock(side_effect=RuntimeError("close failed"))
        with mock.patch("ai_assistant.rag.index.load_image_for_embedding",
                        return_value=exploding), \
             mock.patch("ai_assistant.rag.index.release_ml_memory"):
            index, _, _ = build_index_from_scratch(
                self.images, _FakeEmbed(dim=3), _settings(embedding_supports_images=True))
        self.assertEqual(index.ntotal, 2)


class LoadOrBuildTests(unittest.TestCase):
    def test_cache_hit_returns_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            chunk = _chunk("a", "a.md")
            with mock.patch("ai_assistant.rag.index.is_cache_valid") as valid, \
                 mock.patch("ai_assistant.rag.index.load_cache") as load, \
                 mock.patch("ai_assistant.rag.index.save_cache") as save:
                valid.return_value = True
                load.return_value = ("idx", [chunk], "bm25")
                result = load_or_build_index([chunk], object(), paths, _settings(),
                                             cipher=_cipher())
        self.assertEqual(result, ("idx", [chunk], "bm25"))
        save.assert_not_called()

    def test_force_rebuild_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            chunk = _chunk("a", "a.md")
            with mock.patch("ai_assistant.rag.index.is_cache_valid") as valid, \
                 mock.patch("ai_assistant.rag.index.build_index_from_scratch") as build, \
                 mock.patch("ai_assistant.rag.index.save_cache") as save:
                valid.return_value = True
                build.return_value = ("idx", [chunk], "bm25")
                result = load_or_build_index([chunk], object(), paths, _settings(),
                                             force_rebuild=True, cipher=_cipher())
        self.assertEqual(result, ("idx", [chunk], "bm25"))
        save.assert_called_once()

    def test_cache_hit_but_load_failure_rebuilds(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            chunk = _chunk("a", "a.md")
            with mock.patch("ai_assistant.rag.index.is_cache_valid") as valid, \
                 mock.patch("ai_assistant.rag.index.load_cache") as load, \
                 mock.patch("ai_assistant.rag.index.build_index_from_scratch") as build, \
                 mock.patch("ai_assistant.rag.index.save_cache"):
                valid.return_value = True
                load.return_value = (None, [], None)
                build.return_value = ("idx2", [chunk], "bm25")
                result = load_or_build_index([chunk], object(), paths, _settings(),
                                             cipher=_cipher())
        self.assertEqual(result, ("idx2", [chunk], "bm25"))


class ScanDocumentsTests(unittest.TestCase):
    def test_scans_docs_and_project_trees(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            (paths.docs_dirs[0] / "guide.txt").write_text("guide", encoding="utf-8")
            (paths.project_root / "a.md").write_text("a", encoding="utf-8")
            (paths.project_root / "m.py").write_text("b", encoding="utf-8")
            (paths.project_root / "junk.txt").write_text("c", encoding="utf-8")

            def fake_load(fp, project_dir):
                fp = str(fp)
                if fp.endswith((".md", ".py", ".txt")):
                    return [_chunk("text", fp)]
                return []

            registry = SimpleNamespace(load_file=fake_load)
            chunks = scan_documents(paths, registry)
        self.assertEqual(len(chunks), 3)

    def test_scan_skips_unsupported_ext_tilde_and_user_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            (paths.docs_dirs[0] / "guide.txt").write_text("guide", encoding="utf-8")
            (paths.docs_dirs[0] / "notes.bin").write_text("n", encoding="utf-8")
            (paths.project_root / "a.md").write_text("a", encoding="utf-8")
            (paths.project_root / "junk.bin").write_text("c", encoding="utf-8")
            (paths.project_root / "~scratch.py").write_text("d", encoding="utf-8")
            (paths.project_root / "layout.user").write_text("e", encoding="utf-8")

            def fake_load(fp, project_dir):
                fp = str(fp)
                if fp.endswith((".md", ".py", ".txt")):
                    return [_chunk("text", fp)]
                return []

            chunks = scan_documents(paths, SimpleNamespace(load_file=fake_load))
        self.assertEqual(len(chunks), 2)  # guide.txt + a.md; .bin, ~ and .user are skipped

    def test_scan_counts_unloadable_docs_as_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(tmp)
            (paths.docs_dirs[0] / "broken.txt").write_text("b", encoding="utf-8")

            def fake_load(fp, project_dir):
                return []  # loader failure → treated as an error, not a chunk

            chunks = scan_documents(paths, SimpleNamespace(load_file=fake_load))
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
