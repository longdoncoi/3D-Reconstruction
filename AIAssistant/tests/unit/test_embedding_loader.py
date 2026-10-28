"""Coverage for the deferred encoder and its two failure-aware fallbacks.

Loading a 4 GB embedding model on a cache-hit startup is wasted work, so the
encoder resolves lazily; but an operator has to be able to tell a real model
from the deterministic hashing fallback. Both facts are asserted here: the
factory's three outcomes (import missing, weights missing, loaded) and the
lazy wrapper that hides them behind one ``encode`` call.
"""
from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from unittest import mock

from ai_assistant.config.rag_config import RAGSettings
from ai_assistant.rag.embeddings import (
    HashingEmbeddingEncoder,
    LazyEncoder,
    lazy_embedding_encoder,
    load_embedding_model,
)


class HashingEncoderTests(unittest.TestCase):
    def test_dimension_is_reported_and_clamped(self) -> None:
        self.assertGreaterEqual(HashingEmbeddingEncoder(8).dimension, 8)
        self.assertEqual(HashingEmbeddingEncoder(64).dimension, 64)

    def test_encoding_is_deterministic_and_normalised(self) -> None:
        encoder = HashingEmbeddingEncoder(32)
        first = encoder.encode("hệ thống mô hình", normalize_embeddings=True)
        second = encoder.encode("hệ thống mô hình", normalize_embeddings=True)
        self.assertEqual(first.tolist(), second.tolist())

    def test_an_empty_text_keeps_the_zero_vector(self) -> None:
        """``norm == 0`` must not divide by zero (NaN would poison FAISS)."""
        vector = HashingEmbeddingEncoder(16).encode("", normalize_embeddings=True)
        self.assertEqual(float(abs(vector).sum()), 0.0)

    def test_normalisation_can_be_skipped(self) -> None:
        vector = HashingEmbeddingEncoder(16).encode("hello world")
        self.assertGreater(float(abs(vector).sum()), 0.0)


class LoadEmbeddingModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = RAGSettings.from_env_and_dict({})

    def test_missing_package_falls_back_and_says_so(self) -> None:
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            encoder, is_fallback = load_embedding_model(self.settings)
        self.assertIsInstance(encoder, HashingEmbeddingEncoder)
        self.assertTrue(is_fallback)

    def test_unloadable_weights_fall_back_and_say_so(self) -> None:
        module = types.ModuleType("sentence_transformers")

        class _Boom:
            def __init__(self, name):
                msg = "weights not on disk"
                raise OSError(msg)

        module.SentenceTransformer = _Boom
        with mock.patch.dict(sys.modules, {"sentence_transformers": module}):
            encoder, is_fallback = load_embedding_model(self.settings)
        self.assertIsInstance(encoder, HashingEmbeddingEncoder)
        self.assertTrue(is_fallback)

    def test_a_loadable_model_is_used_verbatim(self) -> None:
        module = types.ModuleType("sentence_transformers")

        class _Model:
            def __init__(self, name):
                self.name = name

        module.SentenceTransformer = _Model
        with mock.patch.dict(sys.modules, {"sentence_transformers": module}):
            encoder, is_fallback = load_embedding_model(self.settings)
        self.assertIsInstance(encoder, _Model)
        self.assertFalse(is_fallback)


class LazyEncoderTests(unittest.TestCase):
    def test_the_factory_runs_once_and_then_is_cached(self) -> None:
        factory = mock.Mock(return_value=(HashingEmbeddingEncoder(16), True))
        lazy = LazyEncoder(factory)

        self.assertFalse(lazy.is_loaded)
        self.assertIsInstance(lazy.encoder, HashingEmbeddingEncoder)
        self.assertTrue(lazy.is_loaded)
        self.assertIsInstance(lazy.encode("hi"), object)
        self.assertTrue(lazy.is_loaded)
        factory.assert_called_once_with()

    def test_lazy_embedding_encoder_is_the_shared_wrapper(self) -> None:
        lazy = lazy_embedding_encoder(RAGSettings.from_env_and_dict({}))
        self.assertIsInstance(lazy, LazyEncoder)
        self.assertFalse(lazy.is_loaded)

    def test_a_cold_load_runs_once_under_concurrent_encode(self) -> None:
        """ADR 0044 fans ``rag_search`` branches out on worker threads.

        The first query against a warm cache is where two branches race on the
        check-then-set in ``LazyEncoder.encoder``: without the lock both call
        the factory and the process pays for two transformer instances (one is
        leaked when the second assignment overwrites it). The barrier starts
        every thread inside the same window and the factory sleeps long enough
        that an unsynchronised implementation is caught, not merely unlucky.
        """
        calls = 0
        calls_lock = threading.Lock()
        barrier = threading.Barrier(8)
        resolved = HashingEmbeddingEncoder(16)

        def factory() -> tuple[HashingEmbeddingEncoder, bool]:
            nonlocal calls
            with calls_lock:
                calls += 1
            # Long enough that every thread reaches the check before the first
            # load completes; the lock must make them wait, not duplicate it.
            time.sleep(0.02)
            return resolved, False

        lazy = LazyEncoder(factory)
        results: list[object] = []
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=5)
                results.append(lazy.encoder)
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive(), "the load must not deadlock")

        self.assertEqual(errors, [])
        self.assertEqual(calls, 1, "the factory must run exactly once")
        self.assertEqual(len(results), 8)
        self.assertTrue(all(result is resolved for result in results))
        self.assertTrue(lazy.is_loaded)


if __name__ == "__main__":
    unittest.main()
