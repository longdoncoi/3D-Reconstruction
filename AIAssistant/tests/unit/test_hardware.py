"""Unit tests for GPU/memory cleanup helpers.

``torch`` is not part of the offline CI dependency set, so the tests inject a
fake torch module through ``sys.modules`` or force the ImportError path.
"""
from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from ai_assistant.llm import hardware


class ReleaseMlMemoryTest(unittest.TestCase):
    def test_degrades_when_torch_missing(self) -> None:
        with (
            mock.patch.dict(sys.modules, {"torch": None}),
            mock.patch.object(hardware.gc, "collect") as collect,
        ):
            hardware.release_ml_memory()  # must not raise
        # Only the CUDA half degrades to silence: the collector still ran.
        collect.assert_called_once()

    def test_clears_cuda_cache_when_available(self) -> None:
        empty_cache = mock.Mock()
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, empty_cache=empty_cache)
        )
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            hardware.release_ml_memory()
        empty_cache.assert_called_once()

    def test_skips_cache_clear_without_cuda(self) -> None:
        empty_cache = mock.Mock()
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False, empty_cache=empty_cache)
        )
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            hardware.release_ml_memory()
        empty_cache.assert_not_called()


class LlamaCppNullStreamCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._streams = hardware._llama_cpp_null_streams[:]
        hardware._llama_cpp_null_streams.clear()

    def tearDown(self) -> None:
        hardware._llama_cpp_null_streams[:] = self._streams

    def test_closes_the_optional_backend_streams_without_importing_it(self) -> None:
        stdout, stderr = mock.Mock(closed=False), mock.Mock(closed=False)
        utils = SimpleNamespace(outnull_file=stdout, errnull_file=stderr)
        with mock.patch.dict(sys.modules, {"llama_cpp._utils": utils}):
            hardware.close_llama_cpp_null_streams()
        stdout.close.assert_called_once_with()
        stderr.close.assert_called_once_with()

    def test_cleanup_tolerates_an_already_closed_or_failing_stream(self) -> None:
        closed = mock.Mock(closed=True)
        failing = mock.Mock(closed=False)
        failing.close.side_effect = OSError("stream already gone")
        utils = SimpleNamespace(outnull_file=closed, errnull_file=failing)
        with mock.patch.dict(sys.modules, {"llama_cpp._utils": utils}):
            hardware.close_llama_cpp_null_streams()
        closed.close.assert_not_called()
        failing.close.assert_called_once_with()

    def test_registers_the_shutdown_cleanup_once(self) -> None:
        previous = hardware._llama_cpp_cleanup_registered
        hardware._llama_cpp_cleanup_registered = False
        try:
            with mock.patch.object(hardware.atexit, "register") as register:
                hardware.register_llama_cpp_cleanup()
                hardware.register_llama_cpp_cleanup()
            register.assert_called_once_with(hardware.close_llama_cpp_null_streams)
        finally:
            hardware._llama_cpp_cleanup_registered = previous

    def test_registration_remembers_a_reloaded_backend_module(self) -> None:
        stdout, stderr = mock.Mock(closed=False), mock.Mock(closed=False)
        utils = SimpleNamespace(outnull_file=stdout, errnull_file=stderr)
        previous = hardware._llama_cpp_cleanup_registered
        hardware._llama_cpp_cleanup_registered = True
        try:
            with mock.patch.dict(sys.modules, {"llama_cpp._utils": utils}):
                hardware.register_llama_cpp_cleanup()
            self.assertEqual(hardware._llama_cpp_null_streams, [stdout, stderr])
        finally:
            hardware._llama_cpp_cleanup_registered = previous


if __name__ == "__main__":
    unittest.main()
