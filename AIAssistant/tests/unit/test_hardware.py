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
        with mock.patch.dict(sys.modules, {"torch": None}):
            hardware.release_ml_memory()  # must not raise

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


if __name__ == "__main__":
    unittest.main()
