"""Coverage for the llama.cpp backend adapter.

``llama_cpp`` is installed in this environment, so we patch the constructor
classes to exercise the load geometries (GPU, CPU fallback, vision) and the
attribute-proxy behaviour without loading real weights.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from ai_assistant.llm.llama_cpp_backend import LlamaCppBackend


class _PatchedLoad:
    """Context manager that neutralises ML memory release during load tests."""

    def __init__(self, instance: MagicMock | None = None) -> None:
        self._instance = instance or MagicMock()

    def __enter__(self):
        release = patch("ai_assistant.llm.llama_cpp_backend.release_ml_memory")
        llama = patch("llama_cpp.Llama", return_value=self._instance)
        self._release = release.start()
        self._llama = llama.start()
        return self._instance

    def __exit__(self, *exc) -> None:
        self._llama.stop()
        self._release.stop()


class LlamaCppBackendLoadTests(unittest.TestCase):
    """Exercise the load geometries."""

    def test_non_vision_load(self) -> None:
        instance = MagicMock()
        with _PatchedLoad(instance):
            backend = LlamaCppBackend(model_path="m.gguf", n_ctx=2048)
        self.assertIs(backend._llm, instance)
        self.assertFalse(backend._used_cpu)
        self.assertFalse(backend.is_vision_supported)

    def test_vision_requires_mmproj(self) -> None:
        with _PatchedLoad():
            with self.assertRaises(ValueError):
                LlamaCppBackend(model_path="vl.gguf", is_vision=True, mmproj_path=None)

    def test_vision_load(self) -> None:
        instance = MagicMock()
        with _PatchedLoad(instance), patch(
            "llama_cpp.llama_chat_format.Qwen25VLChatHandler"
        ) as mock_handler:
            backend = LlamaCppBackend(
                model_path="vl.gguf", is_vision=True, mmproj_path="mmproj.gguf"
            )
        mock_handler.assert_called_once_with(clip_model_path="mmproj.gguf")
        self.assertIs(backend._llm, instance)
        self.assertTrue(backend.is_vision_supported)

    def test_cpu_fallback_on_gpu_failure(self) -> None:
        second_llm = MagicMock()
        with _PatchedLoad(), patch(
            "llama_cpp.Llama",
            side_effect=[RuntimeError("CUDA OOM"), second_llm],
        ):
            backend = LlamaCppBackend(model_path="m.gguf", n_ctx=2048)
        self.assertTrue(backend._used_cpu)
        self.assertIn("CPU", backend.model_description)
        self.assertIs(backend._llm, second_llm)

    def test_vision_cpu_fallback(self) -> None:
        handler = MagicMock()
        second_llm = MagicMock()
        with _PatchedLoad(), patch(
            "llama_cpp.Llama",
            side_effect=[RuntimeError("GPU load failed"), second_llm],
        ), patch(
            "llama_cpp.llama_chat_format.Qwen25VLChatHandler",
            return_value=handler,
        ):
            backend = LlamaCppBackend(
                model_path="vl.gguf", is_vision=True, mmproj_path="mmproj.gguf"
            )
        self.assertTrue(backend._used_cpu)
        self.assertIs(backend._chat_handler, handler)
        self.assertIs(backend._llm, second_llm)


class LlamaCppBackendBehaviourTests(unittest.TestCase):
    """Exercise generate, reload, and attribute proxying."""

    def setUp(self) -> None:
        self.instance = MagicMock()
        self._ctx = _PatchedLoad(self.instance)
        self._ctx.__enter__()
        self.addCleanup(self._ctx.__exit__)
        self.backend = LlamaCppBackend(model_path="m.gguf")

    def test_generate_delegates_to_llama(self) -> None:
        self.instance.create_chat_completion.return_value = {"content": "hi"}
        result = self.backend.generate(
            [{"role": "user", "content": "hi"}], temperature=0.5, max_tokens=64, stop=["\n"]
        )
        self.assertEqual(result, {"content": "hi"})
        self.instance.create_chat_completion.assert_called_once()

    def test_generate_without_model_raises(self) -> None:
        self.backend._llm = None
        with self.assertRaises(RuntimeError):
            self.backend.generate([{"role": "user", "content": "hi"}])

    def test_reload_is_noop(self) -> None:
        self.assertIsNone(self.backend.reload())

    def test_attribute_proxy_forwarded(self) -> None:
        self.instance.n_ctx = 4096
        self.assertEqual(self.backend.n_ctx, 4096)

    def test_attribute_proxy_missing_raises(self) -> None:
        self.backend._llm = object()  # plain object has no proxied attributes
        with self.assertRaises(AttributeError):
            self.backend.does_not_exist

    def test_model_description_plain(self) -> None:
        self.assertEqual(self.backend.model_description, "llama.cpp model")


if __name__ == "__main__":
    unittest.main()
