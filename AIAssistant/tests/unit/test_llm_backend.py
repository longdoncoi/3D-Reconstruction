"""Coverage for the llama.cpp backend adapter.

``llama_cpp`` may or may not be installed: the constructor classes are always
patched with fakes so the load geometries (GPU, CPU fallback, vision) and the
attribute-proxy behaviour are exercised without loading real weights.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai_assistant.config.model_registry import ModelDefinition, ModelRegistry
from ai_assistant.llm import model_loader
from ai_assistant.llm.llama_cpp_backend import LlamaCppBackend
from ai_assistant.llm.model_loader import get_backend, load_model

# Stand-in for the optional ``llama_cpp`` package so ``mock.patch`` targets
# like "llama_cpp.Llama" resolve in the offline CI environment as well.
_FAKE_LLAMA_CPP = SimpleNamespace(
    Llama=object,
    llama_chat_format=SimpleNamespace(Qwen25VLChatHandler=object),
)


class _PatchedLoad:
    """Context manager that neutralises ML memory release during load tests."""

    def __init__(self, instance: MagicMock | None = None) -> None:
        self._instance = instance or MagicMock()

    def __enter__(self):
        self._modules_patcher = patch.dict(
            sys.modules,
            {
                "llama_cpp": _FAKE_LLAMA_CPP,
                "llama_cpp.llama_chat_format": _FAKE_LLAMA_CPP.llama_chat_format,
            },
        )
        release = patch("ai_assistant.llm.llama_cpp_backend.release_ml_memory")
        llama = patch("llama_cpp.Llama", return_value=self._instance)
        self._modules_patcher.start()
        self._release = release.start()
        self._llama = llama.start()
        return self._instance

    def __exit__(self, *exc) -> None:
        self._llama.stop()
        self._release.stop()
        self._modules_patcher.stop()


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

    def test_vision_cpu_fallback_builds_a_missing_handler(self) -> None:
        """The GPU phase may die inside the handler constructor itself; the
        CPU retry then builds the handler it never got."""
        fresh_handler = MagicMock()
        second_llm = MagicMock()
        # The GPU phase dies inside the handler constructor, so ``Llama`` is
        # only ever attempted once — on the CPU retry.
        with _PatchedLoad(), patch(
            "llama_cpp.Llama",
            return_value=second_llm,
        ), patch(
            "llama_cpp.llama_chat_format.Qwen25VLChatHandler",
            side_effect=[RuntimeError("GPU handler failed"), fresh_handler],
        ) as mock_handler:
            backend = LlamaCppBackend(
                model_path="vl.gguf", is_vision=True, mmproj_path="mmproj.gguf"
            )
        self.assertEqual(mock_handler.call_count, 2)
        self.assertIs(backend._chat_handler, fresh_handler)
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



# ── Constructor attribute pins (moved from test_coverage_boost) ──

class LlamaCppBackendTests(unittest.TestCase):
    """Test LlamaCppBackend coverage."""

    def test_backend_initialization(self) -> None:
        """LlamaCppBackend initializes with correct attributes."""
        with patch("ai_assistant.llm.llama_cpp_backend.LlamaCppBackend._load"):
            backend = LlamaCppBackend(
                model_path="test.gguf",
                n_ctx=4096,
            )
            self.assertIsNotNone(backend)
            self.assertEqual(backend.model_path, "test.gguf")

    def test_backend_is_vision_supported_false(self) -> None:
        """LlamaCppBackend reports vision not supported by default."""
        with patch("ai_assistant.llm.llama_cpp_backend.LlamaCppBackend._load"):
            backend = LlamaCppBackend(model_path="test.gguf")
            self.assertFalse(backend.is_vision_supported)

    def test_backend_is_vision_supported_true(self) -> None:
        """LlamaCppBackend reports vision supported when configured."""
        with patch("ai_assistant.llm.llama_cpp_backend.LlamaCppBackend._load"):
            backend = LlamaCppBackend(model_path="test.gguf", is_vision=True)
            self.assertTrue(backend.is_vision_supported)

    def test_backend_model_description(self) -> None:
        """Backend returns model description."""
        with patch("ai_assistant.llm.llama_cpp_backend.LlamaCppBackend._load"):
            backend = LlamaCppBackend(model_path="test.gguf", desc="Test model")
            self.assertIn("Test model", backend.model_description)


# ── Backend singleton lifecycle (moved from test_coverage_boost) ──

class ModelLoaderBackendStateTests(unittest.TestCase):
    """Backend singleton accessors: ``get_backend`` before/after publish."""

    def test_get_backend_returns_none_when_not_loaded(self) -> None:
        """get_backend returns None when no model is loaded."""
        with patch("ai_assistant.llm.model_loader._global_backend", None):
            backend = get_backend()
            self.assertIsNone(backend)

    def test_get_backend_returns_loaded_backend(self) -> None:
        """get_backend returns the loaded backend."""
        mock_backend = MagicMock()
        with patch("ai_assistant.llm.model_loader._global_backend", mock_backend):
            backend = get_backend()
            self.assertIs(backend, mock_backend)

    def test_load_model_publishes_global_backend(self) -> None:
        """load_model publishes the backend as global."""
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_model.is_vision = False
        mock_model.repo_id = "test-repo"
        mock_model.filename = "test.gguf"
        mock_registry.get_by_index.return_value = mock_model

        with patch("ai_assistant.llm.model_loader.LlamaCppBackend") as mock_backend_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a fake model file so download is skipped
                model_file = Path(tmpdir) / "test.gguf"
                model_file.write_text("fake")

                backend = load_model(mock_registry, tmpdir, model_idx=0, enable_vision=False)
                self.assertIsNotNone(backend)



# ── Singleton lifecycle (moved from test_coverage_boost4) ──────

class ModelLoaderLifecycleTests(unittest.TestCase):
    """Coverage for download/load/reload singleton lifecycle."""

    def setUp(self) -> None:
        self._original = model_loader._global_backend

    def tearDown(self) -> None:
        model_loader._global_backend = self._original

    def _registry(self, *, vision: bool = False) -> ModelRegistry:
        primary = ModelDefinition(
            "repo/primary", "primary.gguf", "Primary", is_vision=vision,
            mmproj_repo_id="repo/mmproj" if vision else None,
            mmproj_filename="mmproj.gguf" if vision else None,
        )
        fallback = ModelDefinition("repo/fallback", "fallback.gguf", "Fallback")
        return ModelRegistry(models=(primary,), fallback=fallback)

    def test_download_if_missing_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "m.gguf").write_text("weights", encoding="utf-8")
            path = model_loader.download_if_missing(ModelDefinition("repo", "m.gguf", "desc"), tmpdir)
        self.assertEqual(path, os.path.join(tmpdir, "m.gguf"))

    def test_download_if_missing_triggers_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "huggingface_hub.hf_hub_download"
        ) as mock_download:
            model_loader.download_if_missing(ModelDefinition("repo", "m.gguf", "desc"), tmpdir)
        mock_download.assert_called_once()

    def test_load_model_publishes_backend(self) -> None:
        registry = self._registry()
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            model_loader, "download_if_missing", return_value="primary.gguf"
        ), patch.object(model_loader, "LlamaCppBackend") as backend_cls, patch.object(
            model_loader, "release_ml_memory"
        ):
            backend = model_loader.load_model(registry, tmpdir, model_idx=0, enable_vision=True)

        self.assertIs(model_loader.get_backend(), backend)
        backend_cls.assert_called_once()

    def test_load_model_vision_disabled_uses_fallback(self) -> None:
        registry = self._registry(vision=True)
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            model_loader, "download_if_missing", return_value="fallback.gguf"
        ), patch.object(model_loader, "LlamaCppBackend") as backend_cls, patch.object(
            model_loader, "release_ml_memory"
        ):
            model_loader.load_model(registry, tmpdir, enable_vision=False)

        self.assertFalse(backend_cls.call_args.kwargs["is_vision"])
        self.assertEqual(backend_cls.call_args.kwargs["desc"], "Fallback")

    def test_load_model_replaces_existing_backend(self) -> None:
        model_loader._global_backend = MagicMock()
        registry = self._registry()
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            model_loader, "download_if_missing", return_value="primary.gguf"
        ), patch.object(model_loader, "LlamaCppBackend"), patch.object(
            model_loader, "release_ml_memory"
        ) as mock_release:
            model_loader.load_model(registry, tmpdir)
        mock_release.assert_called()

    def test_load_model_downloads_mmproj_for_vision(self) -> None:
        registry = self._registry(vision=True)
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            model_loader, "download_if_missing", return_value="primary.gguf"
        ), patch.object(model_loader, "LlamaCppBackend") as backend_cls, patch.object(
            model_loader, "release_ml_memory"
        ), patch("huggingface_hub.hf_hub_download") as mock_download:
            model_loader.load_model(registry, tmpdir, enable_vision=True)

        mock_download.assert_called_once()
        self.assertTrue(backend_cls.call_args.kwargs["is_vision"])

    def test_reload_model_without_backend_raises(self) -> None:
        model_loader._global_backend = None
        with self.assertRaises(RuntimeError):
            model_loader.reload_model(self._registry(), "models_dir")

    def test_reload_model_delegates(self) -> None:
        model_loader._global_backend = MagicMock()
        with patch.object(model_loader, "load_model", return_value="new-backend") as mock_load:
            result = model_loader.reload_model(self._registry(), "models_dir")
        self.assertEqual(result, "new-backend")
        mock_load.assert_called_once()

    def test_get_backend_initially_none(self) -> None:
        model_loader._global_backend = None
        self.assertIsNone(model_loader.get_backend())


if __name__ == "__main__":
    unittest.main()
