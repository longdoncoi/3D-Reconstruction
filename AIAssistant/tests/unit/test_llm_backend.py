"""Coverage for the llama.cpp backend adapter.

``llama_cpp`` may or may not be installed: the constructor classes are always
patched with fakes so the load geometries (GPU, CPU fallback, vision) and the
attribute-proxy behaviour are exercised without loading real weights.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai_assistant.config.model_registry import ModelDefinition, ModelRegistry
from ai_assistant.llm import model_loader
from ai_assistant.llm.llama_cpp_backend import LlamaCppBackend

# Stand-in for the optional ``llama_cpp`` package so ``mock.patch`` targets
# like "llama_cpp.Llama" resolve in the offline CI environment as well.
_FAKE_LLAMA_CPP = SimpleNamespace(
    Llama=object,
    LlamaRAMCache=MagicMock,
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


class LlamaCppBackendCacheAndConfigTests(unittest.TestCase):
    """Exercise RAM cache attachment and GPU/flash-attn env settings."""

    def test_cache_attached_by_default(self) -> None:
        instance = MagicMock()
        with _PatchedLoad(instance):
            backend = LlamaCppBackend(model_path="m.gguf")
        self.assertIsNotNone(backend)
        instance.set_cache.assert_called_once()

    def test_cache_disabled_via_env(self) -> None:
        instance = MagicMock()
        with _PatchedLoad(instance), patch.dict(os.environ, {"AI_LLM_CACHE_ENABLED": "0"}):
            backend = LlamaCppBackend(model_path="m.gguf")
        self.assertIsNotNone(backend)
        instance.set_cache.assert_not_called()

    def test_cache_capacity_custom_and_fallback(self) -> None:
        instance = MagicMock()
        with _PatchedLoad(instance), patch.dict(os.environ, {"AI_LLM_CACHE_CAPACITY_MB": "256"}):
            backend = LlamaCppBackend(model_path="m.gguf")
        self.assertIsNotNone(backend)
        instance.set_cache.assert_called_once()

        instance_invalid = MagicMock()
        with _PatchedLoad(instance_invalid), patch.dict(os.environ, {"AI_LLM_CACHE_CAPACITY_MB": "invalid"}):
            backend_invalid = LlamaCppBackend(model_path="m.gguf")
        self.assertIsNotNone(backend_invalid)
        instance_invalid.set_cache.assert_called_once()

    def test_cache_attach_error_is_contained(self) -> None:
        instance = MagicMock()
        instance.set_cache.side_effect = RuntimeError("cache error")
        with _PatchedLoad(instance):
            backend = LlamaCppBackend(model_path="m.gguf")
        self.assertIs(backend._llm, instance)

    def test_attach_cache_noop_when_llm_is_none(self) -> None:
        with patch("ai_assistant.llm.llama_cpp_backend.LlamaCppBackend._load"):
            backend = LlamaCppBackend(model_path="m.gguf")
            backend._llm = None
            backend._attach_cache()

    def test_attach_cache_noop_when_class_missing(self) -> None:
        instance = MagicMock()
        with _PatchedLoad(instance), patch.object(_FAKE_LLAMA_CPP, "LlamaRAMCache", None):
            backend = LlamaCppBackend(model_path="m.gguf")
        self.assertIsNotNone(backend)
        instance.set_cache.assert_not_called()

    def test_gpu_layers_and_flash_attn_env_parsing(self) -> None:
        instance = MagicMock()
        with _PatchedLoad(instance), patch.dict(
            os.environ, {"AI_LLM_GPU_LAYERS": "33", "AI_LLM_FLASH_ATTN": "1"}
        ):
            backend = LlamaCppBackend(model_path="m.gguf")
        self.assertIs(backend._llm, instance)

        instance2 = MagicMock()
        with _PatchedLoad(instance2), patch.dict(
            os.environ, {"AI_LLM_GPU_LAYERS": "not_an_int"}
        ):
            backend = LlamaCppBackend(model_path="m.gguf")
        self.assertIs(backend._llm, instance2)


# ── Backend factory lifecycle (ADR 0041: no process-wide owner) ──

class ModelLoaderBackendStateTests(unittest.TestCase):
    """The loader builds a backend; the caller owns it."""

    def test_loader_exposes_no_process_wide_backend(self) -> None:
        """A module-global second owner is what made ``release-vram`` a no-op."""
        self.assertFalse(hasattr(model_loader, "_global_backend"))
        self.assertFalse(hasattr(model_loader, "get_backend"))
        self.assertFalse(hasattr(model_loader, "load_model"))

    def test_load_backend_returns_the_constructed_backend(self) -> None:
        # A real ModelDefinition, not a MagicMock: the pin fields are part of
        # the dataclass contract the loader reads (ADR 0023), and an
        # auto-attribute MagicMock would fake a pin that nothing verifies.
        mock_model = ModelDefinition("test-repo", "test.gguf", "test")
        mock_registry = ModelRegistry(models=(mock_model,), fallback=mock_model)

        with patch("ai_assistant.llm.model_loader.LlamaCppBackend") as mock_backend_class:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a fake model file so download is skipped
                model_file = Path(tmpdir) / "test.gguf"
                model_file.write_text("fake")

                backend = model_loader.load_backend(
                    mock_registry, tmpdir, model_idx=0, enable_vision=False)
        self.assertIs(backend, mock_backend_class.return_value)



# ── Factory lifecycle (moved from test_coverage_boost4) ──────

class ModelLoaderLifecycleTests(unittest.TestCase):
    """Coverage for download/load lifecycle; the module keeps no state."""

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

    def test_load_backend_builds_from_the_selected_entry(self) -> None:
        registry = self._registry()
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            model_loader, "download_if_missing", return_value="primary.gguf"
        ), patch.object(model_loader, "LlamaCppBackend") as backend_cls:
            backend = model_loader.load_backend(registry, tmpdir, model_idx=0, enable_vision=True)

        self.assertIs(backend, backend_cls.return_value)
        backend_cls.assert_called_once()

    def test_load_backend_vision_disabled_uses_fallback(self) -> None:
        registry = self._registry(vision=True)
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            model_loader, "download_if_missing", return_value="fallback.gguf"
        ), patch.object(model_loader, "LlamaCppBackend") as backend_cls:
            model_loader.load_backend(registry, tmpdir, enable_vision=False)

        self.assertFalse(backend_cls.call_args.kwargs["is_vision"])
        self.assertEqual(backend_cls.call_args.kwargs["desc"], "Fallback")

    def test_load_backend_downloads_mmproj_for_vision(self) -> None:
        registry = self._registry(vision=True)
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            model_loader, "download_if_missing", return_value="primary.gguf"
        ), patch.object(model_loader, "LlamaCppBackend") as backend_cls, patch(
            "huggingface_hub.hf_hub_download"
        ) as mock_download:
            model_loader.load_backend(registry, tmpdir, enable_vision=True)

        mock_download.assert_called_once()
        self.assertTrue(backend_cls.call_args.kwargs["is_vision"])


class PinnedArtifactTests(unittest.TestCase):
    """ADR 0023: registry pins are enforced, never advisory."""

    @staticmethod
    def _write_download(content: bytes):
        def fake_download(*, repo_id, filename, local_dir, revision=None):
            Path(local_dir, filename).write_bytes(content)
        return fake_download

    def test_pinned_existing_file_is_verified_without_a_download(self) -> None:
        content = b"pinned-weights"
        sha = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "m.gguf").write_bytes(content)
            with patch("huggingface_hub.hf_hub_download") as mock_download:
                path = model_loader.download_if_missing(
                    ModelDefinition("repo", "m.gguf", "desc", sha256=sha), tmpdir)
        mock_download.assert_not_called()
        self.assertEqual(path, os.path.join(tmpdir, "m.gguf"))

    def test_corrupt_pinned_file_is_discarded_and_redownloaded(self) -> None:
        good = b"good-weights"
        sha = hashlib.sha256(good).hexdigest()
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "m.gguf").write_bytes(b"corrupt")
            with patch(
                "huggingface_hub.hf_hub_download",
                side_effect=self._write_download(good),
            ) as mock_download:
                path = model_loader.download_if_missing(
                    ModelDefinition("repo", "m.gguf", "desc", sha256=sha), tmpdir)
            self.assertEqual(Path(path).read_bytes(), good)
        # No revision declared: the loader asks for the repo default, not a guess.
        self.assertIsNone(mock_download.call_args.kwargs["revision"])

    def test_revision_pin_is_forwarded_to_the_download(self) -> None:
        sha = hashlib.sha256(b"x").hexdigest()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "huggingface_hub.hf_hub_download",
                side_effect=self._write_download(b"x"),
            ) as mock_download:
                model_loader.download_if_missing(
                    ModelDefinition("repo", "m.gguf", "desc", revision="abc123", sha256=sha), tmpdir)
        self.assertEqual(mock_download.call_args.kwargs["revision"], "abc123")

    def test_a_persistent_mismatch_raises_instead_of_loading(self) -> None:
        sha = hashlib.sha256(b"expected").hexdigest()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "huggingface_hub.hf_hub_download",
                side_effect=self._write_download(b"wrong"),
            ):
                with self.assertRaises(model_loader.ModelIntegrityError):
                    model_loader.download_if_missing(
                        ModelDefinition("repo", "m.gguf", "desc", sha256=sha), tmpdir)

    def test_an_unreadable_file_counts_as_a_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = str(Path(tmpdir, "m.gguf"))
            Path(target).write_bytes(b"x")
            with patch.object(model_loader, "_sha256_of", side_effect=OSError("locked")):
                self.assertFalse(model_loader._matches_pin(target, "deadbeef"))

    def test_hash_comparison_is_case_insensitive(self) -> None:
        content = b"weights"
        sha = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as tmpdir:
            target = str(Path(tmpdir, "m.gguf"))
            Path(target).write_bytes(content)
            self.assertTrue(model_loader._matches_pin(target, sha.upper()))
            self.assertFalse(model_loader._matches_pin(target, "0" * 64))

    def test_an_unpinned_artifact_is_accepted(self) -> None:
        """An empty pin means "no pin"; it must not compare against ""."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = str(Path(tmpdir, "m.gguf"))
            Path(target).write_bytes(b"weights")
            self.assertTrue(model_loader._matches_pin(target, ""))


class MmprojRevisionTests(unittest.TestCase):
    """The vision projector inherits the model pin only from the same repo."""

    def test_explicit_mmproj_revision_wins(self) -> None:
        model = ModelDefinition("repo/a", "a.gguf", "a", mmproj_repo_id="repo/b",
                                mmproj_revision="proj-rev", revision="model-rev")
        self.assertEqual(model_loader._mmproj_revision(model), "proj-rev")

    def test_same_repo_inherits_the_model_revision(self) -> None:
        model = ModelDefinition("repo/a", "a.gguf", "a", mmproj_repo_id="repo/a",
                                mmproj_revision="", revision="model-rev")
        self.assertEqual(model_loader._mmproj_revision(model), "model-rev")

    def test_foreign_repo_without_a_pin_stays_unpinned(self) -> None:
        model = ModelDefinition("repo/a", "a.gguf", "a", mmproj_repo_id="repo/b",
                                mmproj_revision="", revision="model-rev")
        self.assertEqual(model_loader._mmproj_revision(model), "")


class TrackedModelPinsTests(unittest.TestCase):
    """The tracked registry pins every downloadable artifact (ADR 0023)."""

    def test_every_tracked_entry_pins_revision_and_sha256(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "config"
        from ai_assistant.config.model_registry import load_model_registry

        registry = load_model_registry(config_dir)
        entries = [*registry.models, registry.fallback]
        self.assertTrue(entries)
        for entry in entries:
            with self.subTest(model=entry.filename):
                self.assertEqual(len(entry.revision), 40, "revision must be a commit SHA")
                self.assertEqual(len(entry.sha256), 64, "sha256 must be a digest")
                self.assertEqual(entry.sha256, entry.sha256.lower())
                if entry.is_vision:
                    self.assertEqual(len(entry.mmproj_sha256), 64)
                    self.assertTrue(entry.mmproj_revision)


class BackendGenerationSerializationTests(unittest.TestCase):
    """The backend that owns the single-threaded llama context serializes it.

    ``llm_lock`` only guards the *handle swap* now (review finding on
    ``chat_routes`` holding a business lock across generation): two
    overlapping ``create_chat_completion`` calls on one llama context would
    corrupt each other, so the lock lives here, in the owner.
    """

    def test_two_generations_never_overlap(self) -> None:
        active = 0
        max_active = 0
        guard = threading.Lock()
        entered = threading.Event()
        release = threading.Event()

        def create_chat_completion(**_kwargs):
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            try:
                entered.set()
                release.wait(timeout=30)
            finally:
                with guard:
                    active -= 1
            return {"choices": []}

        instance = MagicMock()
        instance.create_chat_completion.side_effect = create_chat_completion
        with _PatchedLoad(instance):
            backend = LlamaCppBackend(model_path="m.gguf")

        results: list[dict] = []

        def run() -> None:
            results.append(backend.generate([{"role": "user", "content": "hi"}]))

        first = threading.Thread(target=run, name="gen-1")
        first.start()
        self.assertTrue(entered.wait(timeout=30))
        second = threading.Thread(target=run, name="gen-2")
        second.start()
        # The second call must be parked on the backend lock while the first
        # is provably inside the context.
        time.sleep(0.2)
        with guard:
            self.assertEqual(max_active, 1, "generations must queue on the backend lock")
        release.set()
        first.join(timeout=30)
        second.join(timeout=30)
        self.assertEqual(len(results), 2)
        with guard:
            self.assertEqual(max_active, 1)

    def test_a_streamed_completion_holds_the_lock_while_it_is_consumed(self) -> None:
        """A lazy stream keeps generating *after* ``generate`` returns, so the
        lock has to cover the iteration — not just the call that created it."""
        lock_held_when_iterated: list[bool] = []

        def create_chat_completion(**_kwargs) -> object:
            def chunks():
                lock_held_when_iterated.append(backend._generation_lock.locked())
                yield {"choices": [{"delta": {"content": "x"}}]}
                yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

            return chunks()

        instance = MagicMock()
        instance.create_chat_completion.side_effect = create_chat_completion
        with _PatchedLoad(instance):
            backend = LlamaCppBackend(model_path="m.gguf")
        stream = backend.generate([{"role": "user", "content": "hi"}], stream=True)
        chunks = list(stream)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(lock_held_when_iterated, [True])

    def test_a_non_streamed_generation_takes_the_lock(self) -> None:
        instance = MagicMock()
        instance.create_chat_completion.return_value = {"choices": []}
        with _PatchedLoad(instance):
            backend = LlamaCppBackend(model_path="m.gguf")
        held: list[bool] = []

        def record(**_kwargs):
            held.append(backend._generation_lock.locked())
            return {"choices": []}

        instance.create_chat_completion.side_effect = record
        backend.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(held, [True])


if __name__ == "__main__":
    unittest.main()
