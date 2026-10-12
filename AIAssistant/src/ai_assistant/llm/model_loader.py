"""Model registry loading and management."""
from __future__ import annotations

import logging
import os
import threading

from ai_assistant.config import ModelDefinition, ModelRegistry

from .contracts import LLMBackend
from .hardware import release_ml_memory
from .llama_cpp_backend import LlamaCppBackend

logger = logging.getLogger("ai_assistant.llm.loader")

# Module-level state for backward compatibility (singleton backend)
_global_backend: LLMBackend | None = None
_global_lock = threading.RLock()


def download_if_missing(model: ModelDefinition, models_dir: str) -> str:
    """Return a local model path, downloading the selected artifact if needed."""
    from huggingface_hub import hf_hub_download
    
    path = os.path.join(models_dir, model.filename)
    if not os.path.exists(path):
        logger.info("Downloading %s", model.desc)
        # Using a context manager like startup_step should be handled in the application entrypoint,
        # here we just log.
        hf_hub_download(repo_id=model.repo_id, filename=model.filename, local_dir=models_dir)
    return path


def get_backend() -> LLMBackend | None:
    """Get the currently loaded backend instance."""
    return _global_backend


def load_model(
    registry: ModelRegistry,
    models_dir: str,
    model_idx: int = 0,
    enable_vision: bool = True,
    n_ctx: int = 8192
) -> LLMBackend:
    """Load a model from the registry and publish it as the shared runtime."""
    global _global_backend

    selected = registry.get_by_index(model_idx)
    if selected.is_vision and not enable_vision:
        logger.warning("Vision LLM disabled; using text fallback")
        selected = registry.fallback

    with _global_lock:
        if _global_backend is not None:
            del _global_backend
            release_ml_memory()

        model_path = download_if_missing(selected, models_dir)
        mmproj_path = None
        if selected.is_vision and selected.mmproj_filename and selected.mmproj_repo_id:
            from huggingface_hub import hf_hub_download
            mmproj_path = os.path.join(models_dir, selected.mmproj_filename)
            if not os.path.exists(mmproj_path):
                hf_hub_download(repo_id=selected.mmproj_repo_id, filename=selected.mmproj_filename, local_dir=models_dir)

        backend = LlamaCppBackend(
            model_path=model_path,
            is_vision=selected.is_vision,
            mmproj_path=mmproj_path,
            n_ctx=n_ctx,
            desc=selected.desc
        )
        
        _global_backend = backend
        return backend


def reload_model(
    registry: ModelRegistry,
    models_dir: str,
    model_idx: int = 0,
    enable_vision: bool = True,
    n_ctx: int = 8192
) -> LLMBackend:
    """Release the active model before reloading it."""
    with _global_lock:
        if _global_backend is None:
            msg = "LLM has not been initialized"
            raise RuntimeError(msg)
        return load_model(registry, models_dir, model_idx, enable_vision, n_ctx)
