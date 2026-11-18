"""llama.cpp backend implementation."""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, cast

from .contracts import LLMBackend
from .hardware import release_ml_memory

logger = logging.getLogger("ai_assistant.llm.llama_cpp")


class LlamaCppBackend(LLMBackend):
    """Adapter for llama-cpp-python, implementing the LLMBackend protocol.
    
    For backward compatibility during migration, it proxies missing attributes
    to the underlying ``Llama`` instance.
    """

    def __init__(
        self,
        model_path: str,
        is_vision: bool = False,
        mmproj_path: str | None = None,
        n_ctx: int = 8192,
        desc: str = "llama.cpp model",
    ) -> None:
        self.model_path = model_path
        self._is_vision = is_vision
        self._desc = desc
        self._llm: Any = None
        self._chat_handler: Any = None
        self._used_cpu = False
        # llama-cpp-python's context is single-threaded: two overlapping
        # generations corrupt each other's state. That serialization belongs
        # to the object owning the context, not to every caller's shared
        # ``llm_lock`` — with it here, ``llm_lock`` stays what its docstring
        # says (the handle swap guard) and concurrent chat requests no longer
        # queue behind one another at the transport layer.
        self._generation_lock = threading.Lock()

        self._load(mmproj_path, n_ctx)

    @property
    def is_vision_supported(self) -> bool:
        return self._is_vision

    @property
    def model_description(self) -> str:
        return self._desc + (" — CPU" if self._used_cpu else "")

    def _load(self, mmproj_path: str | None, n_ctx: int) -> None:
        from llama_cpp import Llama

        if self._is_vision and not mmproj_path:
            msg = "A vision model requires mmproj_path (clip/mmproj weights)."
            raise ValueError(msg)

        release_ml_memory()
        
        raw_layers = os.environ.get("AI_LLM_GPU_LAYERS")
        gpu_layers = 99
        if raw_layers is not None:
            try:
                gpu_layers = int(raw_layers)
            except ValueError:
                gpu_layers = 99

        raw_flash = os.environ.get("AI_LLM_FLASH_ATTN")
        if raw_flash is not None:
            use_flash_attn = raw_flash.strip() in {"1", "true", "True"}
        else:
            use_flash_attn = (gpu_layers > 0)

        raw_threads = os.environ.get("AI_LLM_THREADS")
        cpu_count = os.cpu_count() or 4
        if raw_threads is not None:
            try:
                n_threads = int(raw_threads)
            except ValueError:
                n_threads = max(1, min(8, cpu_count))
        else:
            n_threads = max(1, min(8, cpu_count))
        n_threads_batch = max(n_threads, min(16, cpu_count))

        if gpu_layers == 0:
            self._used_cpu = True
            logger.info("loading model on CPU: %s (threads=%d, threads_batch=%d)",
                        self.model_path, n_threads, n_threads_batch)
        else:
            logger.info("loading model with GPU offload (gpu_layers=%d, threads=%d, flash_attn=%s): %s",
                        gpu_layers, n_threads, use_flash_attn, self.model_path)

        try:
            if self._is_vision:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                # ``mmproj_path`` is guaranteed by the guard above, so the
                # handler is always built with real clip weights here.
                self._chat_handler = Qwen25VLChatHandler(
                    clip_model_path=cast(str, mmproj_path))
                self._llm = Llama(
                    model_path=self.model_path,
                    chat_handler=self._chat_handler,
                    chat_format="qwen2.5-vl",
                    n_gpu_layers=gpu_layers,
                    n_ctx=n_ctx,
                    n_batch=256,
                    n_threads=n_threads,
                    n_threads_batch=n_threads_batch,
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    flash_attn=use_flash_attn,
                    chat_format_kwargs={"enable_thinking": False}
                )
            else:
                self._llm = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=gpu_layers,
                    n_ctx=n_ctx,
                    n_batch=512,
                    n_threads=n_threads,
                    n_threads_batch=n_threads_batch,
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    flash_attn=use_flash_attn,
                    chat_format_kwargs={"enable_thinking": False}
                )
        except Exception as gpu_error:
            logger.warning("GPU model load failed (%s); retrying on CPU", gpu_error)
            release_ml_memory()
            self._used_cpu = True
            if self._is_vision:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                if self._chat_handler is None:
                    # Same guarantee as above: a vision load never reaches
                    # this retry without clip weights, so just build it.
                    self._chat_handler = Qwen25VLChatHandler(
                        clip_model_path=cast(str, mmproj_path))
                self._llm = Llama(
                    model_path=self.model_path,
                    chat_handler=self._chat_handler,
                    chat_format="qwen2.5-vl",
                    n_gpu_layers=0,
                    n_ctx=n_ctx,
                    n_batch=128,
                    n_threads=n_threads,
                    n_threads_batch=n_threads_batch,
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    chat_format_kwargs={"enable_thinking": False}
                )
            else:
                self._llm = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=0,
                    n_ctx=n_ctx,
                    n_batch=128,
                    n_threads=n_threads,
                    n_threads_batch=n_threads_batch,
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    chat_format="qwen3",
                    chat_format_kwargs={"enable_thinking": False}
                )

        self._attach_cache()
        self._configure_thinking()
        logger.info("[Application Output] Model loaded: %s", self.model_description)

    def _configure_thinking(self) -> None:
        """Configure thinking tokens in the chat template formatter."""
        if self._llm is None:
            return
        enable_thinking = os.environ.get("AI_LLM_ENABLE_THINKING", "0").strip() in {"1", "true", "True"}
        try:
            handlers = getattr(self._llm, "_chat_handlers", {})
            for handler in handlers.values():
                if hasattr(handler, "__closure__") and handler.__closure__:
                    for cell in handler.__closure__:
                        formatter = cell.cell_contents
                        if hasattr(formatter, "_environment") and hasattr(formatter._environment, "globals"):
                            formatter._environment.globals["enable_thinking"] = enable_thinking
                            logger.debug("Configured chat template enable_thinking=%s", enable_thinking)
        except Exception as error:
            logger.debug("Could not configure thinking flag on chat template: %s", error)

    def _attach_cache(self) -> None:
        """Attach a RAM cache to the Llama instance if available and enabled."""
        if self._llm is None or os.environ.get("AI_LLM_CACHE_ENABLED", "1") == "0":
            return
        try:
            import llama_cpp
            cache_cls = getattr(llama_cpp, "LlamaRAMCache", None)
            if cache_cls is not None and hasattr(self._llm, "set_cache"):
                raw_mb = os.environ.get("AI_LLM_CACHE_CAPACITY_MB", "512")
                try:
                    mb = max(16, int(raw_mb))
                except ValueError:
                    mb = 512
                self._llm.set_cache(cache_cls(capacity_bytes=mb * 1024 * 1024))
                logger.debug("LlamaRAMCache attached to LLM context (%d MB)", mb)
        except Exception as error:
            logger.debug("Could not attach LlamaRAMCache: %s", error)

    def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
        stream: bool = False,
        stop: list[str] | None = None,
        **kwargs: Any
    ) -> Any:
        if self._llm is None:
            msg = "LLM not initialized"
            raise RuntimeError(msg)
        if stream:
            # A lazy stream keeps generating while the caller iterates, so the
            # lock has to cover the iteration — not just the call that
            # creates the generator.
            return self._locked_stream(messages, temperature, max_tokens, stop, kwargs)
        with self._generation_lock:
            return self._llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                stop=stop,
                **kwargs
            )

    def _locked_stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
        kwargs: dict[str, Any],
    ) -> Any:
        """Yield a streamed completion while holding the backend lock."""
        with self._generation_lock:
            yield from self._llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                stop=stop,
                **kwargs
            )

    def reload(self) -> None:
        """Reload the model (useful if settings changed, not strictly needed for Llama unless re-instantiating)."""
        pass
        
    def __getattr__(self, name: str) -> Any:
        """Proxy missing attributes to the underlying Llama instance for backward compatibility."""
        if self._llm and hasattr(self._llm, name):
            return getattr(self._llm, name)
        msg = f"'{type(self).__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)
