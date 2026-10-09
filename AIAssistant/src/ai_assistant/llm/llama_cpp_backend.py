"""llama.cpp backend implementation."""
from __future__ import annotations

import logging
from typing import Any

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
        
        self._load(mmproj_path, n_ctx)

    @property
    def is_vision_supported(self) -> bool:
        return self._is_vision

    @property
    def model_description(self) -> str:
        return self._desc + (" — CPU" if self._used_cpu else "")

    def _load(self, mmproj_path: str | None, n_ctx: int) -> None:
        from llama_cpp import Llama
        
        release_ml_memory()
        logger.info("loading model: %s", self.model_path)
        
        try:
            if self._is_vision:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                self._chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
                self._llm = Llama(
                    model_path=self.model_path,
                    chat_handler=self._chat_handler,
                    chat_format="qwen2.5-vl",
                    n_gpu_layers=99,
                    n_ctx=n_ctx,
                    n_batch=256,
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    chat_format_kwargs={"enable_thinking": False}
                )
            else:
                self._llm = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=99,
                    n_ctx=n_ctx,
                    n_batch=512,
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    chat_format_kwargs={"enable_thinking": False}
                )
        except Exception as gpu_error:
            logger.warning("GPU model load failed (%s); retrying on CPU", gpu_error)
            release_ml_memory()
            self._used_cpu = True
            if self._is_vision:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                if self._chat_handler is None:
                    self._chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
                self._llm = Llama(
                    model_path=self.model_path,
                    chat_handler=self._chat_handler,
                    chat_format="qwen2.5-vl",
                    n_gpu_layers=0,
                    n_ctx=n_ctx,
                    n_batch=128,
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
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    chat_format="qwen3",
                    chat_format_kwargs={"enable_thinking": False}
                )
                
        logger.info("[Application Output] Model loaded: %s", self.model_description)

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
            raise RuntimeError("LLM not initialized")
        return self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
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
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
