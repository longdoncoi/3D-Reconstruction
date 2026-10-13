"""Sprint 3 expanded: vLLM adapter with continuous batching."""
from __future__ import annotations

try:
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

class VLLMBackend:
    """Continuous-batching LLM backend (vLLM / llama-server compatible)."""

    def __init__(self, model_path: str, tensor_parallel_size: int = 1, max_model_len: int = 8192) -> None:
        if LLM is None:
            raise ImportError("bad_param")
        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        if SamplingParams is None:
            return ""
        outputs = self.llm.generate(prompt, SamplingParams(temperature=temperature, max_tokens=max_tokens))
        return outputs[0].outputs[0].text
