"""Inference backend selection and remote completion utilities."""
from __future__ import annotations

import json
import re
from typing import Any
from urllib.request import Request, urlopen

from ai_assistant.domain.governance import scrub_messages
from ai_assistant.settings import (
    agent_hybrid_policy,
    agent_inference_backend,
    agent_inference_model,
    agent_inference_timeout,
    agent_inference_url,
)


def strip_think_tags(text: str) -> str:
    """Remove ``<think>…</think>`` blocks emitted by reasoning-mode models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def backend_mode() -> str:
    """Return configured backend mode (llama_cpp, remote, etc.)."""
    return agent_inference_backend()


def cloud_allowed(data_classification: str) -> bool:
    """Privacy gate for hybrid routing; local is the secure default."""
    policy = agent_hybrid_policy()
    return policy == "cloud_allowed" and data_classification.casefold() in {"public", "non_sensitive"}


def openai_compatible_completion(messages: list[dict[str, Any]], **params: Any) -> dict[str, Any]:
    """Call a separately deployed vLLM/TGI OpenAI-compatible server."""
    endpoint = agent_inference_url()
    if not endpoint:
        raise RuntimeError("AGENT_INFERENCE_URL is required for remote inference")

    clean_messages = scrub_messages(messages)
    timeout = agent_inference_timeout()

    body = json.dumps({
        "model": agent_inference_model(),
        "messages": clean_messages,
        "stream": False,
        **params,
    }).encode()
    request = Request(
        endpoint + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 -- configured endpoint
        return json.loads(response.read())


__all__ = [
    "backend_mode",
    "cloud_allowed",
    "openai_compatible_completion",
    "strip_think_tags",
]
