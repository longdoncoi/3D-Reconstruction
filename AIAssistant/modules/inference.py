"""Inference re-export."""
from ai_assistant.llm.inference import (
    backend_mode,
    cloud_allowed,
    openai_compatible_completion,
    strip_think_tags,
)

__all__ = [
    "backend_mode",
    "cloud_allowed",
    "openai_compatible_completion",
    "strip_think_tags",
]
