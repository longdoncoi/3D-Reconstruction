"""Canonical LLM runtime defaults shared by the loader and the agent loop.

These values used to live in ``ai_assistant.legacy.config``. Keeping the single
source of truth here lets framework-free layers import them without touching the
legacy compatibility package (ADR 0001).
"""
from __future__ import annotations

# Average characters per token used for context-budget estimation.
CHARS_PER_TOKEN = 2.2
# Context window used by the local llama.cpp backend.
LLM_N_CTX = 8192

__all__ = ["CHARS_PER_TOKEN", "LLM_N_CTX"]
