"""Compatibility exports for the former ``LangGraphAgent`` module.

The implementation has been migrated to `ai_assistant.orchestration.graph`.
This module provides the old public names without leaving source files at the
project root.
"""
from __future__ import annotations

from ai_assistant.orchestration.graph import (
    LocalAgentGraph,
    _summarize_messages,
)
from ai_assistant.orchestration.state import AgentState

__all__ = [
    "AgentState",
    "LocalAgentGraph",
    "_summarize_messages",
]
