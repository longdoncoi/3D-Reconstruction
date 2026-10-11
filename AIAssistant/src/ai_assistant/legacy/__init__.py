"""Legacy compatibility package for the 3D-Reconstruction AI Assistant.

This package contains backward-compatible shims for the removed legacy
``modules/`` package. New code should import from ``ai_assistant.*`` directly.

Remaining shims (kept only because the desktop surface or the compatibility
adapter still reference them):

- ``config`` → ``ai_assistant.settings`` + ``ai_assistant.config``
- ``llm_module`` → ``ai_assistant.llm``
- ``rag_module`` → ``ai_assistant.rag``

Shims ``a2a_protocol``, ``action_manifest``, ``agent_module``, ``checkpointing``,
``lsp_client`` and ``mcp_server`` were removed (ADR 0012): nothing imported them
anymore, and each only re-exported a single canonical module.
"""
from __future__ import annotations

__all__: list[str] = []
