"""Legacy compatibility package for the 3D-Reconstruction AI Assistant.

This package contains backward-compatible shims for the removed legacy
``modules/`` package. New code should import from ``ai_assistant.*`` directly.

Remaining shims (kept only because the desktop surface or the compatibility
adapter still reference them):

- ``config`` → ``ai_assistant.settings`` + ``ai_assistant.config``
- ``llm_module`` → ``ai_assistant.llm``
- ``rag_module`` → ``ai_assistant.rag``
- ``mcp_server`` → ``ai_assistant.adapters.mcp``
- ``action_manifest`` → ``ai_assistant.tools.action_manifest``
- ``agent_module`` → ``ai_assistant.agents.service``
- ``a2a_protocol`` — legacy A2A message wire format (tests only)
- ``checkpointing`` — legacy checkpoint format (tests only)
- ``lsp_client`` — legacy LSP client bridge (tests only)
"""
from __future__ import annotations

__all__: list[str] = []
