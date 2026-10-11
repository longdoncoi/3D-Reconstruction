"""Legacy MCP server shim — re-exports from ai_assistant.adapters.mcp.

The composition root now builds its own :class:`McpToolServer` with the tool
gateway injected at construction time, so this shim only keeps the legacy
import path (``ai_assistant.legacy.mcp_server``) resolvable.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("ai_assistant.legacy.mcp_server")

# Re-export MCP adapter
from ..adapters.mcp import MCP_AVAILABLE, McpToolServer, bind_plugin_tools  # noqa: E402

__all__ = [
    "MCP_AVAILABLE",
    "McpToolServer",
    "bind_plugin_tools",
]
