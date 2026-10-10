"""Legacy MCP server shim — re-exports from ai_assistant.adapters.mcp."""
from __future__ import annotations

import logging

logger = logging.getLogger("ai_assistant.legacy.mcp_server")

# Re-export MCP adapter
from ..adapters.mcp import asgi_app, configure_tool_gateway, lifespan  # noqa: E402

MCP_AVAILABLE = True

__all__ = [
    "MCP_AVAILABLE",
    "asgi_app",
    "configure_tool_gateway",
    "lifespan",
]
