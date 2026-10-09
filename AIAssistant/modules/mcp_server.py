"""Legacy re-export of mcp_server."""
from ai_assistant.adapters.mcp import (
    MCP_AVAILABLE,
    asgi_app,
    lifespan,
    mcp,
)

__all__ = [
    "MCP_AVAILABLE",
    "asgi_app",
    "lifespan",
    "mcp",
]
