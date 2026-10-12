"""HTTP adapters package for the AI Assistant server.

This package contains the FastAPI route modules extracted from the
former monolithic server and agent-module entry points.
"""
from .admin_routes import build_admin_router
from .agent_routes import build_agent_router
from .chat_routes import build_chat_router
from .health_routes import build_health_router

__all__ = [
    "build_admin_router",
    "build_agent_router",
    "build_chat_router",
    "build_health_router",
]
