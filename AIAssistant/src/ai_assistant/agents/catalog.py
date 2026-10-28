"""Lazy default for the agent tool catalogue.

A bare ``AgentService``/``AgentRunContext`` (tests, embeds) must stay usable
without the composition root, so it builds the built-in catalogue itself. The
import is deliberately inside the function: importing ``agents`` must not pull
the tool package (and its built-in handlers) at module load, and the concrete
factory is a composition default rather than a module-level dependency edge.
"""
from __future__ import annotations

from ..ports import AgentToolCatalog


def default_agent_catalog() -> AgentToolCatalog:
    """Build the built-in tool catalogue (the concrete class satisfies the port)."""
    from ..tools.factory import create_tool_registry

    return create_tool_registry()


__all__ = ["default_agent_catalog"]
