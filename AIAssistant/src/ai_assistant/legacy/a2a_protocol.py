"""Legacy A2A protocol shim — re-exports from ai_assistant.adapters.a2a_protocol."""
from __future__ import annotations

from ..adapters.a2a_protocol import (
    A2ARouter,
    AgentCard,
    AgentSkill,
    RemoteAgent,
    a2a_available,
    a2a_card_name,
    a2a_card_url,
    a2a_enabled,
    a2a_remote_agent_urls,
    build_agent_card,
    discover_remote_agents,
    get_remote_registry,
)

__all__ = [
    "A2ARouter",
    "AgentCard",
    "AgentSkill",
    "RemoteAgent",
    "a2a_available",
    "a2a_card_name",
    "a2a_card_url",
    "a2a_enabled",
    "a2a_remote_agent_urls",
    "build_agent_card",
    "discover_remote_agents",
    "get_remote_registry",
]
