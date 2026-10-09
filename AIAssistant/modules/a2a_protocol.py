"""Legacy re-export of a2a_protocol."""
from ai_assistant.adapters.a2a_protocol import (
    A2A_CARD_NAME,
    A2A_CARD_URL,
    A2A_ENABLED,
    A2A_REMOTE_AGENTS,
    A2ARouter,
    AgentCard,
    AgentSkill,
    RemoteAgent,
    a2a_available,
    build_agent_card,
    discover_remote_agents,
    get_remote_registry,
)

__all__ = [
    "A2A_CARD_NAME",
    "A2A_CARD_URL",
    "A2A_ENABLED",
    "A2A_REMOTE_AGENTS",
    "A2ARouter",
    "AgentCard",
    "AgentSkill",
    "RemoteAgent",
    "a2a_available",
    "build_agent_card",
    "discover_remote_agents",
    "get_remote_registry",
]
