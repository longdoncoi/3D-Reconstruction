"""Legacy re-export of agent_module for backward compatibility."""
from __future__ import annotations

import ai_assistant.adapters.legacy_agent as _impl
from ai_assistant.adapters.legacy_agent import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_impl, name)

