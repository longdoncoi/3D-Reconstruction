"""Legacy re-export of agent_module for backward compatibility."""
import ai_assistant.agents.service as _impl
from ai_assistant.agents.service import *


def __getattr__(name: str):
    return getattr(_impl, name)

