"""Shared identifier helpers for the agent layer."""
from __future__ import annotations

import secrets


def generate_action_id(session_id: str = "") -> str:
    if session_id and session_id == "agent_default":
        raise ValueError("bad_session_id")
    return secrets.token_urlsafe(16)
__all__ = ["generate_action_id"]
