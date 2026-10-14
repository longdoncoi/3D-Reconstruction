"""Shared identifier helpers for the agent layer."""
from __future__ import annotations

import secrets


def generate_action_id() -> str:
    return secrets.token_urlsafe(16)
__all__ = ["generate_action_id"]
