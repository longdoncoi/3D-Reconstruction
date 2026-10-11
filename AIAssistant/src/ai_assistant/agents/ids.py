"""Shared identifier helpers for the agent layer.

``generate_action_id`` used to be duplicated in ``agents/service.py`` and
``agents/ui_action_service.py`` (and re-implemented inline in
``agents/runner.py``). One definition keeps the id shape (12 hex chars) stable
across the request, approval and desktop-acknowledgement paths.
"""
from __future__ import annotations

import hashlib
import threading
import time


def generate_action_id() -> str:
    """Return a short, practically unique id for one pending action."""
    seed = f"{time.time()}:{threading.get_ident()}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()[:12]


__all__ = ["generate_action_id"]
