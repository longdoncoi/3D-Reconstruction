"""A2A protocol adapter — backward-compatible re-exports (ADR 0005).

The implementation has been split into:
- ``a2a_models`` — Pydantic request/response schemas
- ``a2a_payloads`` — payload builders and helpers
- ``a2a_router`` — FastAPI router with all A2A 0.0 endpoints

This module re-exports the public API for backward compatibility.
"""
from __future__ import annotations

from .a2a_models import (
    A2AMessage,
    A2AMessagePart,
    ResumeTaskParams,
    SendMessageConfiguration,
    SendMessageParams,
    SendTaskParams,
)
from .a2a_payloads import (
    APPROVAL_RESUME_EXTENSION,
    build_agent_card,
    task_payload,
)
from .a2a_router import build_a2a_router

__all__ = [
    "APPROVAL_RESUME_EXTENSION",
    "A2AMessage",
    "A2AMessagePart",
    "ResumeTaskParams",
    "SendMessageConfiguration",
    "SendMessageParams",
    "SendTaskParams",
    "build_a2a_router",
    "build_agent_card",
    "task_payload",
]
