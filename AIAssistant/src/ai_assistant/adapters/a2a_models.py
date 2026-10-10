"""A2A protocol request/response models (ADR 0005).

Extracted from ``adapters/a2a.py`` to separate Pydantic schemas from routing logic.
All models comply with A2A 0.3 specification.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..domain.security import DataClassification


class A2AMessagePart(BaseModel):
    kind: str = "text"
    text: str = Field(min_length=1, max_length=32000)


class A2AMessage(BaseModel):
    role: str = "ROLE_USER"
    parts: list[A2AMessagePart] = Field(min_length=1)
    message_id: str | None = Field(default=None, alias="messageId")
    task_id: str | None = Field(default=None, alias="taskId")
    context_id: str | None = Field(default=None, alias="contextId")

    model_config = {"populate_by_name": True}


class SendTaskParams(BaseModel):
    message: A2AMessage
    capability: str = "supervisor"
    context_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    classification: DataClassification = DataClassification.INTERNAL


class SendMessageConfiguration(BaseModel):
    return_immediately: bool = Field(default=True, alias="returnImmediately")

    model_config = {"populate_by_name": True}


class SendMessageParams(BaseModel):
    message: A2AMessage
    configuration: SendMessageConfiguration = Field(default_factory=SendMessageConfiguration)
    metadata: dict[str, Any] = Field(default_factory=dict)
    capability: str = "supervisor"
    classification: DataClassification = DataClassification.INTERNAL


class ResumeTaskParams(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "A2AMessage",
    "A2AMessagePart",
    "ResumeTaskParams",
    "SendMessageConfiguration",
    "SendMessageParams",
    "SendTaskParams",
]
