from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from time import time
from typing import Any
from uuid import uuid4

from .security import DataClassification, ExecutionOrigin


class TaskStatus(StrEnum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    REJECTED = "rejected"


_TERMINAL = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED, TaskStatus.REJECTED}


def is_terminal(status: TaskStatus) -> bool:
    """Whether ``status`` is a state a task can never leave.

    ADR 0032: the terminal set used to be private to this module, so the
    application layer re-spelled it literally — and a second, *different*
    vocabulary of terminal strings lived in the coordinator. The set is now
    public, so there is one answer to "can this task still move?".
    """
    return status in _TERMINAL


@dataclass(frozen=True, slots=True)
class AgentTask:
    id: str
    capability: str
    message: str
    context_id: str | None
    metadata: dict[str, Any]
    classification: DataClassification
    status: TaskStatus = TaskStatus.SUBMITTED
    origin: ExecutionOrigin = ExecutionOrigin.LOCAL
    #: Tenant that owns this task. The A2A surface resolves the caller's
    #: tenant from the bearer token and scopes every read/mutation to it, so
    #: one token can never resume or cancel another tenant's task.
    owner: str = "default"
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)
    result: dict[str, Any] | None = None
    error: str | None = None

    @classmethod
    def new(cls, capability: str, message: str, *, context_id: str | None = None,
            metadata: dict[str, Any] | None = None,
            classification: DataClassification = DataClassification.INTERNAL,
            owner: str = "default") -> "AgentTask":
        return cls(
            id=f"task-{uuid4()}", capability=capability, message=message,
            context_id=context_id, metadata=metadata or {}, classification=classification,
            owner=owner,
        )

    def transition(self, status: TaskStatus, *, result: dict[str, Any] | None = None,
                   error: str | None = None) -> "AgentTask":
        if self.status in _TERMINAL:
            msg = f"Task {self.id} is terminal ({self.status})"
            raise ValueError(msg)
        if status == TaskStatus.SUBMITTED:
            msg = "A task cannot transition back to submitted"
            raise ValueError(msg)
        return AgentTask(
            id=self.id, capability=self.capability, message=self.message, context_id=self.context_id,
            metadata=dict(self.metadata), classification=self.classification, status=status, origin=self.origin,
            owner=self.owner,
            created_at=self.created_at, updated_at=time(), result=result, error=error,
        )

    def with_metadata(self, metadata: dict[str, Any]) -> "AgentTask":
        """Return a durable task snapshot with updated execution metadata."""
        return AgentTask(
            id=self.id, capability=self.capability, message=self.message, context_id=self.context_id,
            metadata=dict(metadata), classification=self.classification, status=self.status, origin=self.origin,
            owner=self.owner,
            created_at=self.created_at, updated_at=time(), result=self.result, error=self.error,
        )
