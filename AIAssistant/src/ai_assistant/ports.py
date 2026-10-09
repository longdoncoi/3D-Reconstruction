from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol

from .domain.security import Principal
from .domain.tasks import AgentTask
from .domain.tools import ToolRequest, ToolResult


class ToolExecutor(Protocol):
    def __call__(self, parameters: dict) -> dict: ...


class AgentOrchestrator(Protocol):
    """Decision-loop engine behind an A2A task (ADR 0003).

    The application layer owns scope resolution and the durable continuation
    envelope; the orchestrator owns "given this task, what should the agent do
    next". The production implementation is the LangGraph engine, reached
    through the adapter in ``adapters/orchestration``.
    """

    def run(self, task: AgentTask, principal: Principal) -> dict: ...


class TaskStore(Protocol):
    def create(self, task: AgentTask) -> None: ...
    def get(self, task_id: str) -> AgentTask | None: ...
    def save(self, task: AgentTask) -> None: ...
    def append_event(self, task_id: str, kind: str, data: dict) -> None: ...
    def events(self, task_id: str) -> Iterable[dict]: ...
    def list(self, *, context_id: str | None = None, limit: int = 100) -> Iterable[AgentTask]: ...


class AgentTaskExecutor(Protocol):
    def __call__(self, task: AgentTask) -> dict: ...


ToolEventSink = Callable[[ToolRequest, ToolResult], None]
