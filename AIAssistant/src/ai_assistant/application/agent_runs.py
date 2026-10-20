"""Durable A2A agent lifecycle over a pluggable orchestration engine."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..domain.security import Principal, scopes_for_capability
from ..domain.steps import Step
from ..domain.tasks import AgentTask
from ..ports import AgentOrchestrator


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    status: str
    content: str
    #: Typed since the vocabulary moved inward (ADR 0019): this tuple is the
    #: application→HTTP boundary, and ``dict[str, Any]`` here was the blank
    #: the orchestration-owned TypedDict could not fill.
    steps: tuple[Step, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "content": self.content, "steps": list(self.steps)}


class AgentRunService:
    """A2A task lifecycle over a pluggable agent orchestrator (ADR 0003).

    Scope resolution and the durable continuation envelope live here; the
    decision loop is an injected :class:`~ai_assistant.ports.AgentOrchestrator`.
    Production wires the canonical LangGraph orchestrator and tests/embeddings
    wire the deterministic adapter — this application component contains no
    decision-loop implementation of its own.
    """

    def __init__(self, *, capability_scopes: Mapping[str, Iterable[str]] | None = None,
                 orchestrator: AgentOrchestrator | None = None) -> None:
        self._orchestrator = orchestrator
        self._capability_scopes = (
            {str(name): frozenset(scopes) for name, scopes in capability_scopes.items()}
            if capability_scopes is not None else None
        )

    def _scopes_for(self, task: AgentTask) -> frozenset[str]:
        """Resolve least-privilege scopes for a task from its capability.

        ADR 0032: the lattice itself lives in ``domain.security`` — the
        scopes are a fact about principals, not about the transport that
        resolves them.
        """
        return scopes_for_capability(task.capability, self._capability_scopes)

    def run(self, task: AgentTask) -> dict[str, Any]:
        principal = Principal(
            subject=f"a2a:{task.id}",
            scopes=self._scopes_for(task),
            classification=task.classification,
        )
        if self._orchestrator is None:
            return AgentRunResult("failed", "No agent orchestrator is configured", ()).to_dict()
        return self._orchestrator.run(task, principal)
