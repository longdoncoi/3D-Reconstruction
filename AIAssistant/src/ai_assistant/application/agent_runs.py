"""Durable A2A agent lifecycle over a pluggable orchestration engine."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..domain.security import Principal
from ..domain.tasks import AgentTask
from ..ports import AgentOrchestrator

# Least-privilege defaults for remote/durable tasks. A capability that is not
# listed in the deployment scope map may only read.
_READ_ONLY_SCOPES = frozenset({"project.read"})
# Trusted in-process callers without an explicit scope map keep the historical
# broad scope set; remote A2A deployments must supply a capability scope map.
_TRUSTED_SCOPES = frozenset({"project.read", "project.write", "project.execute", "desktop.action"})


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    status: str
    content: str
    steps: tuple[dict[str, Any], ...]

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
        """Resolve least-privilege scopes for a task from its capability."""
        if self._capability_scopes is None:
            return _TRUSTED_SCOPES
        return self._capability_scopes.get(task.capability, _READ_ONLY_SCOPES)

    def run(self, task: AgentTask) -> dict[str, Any]:
        principal = Principal(
            subject=f"a2a:{task.id}",
            scopes=self._scopes_for(task),
            classification=task.classification,
        )
        if self._orchestrator is None:
            return AgentRunResult("failed", "No agent orchestrator is configured", ()).to_dict()
        return self._orchestrator.run(task, principal)
