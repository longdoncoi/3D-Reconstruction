"""Two A2A tenants never share long-term memory (ADR 0043).

The composition root builds **one** ``MemoryService`` for the process, so the
tenant has to travel with the call, not live in the service. The production
chain is ``AgentTask.owner`` → ``LangGraphAgentOrchestrator`` →
``AgentService.execute(owner=...)`` → recall / procedural lookup /
consolidation. This test wires the real sealed store, the real memory service
and the real orchestrator adapter through that chain (the decision loop itself
is stubbed, as a unit test must be) and proves both directions:

* tenant B does not recall tenant A's completed run, even when both ask the
  same task, and the local default namespace stays separate too;
* tenant B's consolidation does not replace tenant A's procedural plan, which
  is the erasure the shared record ids used to cause.
"""
from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ai_assistant.adapters.orchestration.langgraph import LangGraphAgentOrchestrator
from ai_assistant.adapters.persistence.memory_store import MemoryStore
from ai_assistant.adapters.persistence.storage_crypto import AesGcmFieldCipher
from ai_assistant.agents import context as agents_context
from ai_assistant.agents.service import AgentService
from ai_assistant.application.agent_runs import AgentRunService
from ai_assistant.application.memory import MemoryService
from ai_assistant.domain.memory import DEFAULT_MEMORY_OWNER
from ai_assistant.domain.tasks import AgentTask

TASK = "tim file main.cpp"


class _NullRuntime:
    """Satisfies the facade's model gate; the decision loop is stubbed."""

    llm = object()

    @contextlib.contextmanager
    def model_lease(self):
        yield None


def _completed(plan: list[str], answer: str) -> dict[str, Any]:
    return {
        "status": "completed",
        "steps": [
            {"type": "plan", "steps": plan},
            {"type": "final_answer", "content": answer},
        ],
    }


class TenantMemoryIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = MemoryStore(
            Path(self._tmp.name) / "memory.sqlite",
            cipher=AesGcmFieldCipher(b"m" * 32),
        )
        self.addCleanup(self.store.close)
        self.memory = MemoryService(self.store, owner=DEFAULT_MEMORY_OWNER)
        self.service = AgentService(
            llm_runtime=_NullRuntime(),  # type: ignore[arg-type]
            memory_service=self.memory,
        )
        self.results: list[dict[str, Any]] = []
        self.prompts: list[str] = []
        # The decision loop is the one thing a unit test cannot run; patch the
        # runner under the *real* context so recall and the context-owned
        # consolidation both execute on the production path.
        patcher = patch.object(
            agents_context, "run_langgraph_agent", side_effect=self._run,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.runs = AgentRunService(orchestrator=LangGraphAgentOrchestrator(
            execute=self.service.execute,
            approve=self.service.approve,
            ui_action_result=self.service.ui_action_result,
        ))

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        self.prompts.append(str(kwargs["system_prompt"]))
        return self.results.pop(0)

    def _submit(self, owner: str) -> dict[str, Any]:
        return self.runs.run(AgentTask.new("supervisor", TASK, owner=owner))

    @staticmethod
    def _text(records: list[dict[str, Any]]) -> str:
        return " ".join(str(record.get("text", "")) for record in records)

    def test_two_tenants_do_not_recall_each_other(self) -> None:
        self.results = [
            _completed(["A step"], "A-only-fact"),
            _completed(["B step"], "B-only-fact"),
        ]

        self._submit("tenant-a")
        self._submit("tenant-b")

        # Tenant B's run recalls before its own records exist; if the owner
        # filter were bypassed it would see tenant A's answer here.
        self.assertNotIn("A-only-fact", self.prompts[1])
        self.assertNotIn("A-only-fact", self._text(
            self.memory.recall(TASK, owner="tenant-b"),
        ))
        self.assertIn("A-only-fact", self._text(
            self.memory.recall(TASK, owner="tenant-a"),
        ))

    def test_a_same_task_write_does_not_erase_the_other_tenants_plan(self) -> None:
        self.results = [
            _completed(["A step"], "A-only-fact"),
            _completed(["B step"], "B-only-fact"),
        ]

        self._submit("tenant-a")
        self._submit("tenant-b")

        self.assertEqual(self.memory.procedural_plan(TASK, owner="tenant-a"), ["A step"])
        self.assertEqual(self.memory.procedural_plan(TASK, owner="tenant-b"), ["B step"])
        self.assertIsNone(self.memory.procedural_plan(TASK))

    def test_a_local_default_run_is_not_visible_to_a_tenant(self) -> None:
        self.results = [_completed([], "local-only-fact")]

        self._submit(DEFAULT_MEMORY_OWNER)

        self.assertNotIn("local-only-fact", self._text(
            self.memory.recall(TASK, owner="tenant-a"),
        ))
        self.assertIn("local-only-fact", self._text(
            self.memory.recall(TASK, owner=DEFAULT_MEMORY_OWNER),
        ))


if __name__ == "__main__":
    unittest.main()
