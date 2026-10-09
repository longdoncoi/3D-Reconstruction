"""Contract tests for the single LangGraph orchestration path (ADR 0003).

The A2A task lifecycle in ``application/agent_runs.py`` must be able to reuse
the canonical LangGraph engine through the ``AgentOrchestrator`` port instead of
running a second, drifting decision loop. These tests pin the translation
between the two public contracts and the least-privilege principal binding.
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_assistant.adapters.orchestration.langgraph import LangGraphAgentOrchestrator
from ai_assistant.adapters.persistence import SqliteTaskStore
from ai_assistant.application.agent_runs import AgentRunService
from ai_assistant.application.tasks import TaskService
from ai_assistant.domain.security import Principal, current_principal
from ai_assistant.domain.tasks import AgentTask, TaskStatus


def _completed(content: str = "done") -> dict:
    return {"status": "completed", "steps": [{"type": "final_answer", "content": content}]}


class _FakeEngine:
    """Records the calls the orchestrator makes into the shared agent engine."""

    def __init__(self, results: list[dict]) -> None:
        self.results = list(results)
        self.executed = []
        self.approved = []
        self.ui_results = []

    def execute(self, request):
        self.executed.append(request)
        return self.results.pop(0)

    def approve(self, request):
        self.approved.append(request)
        return self.results.pop(0)

    def ui_action_result(self, request):
        self.ui_results.append(request)
        return self.results.pop(0)


def _orchestrator(engine: _FakeEngine) -> LangGraphAgentOrchestrator:
    return LangGraphAgentOrchestrator(engine.execute, engine.approve, engine.ui_action_result)


class LangGraphOrchestratorTests(unittest.TestCase):
    def test_completed_engine_result_maps_to_a2a_completion(self):
        engine = _FakeEngine([_completed("hello world")])
        task = AgentTask.new("supervisor", "hi", metadata={"language": "en", "temperature": 0.1})

        result = _orchestrator(engine).run(task, Principal("a2a", frozenset({"project.read"})))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["content"], "hello world")
        self.assertEqual(engine.executed[0].task, "hi")
        self.assertEqual(engine.executed[0].session_id, task.id)
        self.assertEqual(engine.executed[0].language, "en")

    def test_approval_pause_round_trips_through_the_continuation(self):
        engine = _FakeEngine([
            {"status": "pending_approval", "action_id": "act-1", "steps": [
                {"type": "pending_approval", "tool": "write", "params": {"path": "a.txt"}},
            ]},
        ])
        task = AgentTask.new("supervisor", "change", metadata={"language": "vi"})

        paused = _orchestrator(engine).run(task, Principal("a2a", frozenset({"project.write"})))

        self.assertEqual(paused["status"], "input_required")
        self.assertEqual(paused["continuation"]["kind"], "approval")
        self.assertEqual(paused["continuation"]["action_id"], "act-1")
        self.assertEqual(paused["continuation"]["tool"], "write")
        self.assertEqual(engine.approved, [])

        engine.results.append(_completed("applied"))
        resume_task = task.with_metadata({
            "language": "vi",
            "agent_run.continuation": paused["continuation"],
            "agent_run.resume": {"approved": True},
        })
        resumed = _orchestrator(engine).run(resume_task, Principal("a2a", frozenset({"project.write"})))

        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["content"], "applied")
        self.assertEqual(len(engine.approved), 1)
        self.assertEqual(engine.approved[0].action_id, "act-1")
        self.assertTrue(engine.approved[0].approved)
        self.assertEqual(engine.approved[0].session_id, task.id)

    def test_rejected_approval_completes_without_leaking_engine_shape(self):
        engine = _FakeEngine([{"status": "rejected", "steps": []}])
        paused = {
            "kind": "approval", "action_id": "act-2", "tool": "write", "params": {},
        }
        task = AgentTask.new("supervisor", "change").with_metadata({
            "agent_run.continuation": paused, "agent_run.resume": {"approved": False},
        })

        result = _orchestrator(engine).run(task, Principal("a2a", frozenset({"project.write"})))

        self.assertEqual(result["status"], "completed")
        self.assertIn("not approved", result["content"])

    def test_least_privilege_principal_is_bound_during_execution(self):
        observed: list[Principal | None] = []

        def execute(_request):
            observed.append(current_principal())
            return _completed()

        orchestrator = LangGraphAgentOrchestrator(execute, lambda _request: _completed(), None)
        principal = Principal("a2a:task-1", frozenset({"project.read"}))

        orchestrator.run(AgentTask.new("supervisor", "hi"), principal)

        self.assertEqual(observed, [principal])
        self.assertIsNone(current_principal(), "principal must not leak past the call")

    def test_desktop_ack_resume_passes_the_result_back_to_the_engine(self):
        engine = _FakeEngine([_completed("desktop done")])
        continuation = {"kind": "desktop_ack", "request_id": "req-9", "tool": "application_action"}
        task = AgentTask.new("supervisor", "open it").with_metadata({
            "agent_run.continuation": continuation,
            "agent_run.resume": {"success": True, "result": {"opened": True}},
        })

        result = _orchestrator(engine).run(task, Principal("a2a", frozenset({"desktop.action"})))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(engine.ui_results[0].request_id, "req-9")
        self.assertEqual(engine.ui_results[0].result, {"opened": True})


class AgentRunServiceOrchestrationTests(unittest.TestCase):
    def test_service_delegates_to_the_injected_orchestrator(self):
        calls: list[tuple[str, frozenset[str]]] = []

        class _RecordingOrchestrator:
            def run(self, task, principal):
                calls.append((task.id, principal.scopes))
                return {"status": "completed", "content": "delegated", "steps": []}

        service = AgentRunService(
            orchestrator=_RecordingOrchestrator(),
            capability_scopes={"supervisor": ["project.read"]},
        )
        task = AgentTask.new("supervisor", "hi")

        result = service.run(task)

        self.assertEqual(result["content"], "delegated")
        self.assertEqual(calls, [(task.id, frozenset({"project.read"}))])

    def test_unknown_capability_falls_back_to_read_only_scopes(self):
        seen: list[frozenset[str]] = []

        class _RecordingOrchestrator:
            def run(self, _task, principal):
                seen.append(principal.scopes)
                return {"status": "completed", "content": "", "steps": []}

        service = AgentRunService(
            orchestrator=_RecordingOrchestrator(),
            capability_scopes={"supervisor": ["project.write"]},
        )
        service.run(AgentTask.new("not-listed", "hi"))

        self.assertEqual(seen, [frozenset({"project.read"})])


class A2AOrchestratorIntegrationTests(unittest.TestCase):
    def test_durable_task_lifecycle_runs_through_the_orchestrator_port(self):
        engine = _FakeEngine([
            {"status": "pending_approval", "action_id": "act-e2e", "steps": [
                {"type": "pending_approval", "tool": "write", "params": {}},
            ]},
            _completed("finished"),
        ])
        run_service = AgentRunService(
            orchestrator=_orchestrator(engine),
            capability_scopes={"supervisor": ["project.write"]},
        )
        with tempfile.TemporaryDirectory() as directory:
            tasks = TaskService(
                SqliteTaskStore(Path(directory) / "tasks.sqlite"),
                run_service.run,
                frozenset({"supervisor"}),
            )
            task = tasks.submit("supervisor", "change")
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and tasks.get(task.id).status != TaskStatus.INPUT_REQUIRED:
                time.sleep(0.01)
            self.assertEqual(tasks.get(task.id).status, TaskStatus.INPUT_REQUIRED)

            self.assertIsNotNone(tasks.resume(task.id, {"approved": True}))
            while time.monotonic() < deadline and tasks.get(task.id).status != TaskStatus.COMPLETED:
                time.sleep(0.01)
            final = tasks.get(task.id)
            self.assertEqual(final.status, TaskStatus.COMPLETED)
            self.assertEqual(final.result["content"], "finished")
            self.assertEqual(engine.approved[0].action_id, "act-e2e")
            tasks.close()


if __name__ == "__main__":
    unittest.main()
