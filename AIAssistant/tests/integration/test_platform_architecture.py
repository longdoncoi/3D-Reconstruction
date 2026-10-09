"""Contract tests for the modular AI Agent Platform boundary."""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_assistant.adapters.a2a import build_a2a_router
from ai_assistant.adapters.orchestration import DeterministicAgentOrchestrator
from ai_assistant.adapters.persistence import SqliteTaskStore
from ai_assistant.application.agent_runs import AgentRunService
from ai_assistant.application.tasks import TaskService
from ai_assistant.application.tools import ToolExecutionService
from ai_assistant.domain.security import Principal
from ai_assistant.domain.tasks import AgentTask, TaskStatus
from ai_assistant.domain.tools import SideEffect, ToolSpec
from ai_assistant.plugins.registry import PluginRegistry


class PlatformArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.principal = Principal("test", frozenset({"project.read", "project.write"}))
        self.registry = PluginRegistry(frozenset({"test.plugin"}))

    def test_tool_execution_has_one_policy_path(self):
        self.registry.register_tool(
            ToolSpec(
                "inspect", "Inspect",
                {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"],
                 "additionalProperties": False},
                1, plugin_id="test.plugin", required_scope="project.read",
            ),
            lambda params: {"echo": params["value"]},
        )
        result = ToolExecutionService(self.registry).execute("inspect", {"value": "ok"}, self.principal)
        self.assertTrue(result.success)
        self.assertEqual(result.payload["echo"], "ok")
        self.assertEqual(
            ToolExecutionService(self.registry).execute("inspect", {"unexpected": "no"}, self.principal).error_code,
            "invalid_input",
        )

    def test_mutating_tool_is_denied_before_approval(self):
        self.registry.register_tool(
            ToolSpec("write", "Write", {}, 1, side_effect=SideEffect.WRITE, requires_approval=True,
                     required_scope="project.write", plugin_id="test.plugin"),
            lambda _params: {"success": True},
        )
        service = ToolExecutionService(self.registry)
        self.assertEqual(service.execute("write", {}, self.principal).error_code, "approval_required")
        self.assertTrue(service.execute("write", {}, self.principal, approval_granted=True).success)

    def test_task_lifecycle_is_durable_and_cancelable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SqliteTaskStore(Path(directory) / "tasks.sqlite")
            service = TaskService(store, lambda task: {"answer": task.message}, frozenset({"supervisor"}))
            task = service.submit("supervisor", "hello")
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                current = service.get(task.id)
                if current and current.status == TaskStatus.COMPLETED:
                    break
                time.sleep(0.01)
            self.assertEqual(service.get(task.id).status, TaskStatus.COMPLETED)
            self.assertTrue(service.events(task.id))
            service.close()

    def test_unknown_capability_is_explicitly_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            service = TaskService(SqliteTaskStore(Path(directory) / "tasks.sqlite"), lambda _task: {}, frozenset())
            task = service.submit("not-registered", "hello")
            self.assertEqual(task.status, TaskStatus.REJECTED)
            service.close()

    def test_a2a_agent_card_and_task_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            service = TaskService(
                SqliteTaskStore(Path(directory) / "tasks.sqlite"), lambda task: {"answer": task.message},
                frozenset({"supervisor"}),
            )
            app = FastAPI()
            app.include_router(build_a2a_router(service, "test-agent", "1.0.0"))
            with TestClient(app) as client:
                card = client.get("/.well-known/agent.json")
                self.assertEqual(card.status_code, 200)
                self.assertEqual(card.json()["protocolVersion"], "0.3")
                response = client.post("/a2a/tasks/send", json={
                    "capability": "supervisor",
                    "message": {"role": "user", "parts": [{"kind": "text", "text": "hello"}]},
                })
                self.assertEqual(response.status_code, 200)
                task_id = response.json()["task"]["id"]
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    state = client.get(f"/a2a/tasks/{task_id}").json()["task"]["status"]["state"]
                    if state == TaskStatus.COMPLETED:
                        break
                    time.sleep(0.01)
                self.assertEqual(state, TaskStatus.COMPLETED)
                self.assertEqual(len(client.get("/a2a/tasks").json()["tasks"]), 1)
            service.close()

    def test_current_a2a_rest_contract_and_redacted_pause(self):
        self.registry.register_tool(
            ToolSpec("write", "Write", {"type": "object", "properties": {"password": {"type": "string"}},
                                            "required": ["password"]}, 1, side_effect=SideEffect.WRITE,
                     requires_approval=True, required_scope="project.write", plugin_id="test.plugin"),
            lambda _params: {"saved": True},
        )
        agent = AgentRunService(orchestrator=DeterministicAgentOrchestrator(
            complete=lambda _messages, _temperature: '{"kind":"tool","tool":"write","params":{"password":"secret"}}',
            tools=ToolExecutionService(self.registry), tool_descriptions=lambda: "write", max_iterations=1,
        ))
        with tempfile.TemporaryDirectory() as directory:
            service = TaskService(SqliteTaskStore(Path(directory) / "tasks.sqlite"), agent.run, frozenset({"supervisor"}))
            app = FastAPI()
            app.include_router(build_a2a_router(service, "test-agent", "1.0.0"))
            with TestClient(app) as client:
                response = client.post("/message:send", headers={"A2A-Version": "0.3"}, json={
                    "message": {"role": "ROLE_USER", "messageId": "m1", "parts": [{"text": "change"}]},
                })
                self.assertEqual(response.status_code, 200)
                task_id = response.json()["task"]["id"]
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    task = client.get(f"/tasks/{task_id}", headers={"A2A-Version": "0.3"}).json()["task"]
                    if task["status"]["state"] == "input-required":
                        break
                    time.sleep(0.01)
                self.assertEqual(task["status"]["state"], "input-required")
                self.assertNotIn("secret", json.dumps(task))
                resumed = client.post(f"/tasks/{task_id}:resume", headers={"A2A-Version": "0.3"},
                                      json={"input": {"approved": True}})
                self.assertEqual(resumed.status_code, 200)
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    completed = client.get(f"/tasks/{task_id}", headers={"A2A-Version": "0.3"}).json()["task"]
                    if completed["status"]["state"] == "completed":
                        break
                    time.sleep(0.01)
                self.assertEqual(completed["status"]["state"], "completed")
            service.close()

    def test_framework_independent_agent_loop_stops_for_approval(self):
        self.registry.register_tool(
            ToolSpec("write", "Write", {}, 1, side_effect=SideEffect.WRITE, requires_approval=True,
                     required_scope="project.write", plugin_id="test.plugin"),
            lambda _params: {"success": True},
        )
        agent = AgentRunService(orchestrator=DeterministicAgentOrchestrator(
            complete=lambda _messages, _temperature: '{"kind":"tool","tool":"write","params":{}}',
            tools=ToolExecutionService(self.registry), tool_descriptions=lambda: "write", max_iterations=1,
        ))
        result = agent.run(AgentTask.new("supervisor", "change"))
        self.assertEqual(result["status"], "input_required")

    def test_task_service_persists_input_required_agent_state(self):
        with tempfile.TemporaryDirectory() as directory:
            service = TaskService(
                SqliteTaskStore(Path(directory) / "tasks.sqlite"),
                lambda _task: {"status": "input_required", "steps": []},
                frozenset({"supervisor"}),
            )
            task = service.submit("supervisor", "needs approval")
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                current = service.get(task.id)
                if current and current.status == TaskStatus.INPUT_REQUIRED:
                    break
                time.sleep(0.01)
            self.assertEqual(service.get(task.id).status, TaskStatus.INPUT_REQUIRED)
            service.close()

    def test_a2a_resumes_an_approved_agent_tool(self):
        self.registry.register_tool(
            ToolSpec("write", "Write", {}, 1, side_effect=SideEffect.WRITE, requires_approval=True,
                     required_scope="project.write", plugin_id="test.plugin"),
            lambda _params: {"saved": True},
        )
        responses = iter((
            '{"kind":"tool","tool":"write","params":{}}',
            '{"kind":"final","content":"done"}',
        ))
        agent = AgentRunService(orchestrator=DeterministicAgentOrchestrator(
            complete=lambda _messages, _temperature: next(responses), tools=ToolExecutionService(self.registry),
            tool_descriptions=lambda: "write", max_iterations=2,
        ))
        with tempfile.TemporaryDirectory() as directory:
            service = TaskService(SqliteTaskStore(Path(directory) / "tasks.sqlite"), agent.run, frozenset({"supervisor"}))
            task = service.submit("supervisor", "change")
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and service.get(task.id).status != TaskStatus.INPUT_REQUIRED:
                time.sleep(0.01)
            self.assertEqual(service.get(task.id).status, TaskStatus.INPUT_REQUIRED)
            self.assertIsNotNone(service.resume(task.id, {"approved": True}))
            while time.monotonic() < deadline and service.get(task.id).status != TaskStatus.COMPLETED:
                time.sleep(0.01)
            final = service.get(task.id)
            self.assertEqual(final.status, TaskStatus.COMPLETED)
            self.assertEqual(final.result["content"], "done")
            service.close()


if __name__ == "__main__":
    unittest.main()
