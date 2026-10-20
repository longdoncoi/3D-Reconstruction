"""Guard-path coverage for the ``AgentService`` facade seams.

The facade itself is thin, but its thinness is the contract: pending-action
durability (save only when something actually expired), the coding-agent
prompt branch, attachment rendering, the Qt retry index echo, approval
delegation and the reset path that removes the on-disk pending file — or logs
and moves on when the filesystem refuses. None of these arms runs in the
happy-path service tests.
"""
from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai_assistant.adapters.persistence import PendingActionStore
from ai_assistant.agents import context as agents_context
from ai_assistant.agents.models import AgentApproveRequest, AgentCancelRequest, AgentExecuteRequest
from ai_assistant.agents.service import AgentService
from ai_assistant.application.coordination import TaskCoordinator
from ai_assistant.domain.errors import ModelNotLoadedError, NotFoundError
from ai_assistant.tools.factory import create_tool_registry, platform_executors


def _service(**overrides) -> AgentService:
    kwargs = {
        "llm_runtime": SimpleNamespace(llm=object()),
        "pending_actions": MagicMock(),
        "pending_lock": threading.Lock(),
        "task_coordinator": MagicMock(),
    }
    kwargs.update(overrides)
    return AgentService(**kwargs)


def _run_context(result=None):
    """Patch the shared run wiring so ``execute`` stays a unit test."""
    from ai_assistant.agents.context import AgentRunContext

    return patch.object(
        AgentRunContext, "run",
        return_value=dict(result or {"status": "completed", "steps": []}),
    )


def _execute(task: str = "summarise the logs", **overrides) -> AgentExecuteRequest:
    fields = {"task": task}
    fields.update(overrides)
    return AgentExecuteRequest(**fields)


class PendingDurabilityTests(unittest.TestCase):
    def test_save_pending_delegates_to_the_store(self):
        store = MagicMock()
        _service(pending_actions=store)._save_pending()
        store.save.assert_called_once_with()

    def test_cleanup_saves_only_when_something_expired(self):
        store = MagicMock()
        store.cleanup.return_value = True
        service = _service(pending_actions=store)
        with self.assertLogs("ai_assistant.agents.service", level="INFO"):
            service._cleanup_pending()
        store.save.assert_called_once_with()

        store.reset_mock()
        store.cleanup.return_value = False
        service._cleanup_pending()
        store.save.assert_not_called()


class ExecuteBranchTests(unittest.TestCase):
    def _run(self, service: AgentService, request: AgentExecuteRequest):
        with _run_context() as run:
            result = service.execute(request)
        return result, run

    def test_a_coding_task_gets_the_coding_handoff(self):
        _, run = self._run(_service(), _execute("fix the crash on startup"))
        self.assertIn("CODING AGENT", run.call_args.kwargs["system_prompt"])

    def test_a_plain_task_has_no_coding_handoff(self):
        _, run = self._run(_service(), _execute("summarise the logs"))
        self.assertNotIn("CODING AGENT", run.call_args.kwargs["system_prompt"])

    def test_attachments_are_rendered_into_the_user_message(self):
        _, run = self._run(
            _service(), _execute("read these", attachments=["report.pdf", "notes.txt"]),
        )
        self.assertIn("[Attached files: report.pdf, notes.txt]", run.call_args.kwargs["task"])

    def test_history_entries_without_usable_content_are_skipped(self):
        _, run = self._run(_service(), _execute(
            "continue",
            history=[
                {"role": "user", "content": "  "},
                {"role": "system", "content": "not forwarded"},
                {"role": "assistant", "content": 42},
                {"role": "assistant", "content": "previous answer"},
            ],
        ))
        initial = run.call_args.kwargs["initial_messages"]
        forwarded = [m for m in initial if m["role"] == "assistant"]
        self.assertEqual(forwarded, [{"role": "assistant", "content": "previous answer"}])

    def test_the_retry_index_is_echoed_onto_the_result(self):
        result, _ = self._run(_service(), _execute("retry this", retry_message_index=3))
        self.assertEqual(result["retry_message_index"], 3)

    def test_no_retry_index_means_no_extra_key(self):
        result, _ = self._run(_service(), _execute("fresh run"))
        self.assertNotIn("retry_message_index", result)


class ApprovalDelegationTests(unittest.TestCase):
    def test_approve_cleans_up_then_delegates(self):
        service = _service()
        service._approval_service = MagicMock()
        service._approval_service.approve.return_value = {"status": "approved"}
        request = AgentApproveRequest(action_id="act-1", approved=True)
        self.assertEqual(service.approve(request), {"status": "approved"})
        service._approval_service.approve.assert_called_once_with(request)


class ResetStateTests(unittest.TestCase):
    def _service_with_real_store(self, directory: str) -> tuple[AgentService, Path]:
        path = Path(directory) / "pending_agent_actions.json"
        path.write_text("{}", encoding="utf-8")
        store = PendingActionStore(path)
        return _service(pending_actions=store), path

    def test_reset_clears_memory_and_removes_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            service, path = self._service_with_real_store(directory)
            service.reset_state()
            self.assertFalse(path.exists())

    def test_reset_survives_an_unremovable_file(self):
        """A locked or read-only file must not break the reset."""
        with tempfile.TemporaryDirectory() as directory:
            service, path = self._service_with_real_store(directory)
            with (
                patch("os.remove", side_effect=OSError("denied")),
                self.assertLogs("ai_assistant.agents.service", level="WARNING") as logged,
            ):
                service.reset_state()
            self.assertTrue(
                any("Unable to remove pending action state" in line for line in logged.output),
                logged.output,
            )
            self.assertTrue(path.exists())



# ── Facade lifecycle paths (moved from test_coverage_boost3) ────────────

class AgentServiceFacadeTests(unittest.TestCase):
    """Coverage for the AgentService facade edge paths."""

    def test_cancel_unknown_session_raises_not_found(self) -> None:
        service = AgentService()
        with self.assertRaises(NotFoundError):
            service.cancel(AgentCancelRequest(session_id="does-not-exist"))

    def test_cancel_known_session_returns_cancelled(self) -> None:
        coordinator = TaskCoordinator()
        service = AgentService(task_coordinator=coordinator)
        session = f"facade-cancel-{time.time()}"
        coordinator.start(session, task="t")
        result = service.cancel(AgentCancelRequest(session_id=session))
        self.assertEqual(result["status"], "cancelled")
        coordinator.finish(session, success=False)

    def test_execute_without_llm_raises_model_not_loaded(self) -> None:
        service = AgentService(llm_runtime=None)
        with self.assertRaises(ModelNotLoadedError):
            service.execute(AgentExecuteRequest(task="do something"))

    def test_model_lease_uses_and_falls_back_from_the_runtime_gate(self) -> None:
        class _Gate:
            def __init__(self) -> None:
                self.entered = False
                self.exited = False

            def __enter__(self):
                self.entered = True
                return self

            def __exit__(self, *exc) -> bool:
                self.exited = True
                return False

        class _LeasedRuntime:
            llm = object()
            llm_lock = threading.RLock()

            def __init__(self) -> None:
                self.gate = _Gate()

            def model_lease(self):
                return self.gate

        leased = _LeasedRuntime()
        service = AgentService(llm_runtime=leased)
        with service._model_lease():
            pass
        self.assertTrue(leased.gate.entered)
        self.assertTrue(leased.gate.exited)

        legacy = AgentService(llm_runtime=SimpleNamespace(
            llm=object(), llm_lock=threading.RLock(),
        ))
        with legacy._model_lease():
            pass

    def test_execute_delegates_to_agent_runner(self) -> None:
        mock_llm = MagicMock()
        mock_llm.llm = MagicMock()
        service = AgentService(llm_runtime=mock_llm)
        session = f"facade-exec-{time.time()}"
        with patch.object(
            agents_context, "run_langgraph_agent", return_value={"status": "completed", "content": "ok"}
        ) as mock_run:
            result = service.execute(
                AgentExecuteRequest(
                    task="summarise", session_id=session, retry_message_index=3,
                    history=[{"role": "user", "content": "  "}, {"role": "assistant", "content": "hi"}],
                    attachments=["a.txt"],
                )
            )
        mock_run.assert_called_once()
        # The facade must inject its own wiring (LLM, catalog, pending store)
        # into the runner rather than relying on runner-side defaults.
        kwargs = mock_run.call_args.kwargs
        deps = kwargs["deps"]
        self.assertIs(deps.llm_runtime, mock_llm)
        self.assertIs(deps.tool_registry, service.tool_registry)
        self.assertEqual(result["retry_message_index"], 3)

    def test_reset_state_clears_pending_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            from ai_assistant.adapters.persistence import PendingActionStore
            store = PendingActionStore(str(Path(tmpdir) / "pending.json"))
            store["key-1"] = {"tool": "read_file"}
            store.save()

            service = AgentService(pending_actions=store)
            self.assertIn("key-1", service.pending_actions)
            service.reset_state()
            self.assertNotIn("key-1", service.pending_actions)
            self.assertFalse(Path(store.path).exists())


# ── Facade construction paths (moved from test_coverage_boost) ────

class AgentServiceTests(unittest.TestCase):
    """Test AgentService facade coverage."""

    def test_agent_service_initialization(self) -> None:
        """AgentService initializes with default dependencies."""
        service = AgentService()
        self.assertIsNotNone(service.pending_actions)
        self.assertIsNotNone(service.pending_lock)
        self.assertIsNotNone(service.tool_registry)

    def test_agent_service_with_custom_dependencies(self) -> None:
        """AgentService accepts custom dependencies."""
        from ai_assistant.adapters.persistence import PendingActionStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PendingActionStore(str(Path(tmpdir) / "pending.json"))
            lock = threading.Lock()
            service = AgentService(
                pending_actions=store,
                pending_lock=lock,
            )
            self.assertIs(service.pending_actions, store)
            self.assertIs(service.pending_lock, lock)

    def test_agent_service_default_initialization(self) -> None:
        """AgentService initializes without arguments using default pending path."""
        service = AgentService()
        self.assertIsNotNone(service.pending_actions)
        self.assertIsNotNone(service.pending_lock)

    def test_platform_executors_returns_dict(self) -> None:
        """platform_executors returns a dict of executor functions."""
        executors = platform_executors(create_tool_registry())
        self.assertIsInstance(executors, dict)
        self.assertGreater(len(executors), 0)

    def test_tool_registry_has_tools(self) -> None:
        """A created tool registry contains registered tools."""
        tools = create_tool_registry().get_all()
        self.assertGreater(len(tools), 0)

if __name__ == "__main__":
    unittest.main()
