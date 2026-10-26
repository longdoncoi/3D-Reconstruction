"""Unit tests for :class:`ai_assistant.agents.context.AgentRunContext`.

``AgentRunContext`` consolidated the ~26 keyword arguments that ``AgentService``,
``ApprovalService`` and ``UIActionService`` each used to spell out identically
before calling ``run_langgraph_agent`` (ADR 0012). These tests pin the exact
wiring the context injects, so the deduplication cannot silently drop a
dependency or hand the services different behaviour than before.
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_assistant.adapters.persistence import PendingActionStore
from ai_assistant.agents import context as agents_context
from ai_assistant.agents.context import (
    RESUME_ACTION_ID,
    AgentRunContext,
)
from ai_assistant.agents.pending_store import MemoryPendingStore
from ai_assistant.application.coordination import TaskCoordinator


def _capture(**overrides) -> dict:
    """Run a context and return the kwargs the runner actually received."""
    ctx = AgentRunContext(**overrides)
    with patch.object(agents_context, "run_langgraph_agent", return_value={"status": "completed"}) as runner:
        ctx.run(
            system_prompt="sys", task="t", session_id="s", temperature=0.3,
            language="vi", request_started=0.0,
        )
    return runner.call_args.kwargs


class AgentRunContextWiringTests(unittest.TestCase):
    """The context must inject the shared dependencies into the runner."""

    def test_injects_llm_runtime_and_tool_dependencies(self) -> None:
        llm, gateway, registry = MagicMock(), MagicMock(), MagicMock()
        store = PendingActionStore("")
        lock = threading.Lock()
        coordinator = TaskCoordinator()
        kwargs = _capture(
            llm_runtime=llm, tool_gateway=gateway, tool_registry=registry,
            pending_actions=store, pending_lock=lock, task_coordinator=coordinator,
        )
        deps = kwargs["deps"]
        self.assertIs(deps.llm_runtime, llm)
        # The runner gets the *execution* view, not the gateway object: the
        # view delegates every execution call but physically exposes no
        # ``issue_approval_grant`` (separation of duties by construction).
        self.assertIs(deps.tool_gateway._gateway, gateway)
        self.assertFalse(hasattr(deps.tool_gateway, "issue_approval_grant"))
        self.assertIs(deps.tool_registry, registry)
        self.assertIs(deps.pending_actions, store)
        self.assertIs(deps.pending_lock, lock)
        self.assertIs(deps.task_coordinator, coordinator)

    def test_injects_supervisor_and_observability_callables(self) -> None:
        kwargs = _capture()
        deps = kwargs["deps"]
        # Every supervisor/observability/LLM callable the runner needs must be
        # supplied by the context, not left to runner-side defaults.
        for name in (
            "delegate_fn", "authorise_delegation_fn", "audit_agent_fn", "verify_result_fn",
            "reflect_result_fn", "is_coding_task_fn",
            "record_tool_fn", "record_schema_error_fn", "langsmith_trace_fn", "span_fn",
            "backend_mode_fn", "openai_compatible_fn", "record_token_usage_fn",
        ):
            self.assertTrue(callable(getattr(deps, name)), f"{name} must be callable")

    def test_graph_engine_is_injected(self) -> None:
        graph = object()
        kwargs = _capture(local_agent_graph=graph)
        deps = kwargs["deps"]
        self.assertIs(deps.local_agent_graph, graph)

    def test_defaults_to_real_tool_validator_and_action_id(self) -> None:
        kwargs = _capture()
        self.assertIs(kwargs["validate_tool_call_fn"], agents_context.validate_tool_call)
        self.assertIs(kwargs["generate_action_id_fn"], agents_context.generate_action_id)

    def test_caller_can_override_shared_defaults(self) -> None:
        """A caller must be able to swap in a custom validator/id."""
        custom_validator = lambda *a, **k: (None, None)  # noqa: E731
        with patch.object(agents_context, "run_langgraph_agent", return_value={}) as runner:
            AgentRunContext().run(
                system_prompt="s", task="t", session_id="s", temperature=0.3,
                language="vi", request_started=0.0,
                validate_tool_call_fn=custom_validator,
                generate_action_id_fn=RESUME_ACTION_ID,
            )
        self.assertIs(runner.call_args.kwargs["validate_tool_call_fn"], custom_validator)
        self.assertIs(runner.call_args.kwargs["generate_action_id_fn"], RESUME_ACTION_ID)

    def test_passes_per_call_arguments_through(self) -> None:
        with patch.object(agents_context, "run_langgraph_agent", return_value={}) as runner:
            AgentRunContext().run(
                system_prompt="sys", task="do", session_id="sess", temperature=0.1,
                language="en", request_started=1.5, initial_messages=[{"role": "user", "content": "x"}],
                initial_steps=[{"type": "memory_recall", "count": 1}],
                prior_plan=["Buoc mot"],
                approval_granted=True, approval_token="tok", granted_fingerprint="fp",
                prior_step_count=4,
            )
        seen = runner.call_args.kwargs
        self.assertEqual(seen["system_prompt"], "sys")
        self.assertEqual(seen["task"], "do")
        self.assertEqual(seen["session_id"], "sess")
        self.assertEqual(seen["temperature"], 0.1)
        self.assertEqual(seen["language"], "en")
        self.assertEqual(seen["request_started"], 1.5)
        self.assertEqual(seen["initial_messages"], [{"role": "user", "content": "x"}])
        self.assertEqual(seen["initial_steps"], [{"type": "memory_recall", "count": 1}])
        self.assertEqual(seen["prior_plan"], ["Buoc mot"])
        self.assertTrue(seen["approval_granted"])
        self.assertEqual(seen["approval_token"], "tok")
        self.assertEqual(seen["granted_fingerprint"], "fp")
        self.assertEqual(seen["prior_step_count"], 4)

    def test_resume_arguments_default_to_inactive(self) -> None:
        kwargs = _capture()
        self.assertFalse(kwargs["approval_granted"])
        self.assertEqual(kwargs["approval_token"], "")
        self.assertEqual(kwargs["granted_fingerprint"], "")
        self.assertFalse(kwargs["resume_with_reflection"])
        self.assertIsNone(kwargs["event_sink"])


class AgentRunContextPendingTests(unittest.TestCase):
    """``save_pending`` must persist the injected store (the runner's hook)."""

    def test_save_pending_persists_the_injected_store(self) -> None:
        store = MagicMock(spec=PendingActionStore)
        AgentRunContext(pending_actions=store).save_pending()
        store.save.assert_called_once()

    def test_save_pending_is_wired_as_runner_hook(self) -> None:
        store = MagicMock(spec=PendingActionStore)
        kwargs = _capture(pending_actions=store)
        kwargs["deps"].save_pending_fn()
        store.save.assert_called_once()

    def test_defaults_build_own_store_lock_and_coordinator(self) -> None:
        ctx = AgentRunContext()
        self.assertIsInstance(ctx.pending_actions, MemoryPendingStore)
        self.assertIsInstance(ctx.task_coordinator, TaskCoordinator)
        # Two bare contexts must not share mutable state.
        self.assertIsNot(ctx.pending_lock, AgentRunContext().pending_lock)


class ResumeHelpersTests(unittest.TestCase):
    """The resume-path id no-op must be inert and correctly shaped."""

    def test_resume_action_id_is_empty_string(self) -> None:
        self.assertEqual(RESUME_ACTION_ID(), "")


class ContextGuardMethodTests(unittest.TestCase):
    """``require_llm`` and ``resume_available`` gate the resume branch."""

    def test_require_llm_raises_without_a_backend(self) -> None:
        from ai_assistant.domain.errors import ModelNotLoadedError

        with self.assertRaises(ModelNotLoadedError):
            AgentRunContext(llm_runtime=None).require_llm()

    def test_resume_available_is_false_without_a_graph(self) -> None:
        self.assertFalse(AgentRunContext(local_agent_graph=None).resume_available())

    def test_resume_available_requires_a_live_backend_when_asked(self) -> None:
        ctx = AgentRunContext(llm_runtime=MagicMock(llm=None), local_agent_graph=object())
        self.assertFalse(ctx.resume_available(require_llm_ready=True))
        self.assertTrue(ctx.resume_available())

    def test_resume_available_is_true_with_graph_and_backend(self) -> None:
        ctx = AgentRunContext(llm_runtime=MagicMock(llm=MagicMock()), local_agent_graph=object())
        self.assertTrue(ctx.resume_available(require_llm_ready=True))


class ServiceContextWiringTests(unittest.TestCase):
    """Each service must build a context that matches its pre-refactor call."""

    def _sub_service(self, cls):
        llm, gateway, registry = MagicMock(), MagicMock(), MagicMock()
        store = PendingActionStore("")
        coordinator = TaskCoordinator()
        svc = cls(
            llm_runtime=llm, pending_actions=store, tool_gateway=gateway,
            tool_registry=registry, task_coordinator=coordinator,
        )
        return svc, llm, gateway, registry, store, coordinator

    def test_approval_service_run_uses_resume_noops_and_no_a2a(self) -> None:
        from ai_assistant.agents.approval_service import ApprovalService
        svc, llm, gateway, registry, store, coordinator = self._sub_service(ApprovalService)
        with patch.object(agents_context, "run_langgraph_agent", return_value={}) as runner:
            svc._ctx.run(
                system_prompt="s", task="t", session_id="s", temperature=0.3,
                language="vi", request_started=0.0,
                generate_action_id_fn=RESUME_ACTION_ID,
            )
        kwargs = runner.call_args.kwargs
        deps = kwargs["deps"]
        self.assertIsNone(deps.a2a_router)
        self.assertIs(deps.llm_runtime, llm)
        # Resume paths hand the runner the same execution-only view.
        self.assertIs(deps.tool_gateway._gateway, gateway)
        self.assertFalse(hasattr(deps.tool_gateway, "issue_approval_grant"))
        self.assertIs(deps.tool_registry, registry)
        self.assertIs(deps.pending_actions, store)
        self.assertIs(deps.task_coordinator, coordinator)
        # Resume keeps the real validator (review 27); only the id is a no-op.
        self.assertIs(kwargs["validate_tool_call_fn"], agents_context.validate_tool_call)
        self.assertIs(kwargs["generate_action_id_fn"], RESUME_ACTION_ID)

    def test_ui_action_service_run_uses_resume_noops_and_no_a2a(self) -> None:
        from ai_assistant.agents.ui_action_service import UIActionService
        svc, llm, _gateway, _registry, store, coordinator = self._sub_service(UIActionService)
        with patch.object(agents_context, "run_langgraph_agent", return_value={}) as runner:
            svc._ctx.run(
                system_prompt="s", task="t", session_id="s", temperature=0.3,
                language="vi", request_started=0.0,
                generate_action_id_fn=RESUME_ACTION_ID,
            )
        kwargs = runner.call_args.kwargs
        deps = kwargs["deps"]
        self.assertIsNone(deps.a2a_router)
        self.assertIs(deps.llm_runtime, llm)
        self.assertIs(deps.pending_actions, store)
        self.assertIs(deps.task_coordinator, coordinator)
        self.assertIs(kwargs["validate_tool_call_fn"], agents_context.validate_tool_call)
        self.assertIs(kwargs["generate_action_id_fn"], RESUME_ACTION_ID)

    def test_agent_service_run_keeps_a2a_router(self) -> None:
        from ai_assistant.agents.service import AgentService
        router, llm = MagicMock(), MagicMock()
        llm.llm = MagicMock()
        svc = AgentService(llm_runtime=llm, a2a_router=router)
        with patch.object(agents_context, "run_langgraph_agent", return_value={}) as runner:
            svc._ctx.run(
                system_prompt="s", task="t", session_id="s", temperature=0.3,
                language="vi", request_started=0.0,
            )
        kwargs = runner.call_args.kwargs
        # The main execute path keeps the A2A router; sub-services do not.
        self.assertIs(kwargs["deps"].a2a_router, router)
        self.assertIs(kwargs["deps"].llm_runtime, llm)
        self.assertIs(kwargs["validate_tool_call_fn"], agents_context.validate_tool_call)

    def test_sub_services_share_facade_pending_state(self) -> None:
        from ai_assistant.agents.service import AgentService
        store = PendingActionStore("")
        facade = AgentService(pending_actions=store)
        self.assertIs(facade._approval_service.pending_actions, store)
        self.assertIs(facade._ui_action_service.pending_actions, store)
        self.assertIs(facade._ctx.pending_actions, store)


class ExecutionOnlyGatewayViewTests(unittest.TestCase):
    """The runner's gateway view delegates execution and cannot mint grants.

    ``RunDeps`` declares the least-privilege ``ToolExecutionGateway`` port, but
    the composition root builds one object that also mints approval grants.
    The view below is what makes that split physical instead of a type-annotation
    story (review finding 15): every execution method forwards, and the grant
    API simply is not there.
    """

    def test_every_execution_method_delegates(self) -> None:
        gateway = MagicMock()
        view = agents_context._execution_view(gateway)
        view.execute("t", {"a": 1})
        view.execute_approved("t", {"a": 1}, "tok")
        view.preflight("t", {"a": 1}, "tok")
        self.assertTrue(view.approval_covers("t", {"a": 1}, "tok"))
        gateway.execute.assert_called_once_with("t", {"a": 1})
        gateway.execute_approved.assert_called_once_with("t", {"a": 1}, "tok")
        gateway.preflight.assert_called_once_with("t", {"a": 1}, "tok")
        gateway.approval_covers.assert_called_once_with("t", {"a": 1}, "tok")

    def test_the_grant_api_is_absent_from_the_view(self) -> None:
        view = agents_context._execution_view(MagicMock())
        self.assertFalse(hasattr(view, "issue_approval_grant"))
        self.assertFalse(hasattr(view, "issue_ui_ack_grant"))

    def test_an_unwired_gateway_stays_none(self) -> None:
        self.assertIsNone(agents_context._execution_view(None))


if __name__ == "__main__":
    unittest.main()
