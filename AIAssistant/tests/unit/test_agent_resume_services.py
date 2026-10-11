"""Unit tests for the approval and desktop-ack resume paths (ADR 0004/0012).

These are the business-logic halves of ``ApprovalService`` and
``UIActionService`` that the ``AgentRunContext`` consolidation (ADR 0012)
touched: the HITL approve/reject lifecycle, the single-use grant issuance, the
multi-step desktop action chaining, and the LangGraph resume hand-off. Each
asserts on the observable result and on the wiring the service hands the
shared runner.
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_assistant.agents import context as agents_context  # noqa: E402
from ai_assistant.agents.approval_service import ApprovalService  # noqa: E402
from ai_assistant.agents.models import (  # noqa: E402
    AgentApproveRequest,
    AgentUiActionResultRequest,
)
from ai_assistant.agents.pending_store import PendingActionStore  # noqa: E402
from ai_assistant.agents.ui_action_service import UIActionService  # noqa: E402
from ai_assistant.domain.errors import (  # noqa: E402
    AuthorizationError,
    NotFoundError,
    UnprocessableRequestError,
)

# A desktop action id that validates with no required parameters, used by the
# multi-step chaining tests below.
VALID_ACTION = "admin.change_avatar"


def _llm() -> MagicMock:
    llm = MagicMock()
    llm.llm = MagicMock()
    return llm


def _approval_action(**over) -> dict:
    action = {
        "tool": "read_file",
        "params": {"path": "notes.txt"},
        "session_id": "sess-1",
        "task": "read the notes",
        "messages": [],
        "steps": [{"type": "plan", "content": "step"}],
        "iteration": 2,
        "temperature": 0.3,
        "approval_scope": "scope-1",
    }
    action.update(over)
    return action


def _ui_action(**over) -> dict:
    action = {
        "ui_ack": True,
        "params": {"action": VALID_ACTION},
        "task": "change avatar",
        "session_id": "sess-2",
        "steps": [{"type": "plan", "content": "step"}],
        "messages": [],
        "next_actions": [],
        "iteration": 1,
        "temperature": 0.3,
        "language": "vi",
    }
    action.update(over)
    return action


class _ServiceFixture(unittest.TestCase):
    """Shared wiring: an in-memory pending store and a fake LLM."""

    def setUp(self) -> None:
        self.store = PendingActionStore("")
        self.lock = threading.Lock()


class ApprovalServiceLifecycleTests(_ServiceFixture):
    """The HITL approve/reject lifecycle."""

    def _service(self, gateway=None, **over) -> ApprovalService:
        return ApprovalService(
            llm_runtime=_llm(), pending_actions=self.store,
            pending_lock=self.lock, tool_gateway=gateway, **over,
        )

    def test_shared_wiring_properties_surface_the_injected_objects(self) -> None:
        llm, gateway, registry = _llm(), MagicMock(), MagicMock()
        svc = ApprovalService(
            llm_runtime=llm, pending_actions=self.store, pending_lock=self.lock,
            tool_gateway=gateway, tool_registry=registry,
        )
        self.assertIs(svc.llm_runtime, llm)
        self.assertIs(svc.pending_actions, self.store)
        self.assertIs(svc.pending_lock, self.lock)
        self.assertIs(svc.tool_gateway, gateway)
        self.assertIs(svc.tool_registry, registry)
        self.assertIsNotNone(svc.task_coordinator)

    def test_approve_unknown_action_raises_not_found(self) -> None:
        with self.assertRaises(NotFoundError):
            self._service().approve(AgentApproveRequest(action_id="ghost", approved=True))

    def test_approve_cancelled_session_short_circuits(self) -> None:
        self.store["a-1"] = _approval_action()
        svc = self._service()
        svc.task_coordinator.start("sess-1", task="t")
        svc.task_coordinator.cancel("sess-1")
        out = svc.approve(AgentApproveRequest(action_id="a-1", approved=True, session_id="sess-1"))
        self.assertEqual(out["status"], "cancelled")
        self.assertEqual(out["action_id"], "a-1")
        # The cancelled branch must not leave a stale pending entry behind.
        self.assertNotIn("a-1", self.store)

    def test_approve_wrong_session_is_rejected_and_restored(self) -> None:
        self.store["a-1"] = _approval_action()
        svc = self._service()
        with self.assertRaises(AuthorizationError):
            svc.approve(AgentApproveRequest(action_id="a-1", approved=True, session_id="other"))
        # A failed authorisation must put the pending action back, not drop it.
        self.assertIn("a-1", self.store)

    def test_approve_missing_session_is_rejected(self) -> None:
        self.store["a-1"] = _approval_action()
        with self.assertRaises(AuthorizationError):
            self._service().approve(AgentApproveRequest(action_id="a-1", approved=True, session_id=""))

    def test_reject_returns_rejected_status_and_step(self) -> None:
        self.store["a-1"] = _approval_action()
        out = self._service().approve(
            AgentApproveRequest(action_id="a-1", approved=False, session_id="sess-1")
        )
        self.assertEqual(out["status"], "rejected")
        self.assertEqual(out["approval_scope"], "scope-1")
        self.assertEqual(out["prior_step_count"], 1)
        self.assertEqual(out["steps"][-1]["type"], "tool_result")
        self.assertTrue(out["steps"][-1]["result"]["rejected"])


class ApprovalExecutionTests(_ServiceFixture):
    """Grant issuance, execution, and the completed/resume fork."""

    def _service(self, gateway) -> ApprovalService:
        return ApprovalService(
            llm_runtime=_llm(), pending_actions=self.store,
            pending_lock=self.lock, tool_gateway=gateway,
        )

    def test_execute_without_gateway_reports_runtime_unconfigured(self) -> None:
        self.store["a-1"] = _approval_action()
        with patch.object(agents_context, "run_langgraph_agent", return_value={}):
            with patch("ai_assistant.agents.approval_service.use_langgraph_agent", return_value=False):
                out = self._service(None).approve(
                    AgentApproveRequest(action_id="a-1", approved=True, session_id="sess-1")
                )
        self.assertEqual(out["status"], "completed")
        # The tool step records the unconfigured-runtime error rather than
        # pretending the tool ran.
        tool_step = next(s for s in out["steps"] if s["type"] == "tool_result")
        self.assertEqual(tool_step["result"]["error_code"], "runtime_unconfigured")

    def test_issued_two_single_use_grants_bound_to_the_run(self) -> None:
        gateway = MagicMock()
        gateway.execute_approved.return_value = {"success": True}
        self.store["a-1"] = _approval_action()
        with patch.object(agents_context, "run_langgraph_agent", return_value={}):
            with patch("ai_assistant.agents.approval_service.use_langgraph_agent", return_value=False):
                self._service(gateway).approve(
                    AgentApproveRequest(action_id="a-1", approved=True, session_id="sess-1")
                )
        self.assertEqual(gateway.issue_approval_grant.call_count, 2)
        tool, params, run_token = gateway.issue_approval_grant.call_args_list[0].args
        direct_token = gateway.issue_approval_grant.call_args_list[1].args[2]
        self.assertEqual((tool, params), ("read_file", {"path": "notes.txt"}))
        self.assertEqual(run_token, "a-1")
        # The direct-execution grant is namespaced off the run token.
        self.assertEqual(direct_token, "a-1#direct")
        gateway.execute_approved.assert_called_once_with(
            "read_file", {"path": "notes.txt"}, "a-1#direct"
        )

    def test_approved_result_appends_tool_and_verification_steps(self) -> None:
        gateway = MagicMock()
        gateway.execute_approved.return_value = {"content": "notes"}
        self.store["a-1"] = _approval_action()
        with patch.object(agents_context, "run_langgraph_agent", return_value={}):
            with patch("ai_assistant.agents.approval_service.use_langgraph_agent", return_value=False):
                out = self._service(gateway).approve(
                    AgentApproveRequest(action_id="a-1", approved=True, session_id="sess-1")
                )
        types = [s["type"] for s in out["steps"]]
        self.assertEqual(types, ["plan", "tool_result", "verification"])
        self.assertEqual(out["prior_step_count"], 1)
        self.assertEqual(out["steps"][1]["result"], {"content": "notes"})

    def test_huge_tool_result_is_truncated_in_resumed_message(self) -> None:
        gateway = MagicMock()
        gateway.execute_approved.return_value = {"content": "x" * 9000}
        self.store["a-1"] = _approval_action()
        with patch.object(agents_context, "run_langgraph_agent", return_value={}) as runner:
            with patch("ai_assistant.agents.approval_service.use_langgraph_agent", return_value=True):
                self._service(gateway).approve(
                    AgentApproveRequest(action_id="a-1", approved=True, session_id="sess-1")
                )
        messages = runner.call_args.kwargs["initial_messages"]
        last = messages[-1]["content"]
        self.assertIn("[truncated]", last)
        self.assertLess(len(last), 9000)

    def test_resume_fork_hands_the_runner_its_approval_context(self) -> None:
        gateway = MagicMock()
        gateway.execute_approved.return_value = {"ok": True}
        self.store["a-1"] = _approval_action()
        with patch.object(agents_context, "run_langgraph_agent", return_value={"status": "completed"}) as runner:
            with patch("ai_assistant.agents.approval_service.use_langgraph_agent", return_value=True):
                self._service(gateway).approve(
                    AgentApproveRequest(action_id="a-1", approved=True, session_id="sess-1")
                )
        kwargs = runner.call_args.kwargs
        self.assertTrue(kwargs["approval_granted"])
        self.assertEqual(kwargs["approval_scope"], "scope-1")
        self.assertEqual(kwargs["approval_token"], "a-1")
        self.assertTrue(kwargs["granted_fingerprint"])
        self.assertEqual(kwargs["initial_iteration"], 2)
        self.assertEqual(kwargs["prior_step_count"], 1)
        # Resume reuses the first-pass tool result, so it swaps in the no-op
        # validator and the empty action-id generator.
        self.assertIs(kwargs["validate_tool_call_fn"], agents_context.RESUME_VALIDATE_TOOL_CALL)
        self.assertIs(kwargs["generate_action_id_fn"], agents_context.RESUME_ACTION_ID)
        self.assertIn({"type": "approval_granted", "scope_id": "scope-1",
                       "tool": "read_file", "action_id": "a-1"}, kwargs["initial_steps"])

    def test_resume_reuses_the_system_message_as_prompt(self) -> None:
        gateway = MagicMock()
        gateway.execute_approved.return_value = {"ok": True}
        self.store["a-1"] = _approval_action(
            messages=[{"role": "system", "content": "you are terse"}],
        )
        with patch.object(agents_context, "run_langgraph_agent", return_value={}) as runner:
            with patch("ai_assistant.agents.approval_service.use_langgraph_agent", return_value=True):
                self._service(gateway).approve(
                    AgentApproveRequest(action_id="a-1", approved=True, session_id="sess-1")
                )
        self.assertEqual(runner.call_args.kwargs["system_prompt"], "you are terse")


class UIActionServiceTests(_ServiceFixture):
    """Desktop ack processing, multi-step chaining, and finalisation."""

    def _service(self, **over) -> UIActionService:
        return UIActionService(
            llm_runtime=_llm(), pending_actions=self.store, pending_lock=self.lock,
            tool_gateway=over.pop("tool_gateway", None),
            tool_registry=over.pop("tool_registry", None),
        )

    def test_shared_wiring_properties_surface_the_injected_objects(self) -> None:
        llm, gateway, registry = _llm(), MagicMock(), MagicMock()
        svc = UIActionService(
            llm_runtime=llm, pending_actions=self.store, pending_lock=self.lock,
            tool_gateway=gateway, tool_registry=registry,
        )
        self.assertIs(svc.llm_runtime, llm)
        self.assertIs(svc.pending_actions, self.store)
        self.assertIs(svc.pending_lock, self.lock)
        self.assertIs(svc.tool_gateway, gateway)
        self.assertIs(svc.tool_registry, registry)

    def test_unknown_request_raises_not_found(self) -> None:
        with self.assertRaises(NotFoundError):
            self._service().ui_action_result(
                AgentUiActionResultRequest(request_id="ghost", success=True)
            )

    def test_request_without_ui_ack_is_rejected(self) -> None:
        self.store["r-1"] = _ui_action(ui_ack=False)
        with self.assertRaises(NotFoundError):
            self._service().ui_action_result(
                AgentUiActionResultRequest(request_id="r-1", success=True)
            )

    def test_successful_finalise_without_messages_reports_completed(self) -> None:
        self.store["r-1"] = _ui_action(messages=[])
        out = self._service().ui_action_result(
            AgentUiActionResultRequest(request_id="r-1", success=True, result={"detail": "ok"})
        )
        self.assertEqual(out["status"], "completed")
        self.assertEqual(out["session_id"], "sess-2")
        self.assertEqual(out["prior_step_count"], 1)
        final = out["steps"][-1]
        self.assertEqual(final["type"], "final_answer")
        self.assertIn(VALID_ACTION, final["content"])

    def test_failed_finalise_reports_failed(self) -> None:
        self.store["r-1"] = _ui_action()
        out = self._service().ui_action_result(
            AgentUiActionResultRequest(request_id="r-1", success=False)
        )
        self.assertEqual(out["status"], "failed")
        self.assertIn("Không thể thực thi", out["steps"][-1]["content"])

    def test_finalise_propagates_stored_retry_index(self) -> None:
        self.store["r-1"] = _ui_action(retry_message_index=7)
        out = self._service().ui_action_result(
            AgentUiActionResultRequest(request_id="r-1", success=True)
        )
        self.assertEqual(out["retry_message_index"], 7)

    def test_chaining_queues_the_next_desktop_action(self) -> None:
        self.store["r-1"] = _ui_action(
            next_actions=[{"action": "admin.logout"}, {"action": "admin.login"}],
        )
        out = self._service().ui_action_result(
            AgentUiActionResultRequest(request_id="r-1", success=True)
        )
        self.assertEqual(out["status"], "pending_ui_action")
        next_id = out["request_id"]
        self.assertNotEqual(next_id, "r-1")
        self.assertEqual(out["ui_action"]["action"], "admin.logout")
        # The remainder of the plan is carried onto the queued entry.
        self.assertEqual(self.store[next_id]["next_actions"], [{"action": "admin.login"}])
        self.assertEqual(self.store[next_id]["iteration"], 2)
        self.assertTrue(self.store[next_id]["ui_ack"])
        self.assertEqual(out["steps"][-1]["type"], "tool_call")

    def test_chaining_rejects_a_non_dict_entry(self) -> None:
        self.store["r-1"] = _ui_action(next_actions=["admin.logout"])
        with self.assertRaises(UnprocessableRequestError):
            self._service().ui_action_result(
                AgentUiActionResultRequest(request_id="r-1", success=True)
            )

    def test_chaining_rejects_an_unsupported_action(self) -> None:
        self.store["r-1"] = _ui_action(next_actions=[{"action": "not.a.real.action"}])
        with self.assertRaises(UnprocessableRequestError):
            self._service().ui_action_result(
                AgentUiActionResultRequest(request_id="r-1", success=True)
            )

    def test_failed_result_does_not_chain(self) -> None:
        self.store["r-1"] = _ui_action(next_actions=[{"action": "admin.logout"}])
        out = self._service().ui_action_result(
            AgentUiActionResultRequest(request_id="r-1", success=False)
        )
        # A failure ends the workflow instead of queueing the next step.
        self.assertEqual(out["status"], "failed")
        self.assertEqual(len(self.store), 0)

    def test_resume_fork_hands_the_runner_its_ui_context(self) -> None:
        self.store["r-1"] = _ui_action(messages=[{"role": "system", "content": "be brief"}])
        with patch.object(agents_context, "run_langgraph_agent", return_value={"status": "completed"}) as runner:
            with patch("ai_assistant.agents.ui_action_service.use_langgraph_agent", return_value=True):
                self._service().ui_action_result(
                    AgentUiActionResultRequest(request_id="r-1", success=True)
                )
        kwargs = runner.call_args.kwargs
        self.assertEqual(kwargs["system_prompt"], "be brief")
        self.assertTrue(kwargs["resume_with_reflection"])
        self.assertIs(kwargs["validate_tool_call_fn"], agents_context.RESUME_VALIDATE_TOOL_CALL)
        self.assertIs(kwargs["generate_action_id_fn"], agents_context.RESUME_ACTION_ID)
        # The desktop ack is injected into the resumed history for the graph.
        self.assertEqual(kwargs["initial_messages"][-2]["role"], "assistant")
        self.assertEqual(kwargs["initial_messages"][-1]["role"], "user")

    def test_finalise_without_langgraph_availability_reports_completed(self) -> None:
        """Even with the flag on, a missing graph class falls back to plain text."""
        from ai_assistant.agents.context import AgentRunContext

        self.store["r-1"] = _ui_action(messages=[{"role": "user", "content": "hi"}])
        svc = self._service()
        # Swap in a context whose graph class is absent so the resume branch
        # is skipped even though the settings flag is on.
        svc._ctx = AgentRunContext(
            llm_runtime=svc.llm_runtime, pending_actions=self.store,
            pending_lock=self.lock, local_agent_graph=None,
        )
        with patch("ai_assistant.agents.ui_action_service.use_langgraph_agent", return_value=True):
            out = svc.ui_action_result(AgentUiActionResultRequest(request_id="r-1", success=True))
        self.assertEqual(out["status"], "completed")


if __name__ == "__main__":
    unittest.main()
