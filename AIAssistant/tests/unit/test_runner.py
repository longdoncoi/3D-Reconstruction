"""Unit tests for the LangGraph agent runner (agents/runner.py).

Everything the runner needs is injected as parameters, so the graph itself is
faked and no LangGraph or LLM runtime is required (offline-safe).
"""
from __future__ import annotations

import json
import threading
import time
import unittest
from enum import Enum
from types import SimpleNamespace
from unittest import mock

from ai_assistant.agents.deps import RunDeps
from ai_assistant.agents.runner import run_langgraph_agent
from ai_assistant.domain.errors import ServiceUnavailableError


class _Specialist(Enum):
    SUPERVISOR = "supervisor"
    CODE = "code"


def _delegation(remote_endpoint=None):
    return SimpleNamespace(
        specialist=_Specialist.CODE,
        remote_endpoint=remote_endpoint,
        idempotency_key="ik-1",
    )


class _LSContext:
    """Fake langsmith trace context manager (dict-backed __enter__)."""

    def __init__(self):
        self._ctx = {}
        self.exited = []

    def __enter__(self):
        return self._ctx

    def __exit__(self, exc_type, exc, tb):
        self.exited.append((exc_type, exc))
        return False


class _Span:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeGraph:
    state = {"steps": [], "iteration": 0, "pending_tool": None, "cancelled": False}
    run_error = None
    _last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeGraph._last = self

    def run(self, *args, **kwargs):
        self.run_args = args
        self.run_kwargs = kwargs
        if self.run_error is not None:
            raise self.run_error
        return self.state


def _tool_registry(handler=lambda t, p: {"success": True}):
    return SimpleNamespace(
        get_openai_tools=lambda: [{"type": "function", "function": {"name": "x"}}],
        grammar="{grammar}",
        models={},
        get_all=lambda: [SimpleNamespace(name="op_file", requires_approval=True)],
        get=lambda name: (
            SimpleNamespace(name=name, requires_approval=(name == "op_file"), handler=handler)
            if name in ("run_command", "application_action")
            else None
        ),
    )


def _deps_kwargs(**overrides):
    """Wiring that lives in the frozen ``RunDeps`` bundle (ADR 0013)."""
    kwargs = dict(
        llm_runtime=None,
        backend_mode_fn=lambda: "openai",
        openai_compatible_fn=lambda messages, **kw: {
            "choices": [{"message": {"content": '{"passed": true, "decision": "continue", "reason": "ok"}'}}],
            "usage": {},
        },
        record_token_usage_fn=lambda i, o: None,
        tool_registry=_tool_registry(),
        tool_gateway=SimpleNamespace(
            execute=lambda t, p: {"success": True, "tool": t, "params": p},
            execute_approved=lambda t, p, tok="": {"success": True, "approved": True},
            preflight=lambda t, p, tok="": {"success": True, "tool": t, "preflight": True},
        ),
        pending_actions={},
        pending_lock=threading.Lock(),
        task_coordinator=SimpleNamespace(
            is_cancelled=lambda sid: False,
            update=lambda sid, **kw: None,
            finish=lambda sid, **kw: None,
        ),
        delegate_fn=lambda task, session_id, tool_name, params, prefer_code: _delegation(),
        authorise_delegation_fn=lambda d, needs_approval: (True, ""),
        audit_agent_fn=lambda kind, delegation, **kw: None,
        verify_result_fn=lambda d, result: {"passed": True, "reason": "verified"},
        reflect_result_fn=lambda d, result, verification: {"passed": True, "reason": "reflected"},
        record_tool_fn=lambda t, ok, duration: None,
        record_schema_error_fn=lambda t: None,
        langsmith_trace_fn=lambda *a, **k: _LSContext(),
        span_fn=lambda *a, **k: _Span(),
        is_coding_task_fn=lambda t: True,
        save_pending_fn=lambda: None,
        local_agent_graph=_FakeGraph,
        a2a_router=None,
        checkpointer=None,
    )
    kwargs.update(overrides)
    return kwargs


def _base_kwargs(**overrides):
    """Per-invocation kwargs for ``run_langgraph_agent`` plus its deps bundle.

    Dependency-only overrides (anything in ``RunDeps``) are routed into the
    bundle; the rest stay as per-invocation keyword arguments.
    """
    per_invocation = dict(
        system_prompt="sys",
        task="do the thing",
        session_id="session-1",
        temperature=0.3,
        language="vi",
        request_started=time.monotonic(),
        owner="tenant-a",
        validate_tool_call_fn=lambda name, params, models: (params, None),
        generate_action_id_fn=lambda: "id-123",
    )
    dep_keys = set(_deps_kwargs())
    dep_overrides = {k: v for k, v in overrides.items() if k in dep_keys}
    run_overrides = {k: v for k, v in overrides.items() if k not in dep_keys}
    per_invocation.update(run_overrides)
    per_invocation["deps"] = RunDeps(**_deps_kwargs(**dep_overrides))
    return per_invocation


class RunLanggraphAgentTests(unittest.TestCase):
    def setUp(self):
        _FakeGraph.state = {"steps": [], "iteration": 0, "pending_tool": None, "cancelled": False}
        _FakeGraph.run_error = None
        _FakeGraph._last = None
        super().setUp()

    def _run(self, **overrides):
        kwargs = _base_kwargs(**overrides)
        return run_langgraph_agent(**kwargs), kwargs

    def test_requires_langgraph(self):
        with self.assertRaises(ServiceUnavailableError):
            run_langgraph_agent(**_base_kwargs(local_agent_graph=None))

    def test_completed_run(self):
        _FakeGraph.state = {
            "steps": [{"type": "final_answer", "content": "done"}],
            "iteration": 2,
            "pending_tool": None,
            "cancelled": False,
            "messages": [{"role": "user", "content": "hi"}],
        }
        result, _ = self._run()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(result["steps"], [{"type": "final_answer", "content": "done"}])

    def test_completed_without_final_answer_appends_sentinel(self):
        _FakeGraph.state = {
            "steps": [{"type": "tool", "content": "z"}],
            "iteration": 1,
            "pending_tool": None,
            "cancelled": False,
            "messages": [],
        }
        result, _ = self._run()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps"][-1]["type"], "final_answer")
        self.assertIn("chưa có kết luận", result["steps"][-1]["content"])

    def test_prior_plan_reaches_the_graph(self):
        """The procedural-memory seed is a per-invocation argument the runner
        must forward to ``graph.run`` unchanged — or not at all (ADR 0043 §7)."""
        self._run(prior_plan=["Buoc mot", "Buoc hai"])
        self.assertEqual(_FakeGraph._last.run_kwargs["prior_plan"],
                         ["Buoc mot", "Buoc hai"])
        self._run()
        self.assertIsNone(_FakeGraph._last.run_kwargs["prior_plan"])

    def test_cancelled_run(self):
        _FakeGraph.state = {
            "steps": [],
            "iteration": 3,
            "pending_tool": None,
            "cancelled": True,
            "messages": [],
        }
        coordinator = SimpleNamespace(
            is_cancelled=lambda sid: True,
            update=lambda sid, **kw: None,
            finish=mock.Mock(),
        )
        result, _ = self._run(task_coordinator=coordinator)
        self.assertEqual(result["status"], "cancelled")
        coordinator.finish.assert_called_once()
        self.assertEqual(coordinator.finish.call_args.kwargs["success"], False)

    def test_pending_ui_action_request(self):
        _FakeGraph.state = {
            "steps": [],
            "iteration": 1,
            "pending_tool": {
                "tool": "application_action",
                "params": {"request_id": "rq-9", "action": "viewer.load_2d", "description": "load"},
                "ui_ack": True,
            },
            "cancelled": False,
            "messages": [{"role": "user", "content": "m"}],
        }
        result, kwargs = self._run()
        self.assertEqual(result["status"], "pending_ui_action")
        self.assertEqual(result["ui_action"]["request_id"], "rq-9")
        self.assertEqual(kwargs["deps"].pending_actions["rq-9"]["tool"], "application_action")
        # The parked ack resumes in another request, so the run's memory
        # namespace must travel with the entry (ADR 0043).
        self.assertEqual(kwargs["deps"].pending_actions["rq-9"]["owner"], "tenant-a")

    def test_pending_approval_request(self):
        _FakeGraph.state = {
            "steps": [],
            "iteration": 1,
            "pending_tool": {
                "tool": "run_command",
                "params": {"command": "git commit", "description": "commit"},
                "ui_ack": False,
                "approval_scope": "repo",
                "approval_preview": "diff",
            },
            "cancelled": False,
            "messages": [{"role": "user", "content": "m"}],
        }
        result, kwargs = self._run()
        self.assertEqual(result["status"], "pending_approval")
        self.assertEqual(result["action_id"], "id-123")
        self.assertEqual(kwargs["deps"].pending_actions["id-123"]["tool"], "run_command")
        self.assertEqual(kwargs["deps"].pending_actions["id-123"]["owner"], "tenant-a")
        self.assertEqual(result["steps"][0]["type"], "pending_approval")

    def test_a_resumed_run_still_mints_an_id_for_a_newly_parked_approval(self):
        """The resume path installs a no-op id generator, but the action parked
        during that run is new. An empty action_id is refused at the approval
        boundary, so the client could never approve what this returns."""
        _FakeGraph.state = {
            "steps": [],
            "iteration": 1,
            "pending_tool": {
                "tool": "run_command",
                "params": {"command": "git commit", "description": "commit"},
                "ui_ack": False,
                "approval_scope": "repo",
                "approval_preview": "diff",
            },
            "cancelled": False,
            "messages": [{"role": "user", "content": "m"}],
        }
        result, kwargs = self._run(generate_action_id_fn=lambda: "")
        self.assertEqual(result["status"], "pending_approval")
        self.assertTrue(result["action_id"])
        self.assertEqual(
            kwargs["deps"].pending_actions[result["action_id"]]["tool"], "run_command",
        )
        self.assertNotIn("", kwargs["deps"].pending_actions)
        self.assertEqual(result["steps"][0]["action_id"], result["action_id"])

    def test_a_parked_entry_snapshots_the_live_state_lists(self):
        """The graph keeps appending to ``steps``/``messages`` after a run is
        parked, and the store re-serialises whatever it holds on the next
        save — an aliased entry changed on disk after the fact (review
        finding 9)."""
        _FakeGraph.state = {
            "steps": [{"type": "plan", "iteration": 0}],
            "iteration": 1,
            "pending_tool": {
                "tool": "run_command",
                "params": {"command": "git commit", "description": "commit"},
                "ui_ack": False,
            },
            "cancelled": False,
            "messages": [{"role": "user", "content": "m"}],
        }
        result, kwargs = self._run()
        entry = kwargs["deps"].pending_actions[result["action_id"]]
        steps_at_park = list(entry["steps"])
        messages_at_park = list(entry["messages"])

        _FakeGraph.state["steps"].append({"type": "final_answer", "content": "late"})
        _FakeGraph.state["messages"].append({"role": "assistant", "content": "late"})

        self.assertEqual(entry["steps"], steps_at_park)
        self.assertEqual(entry["messages"], messages_at_park)

    def test_graph_exception_propagates_and_trace_closed(self):
        _FakeGraph.run_error = RuntimeError("boom")
        ls = _LSContext()

        def langsmith_trace_fn(*a, **k):
            return ls

        with self.assertRaises(RuntimeError):
            self._run(langsmith_trace_fn=langsmith_trace_fn)
        self.assertEqual(len(ls.exited), 1)
        self.assertIs(ls.exited[0][0], RuntimeError)

    def test_an_aborted_run_still_finishes_the_task_record(self):
        """The abort branch used to re-raise without ever reaching
        ``_finish_task``, so a crashed run left the coordinator's record
        parked in ``running`` forever (review finding 6)."""
        _FakeGraph.run_error = RuntimeError("boom")
        finished = []

        coordinator = SimpleNamespace(
            is_cancelled=lambda sid: False,
            update=lambda sid, **kw: None,
            finish=lambda sid, **kw: finished.append((sid, kw)),
        )
        with self.assertRaises(RuntimeError):
            self._run(task_coordinator=coordinator)
        self.assertEqual(
            finished, [("session-1", {"success": False, "error": "boom"})]
        )


class RunnerCallbackTests(unittest.TestCase):
    """Exercise the closures the runner registers on the graph."""

    def setUp(self):
        _FakeGraph.state = {
            "steps": [{"type": "final_answer", "content": "done"}],
            "iteration": 0,
            "pending_tool": None,
            "cancelled": False,
            "messages": [],
        }
        _FakeGraph.run_error = None
        _FakeGraph._last = None
        super().setUp()

    def test_complete_context_too_long(self):
        kwargs = _base_kwargs()
        run_langgraph_agent(**kwargs)
        complete = _FakeGraph._last.kwargs["complete"]
        long_msgs = [{"role": "user", "content": "x" * 100000}]
        self.assertEqual(complete(long_msgs, 0.3), "Context quá dài, dừng Agent.")

    @mock.patch(
        "ai_assistant.agents.completion._get_precompiled_grammar", return_value=None
    )
    @mock.patch(
        "ai_assistant.agents.completion.load_agent_runtime_settings",
        return_value=SimpleNamespace(native_tool_calls=False),
    )
    def test_complete_calls_llm_local_grammar(self, _settings_patch, _grammar_patch):
        recorded = []
        calls = []

        def create_chat_completion(**kw):
            calls.append(kw)
            return {
                "choices": [{"message": {"content": '{"kind":"final","content":"hi"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

        llm_runtime = SimpleNamespace(
            llm_lock=threading.Lock(),
            # The loop reaches the backend through the protocol ``generate``.
            llm=SimpleNamespace(generate=create_chat_completion),
        )
        kwargs = _base_kwargs(
            backend_mode_fn=lambda: "llama_cpp",
            llm_runtime=llm_runtime,
            record_token_usage_fn=lambda i, o: recorded.append((i, o)),
        )
        run_langgraph_agent(**kwargs)
        complete = _FakeGraph._last.kwargs["complete"]
        answer = complete([{"role": "user", "content": "hello"}], 0.7)
        self.assertEqual(answer, "hi")
        self.assertEqual(recorded, [(10, 5)])
        self.assertIn("messages", calls[0])

    def test_parse_valid_and_invalid(self):
        run_langgraph_agent(**_base_kwargs())
        parse = _FakeGraph._last.kwargs["parse"]
        name, params = parse('{"kind":"tool","tool":"run_command","params":{"command":"ls"}}')
        self.assertEqual((name, params), ("run_command", {"command": "ls"}))

        schema_errors = []
        kwargs = _base_kwargs(
            validate_tool_call_fn=lambda n, p, m: (p, "bad schema"),
            record_schema_error_fn=lambda t: schema_errors.append(t),
        )
        run_langgraph_agent(**kwargs)
        parse = _FakeGraph._last.kwargs["parse"]
        name, params = parse('{"kind":"tool","tool":"run_command","params":{"command":"ls"}}')
        self.assertEqual(name, "_validation_error")
        self.assertEqual(params["tool"], "run_command")
        self.assertEqual(schema_errors, ["run_command"])

    def test_execute_paths(self):
        # Validation-error tool
        run_langgraph_agent(**_base_kwargs())
        execute = _FakeGraph._last.kwargs["execute"]
        err = execute("_validation_error", {"tool": "t", "error": "invalid"})
        self.assertIn("Lỗi xác thực", err["error"])

        # Unknown tool
        err = execute("no_such_tool", {})
        self.assertIn("không tồn tại", err["error"])

        # Denied by supervisor policy
        def deny(d, needs_approval):
            return (False, "policy")

        run_langgraph_agent(**_base_kwargs(authorise_delegation_fn=deny))
        execute = _FakeGraph._last.kwargs["execute"]
        res = execute("run_command", {"command": "ls"})
        self.assertEqual(res["error"], "policy")

    def test_execute_unconfigured_gateway(self):
        run_langgraph_agent(**_base_kwargs(tool_gateway=None))
        execute = _FakeGraph._last.kwargs["execute"]
        res = execute("run_command", {"command": "ls"})
        self.assertEqual(res["error_code"], "runtime_unconfigured")

    def test_execute_approved_vs_normal_gateway(self):
        used = []

        def make_gateway():
            return SimpleNamespace(
                execute=lambda t, p: used.append("normal") or {"ok": True},
                execute_approved=lambda t, p, token="": used.append("approved") or {"ok": True},
            )

        run_langgraph_agent(**_base_kwargs(tool_gateway=make_gateway(), approval_granted=False))
        execute = _FakeGraph._last.kwargs["execute"]
        execute("run_command", {})
        # granted path: only a real grant token selects execute_approved
        run_langgraph_agent(**_base_kwargs(
            tool_gateway=make_gateway(), approval_granted=True, approval_token="grant-1",
        ))
        execute = _FakeGraph._last.kwargs["execute"]
        execute("run_command", {})
        self.assertEqual(used, ["normal", "approved"])

    @mock.patch("ai_assistant.agents.graph_wiring.validate_action_params")
    def test_execute_application_action_canonicalises(self, validate):
        validate.return_value = ({"action": "viewer.load_2d"}, None)
        run_langgraph_agent(**_base_kwargs())
        execute = _FakeGraph._last.kwargs["execute"]
        res = execute("application_action", {"action": "viewer.load_2d"})
        self.assertEqual(res["params"]["action"], "viewer.load_2d")
        self.assertEqual(res["params"]["request_id"], "id-123")

    @mock.patch("ai_assistant.agents.graph_wiring.validate_action_params")
    def test_execute_application_action_invalid_params(self, validate):
        validate.return_value = (None, "invalid action")
        run_langgraph_agent(**_base_kwargs())
        execute = _FakeGraph._last.kwargs["execute"]
        res = execute("application_action", {"action": "bogus"})
        self.assertEqual(res["error"], "invalid action")

    def test_execute_remote_a2a_delegation(self):
        routed = []

        class _A2ARouter:
            def route(self, specialist, task, payload):
                routed.append((specialist, task, payload))
                return {"source": "remote", "result": "ok"}

        kwargs = _base_kwargs(
            delegate_fn=lambda task, session_id, tool_name, params, prefer_code: _delegation(
                remote_endpoint="http://remote"
            ),
            a2a_router=_A2ARouter(),
        )
        run_langgraph_agent(**kwargs)
        execute = _FakeGraph._last.kwargs["execute"]
        res = execute("run_command", {"command": "ls"})
        self.assertEqual(res["source"], "remote")
        self.assertEqual(routed[0][1], "do the thing")
        self.assertEqual(routed[0][2]["idempotency_key"], "ik-1")

    def test_remote_dispatch_clears_the_gateway_gate_first(self):
        # The remote leg leaves the process, so it must clear the same policy
        # gate — validation, scope, classification, approval grant — or not go.
        gates: list[tuple] = []
        routed = []

        class _A2ARouter:
            def route(self, specialist, task, payload):
                routed.append(payload)
                return {"source": "remote"}

        def _preflight(tool, params, token=""):
            gates.append((tool, token))
            return {"success": False, "error_code": "forbidden", "error": "no scope"}

        kwargs = _base_kwargs(
            delegate_fn=lambda task, session_id, tool_name, params, prefer_code: _delegation(
                remote_endpoint="http://remote"
            ),
            a2a_router=_A2ARouter(),
            tool_gateway=SimpleNamespace(preflight=_preflight),
        )
        run_langgraph_agent(**kwargs)
        execute = _FakeGraph._last.kwargs["execute"]

        refused = execute("run_command", {"command": "ls"})
        self.assertEqual(gates, [("run_command", "")])
        self.assertEqual(routed, [])
        self.assertEqual(refused["error_code"], "forbidden")

    def test_remote_dispatch_never_leaves_an_unbootstrapped_platform(self):
        routed = []

        class _A2ARouter:
            def route(self, specialist, task, payload):
                routed.append(payload)
                return {"source": "remote"}

        kwargs = _base_kwargs(
            delegate_fn=lambda task, session_id, tool_name, params, prefer_code: _delegation(
                remote_endpoint="http://remote"
            ),
            a2a_router=_A2ARouter(),
            tool_gateway=None,
        )
        run_langgraph_agent(**kwargs)
        execute = _FakeGraph._last.kwargs["execute"]

        result = execute("run_command", {"command": "ls"})
        self.assertEqual(routed, [])
        self.assertEqual(result["error_code"], "runtime_unconfigured")

    def test_remote_dispatch_refuses_to_put_a_credential_on_the_wire(self) -> None:
        # ``admin.login`` carries username/password; masking them would hand the
        # remote peer a falsified payload to run, so the dispatch is refused
        # before the gateway gate is even consulted (the grant stays unspent).
        routed = []
        gates = []

        class _A2ARouter:
            def route(self, specialist, task, payload):
                routed.append(payload)
                return {"source": "remote"}

        kwargs = _base_kwargs(
            delegate_fn=lambda task, session_id, tool_name, params, prefer_code: _delegation(
                remote_endpoint="http://remote"
            ),
            a2a_router=_A2ARouter(),
            tool_gateway=SimpleNamespace(
                preflight=lambda t, p, tok="": gates.append(t) or {"success": True},
            ),
        )
        run_langgraph_agent(**kwargs)
        execute = _FakeGraph._last.kwargs["execute"]

        result = execute("application_action",
                         {"action": "admin.login", "username": "alice", "password": "s3cret"})
        self.assertEqual(routed, [])
        self.assertEqual(gates, [])
        self.assertEqual(result["error_code"], "data_policy_denied")
        self.assertNotIn("s3cret", json.dumps(result))

    def test_graph_helpers_and_structured_completions(self):
        run_langgraph_agent(**_base_kwargs())
        g = _FakeGraph._last
        k = g.kwargs
        self.assertTrue(k["needs_approval"]("op_file"))
        self.assertFalse(k["needs_approval"]("run_command"))
        self.assertFalse(k["cancel_checker"]())

        # plan/reflect lambdas go through structured completion (openai branch).
        # The decoder now runs the ADR 0018 gate first, so exercising that
        # branch means opting into cloud routing — the refusal itself is
        # pinned by HybridPolicyGateTests.
        with mock.patch.dict("os.environ", {"AGENT_HYBRID_POLICY": "cloud_allowed"}):
            plan = k["plan_complete"]([{"role": "user", "content": "p"}], 0.4)
            self.assertIn("passed", plan)
            critic = k["reflect_complete"]([{"role": "user", "content": "c"}], 0.4)
            self.assertIn("passed", critic)
            pr = k["plan_reflect_complete"]([{"role": "user", "content": "pr"}], 0.4)
            self.assertIn("passed", pr)

    def test_specialist_delegation_helpers(self):
        run_langgraph_agent(**_base_kwargs())
        k = _FakeGraph._last.kwargs
        sel = k["select_specialist"]("read_file", {"path": "x"})
        self.assertEqual(sel["specialist"], str(_Specialist.CODE))
        self.assertNotIn("instruction", sel)
        self.assertEqual(k["select_specialist"]("_validation_error", {}), {})

        ver = k["verify_result"]("run_command", {}, {"success": True})
        self.assertEqual(ver["reason"], "verified")
        self.assertEqual(k["verify_result"]("_validation_error", {}, {"error": "e"})["passed"], False)

        ref = k["reflect_result"]("run_command", {}, {"success": True}, {"passed": True})
        self.assertEqual(ref["reason"], "reflected")

    def test_one_delegation_is_shared_across_the_tool_callbacks(self):
        """Dispatch, execution, verification and reflection classify once.

        The four callbacks used to each re-run the supervisor's payload dump,
        SHA-256 fingerprint, remote lookup and coding heuristic for the very
        same (tool, params) pair (review finding 14); one run must now do one
        classification per distinct call.
        """
        delegate = mock.Mock(return_value=_delegation())
        run_langgraph_agent(**_base_kwargs(delegate_fn=delegate))
        k = _FakeGraph._last.kwargs
        k["select_specialist"]("read_file", {"path": "x"})
        k["verify_result"]("read_file", {"path": "x"}, {"content": "# hi"})
        k["reflect_result"]("read_file", {"path": "x"}, {"content": "# hi"},
                            {"passed": True})
        k["execute"]("read_file", {"path": "x"})
        self.assertEqual(delegate.call_count, 1)
        # A different call is a different classification.
        k["select_specialist"]("run_command", {"command": "pytest"})
        self.assertEqual(delegate.call_count, 2)

    def test_pending_store_save_via_fallback(self):
        store = mock.MagicMock()
        _FakeGraph.state = {
            "steps": [],
            "iteration": 1,
            "pending_tool": {
                "tool": "run_command",
                "params": {"command": "git commit"},
                "ui_ack": False,
                "approval_scope": "",
                "approval_preview": None,
            },
            "cancelled": False,
            "messages": [{"role": "user", "content": "m"}],
        }
        result = run_langgraph_agent(
            **_base_kwargs(
                pending_actions=store,
                save_pending_fn=None,
            )
        )
        self.assertEqual(result["status"], "pending_approval")
        store.save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
