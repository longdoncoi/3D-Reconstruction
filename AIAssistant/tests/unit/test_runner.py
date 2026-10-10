"""Unit tests for the LangGraph agent runner (agents/runner.py).

Everything the runner needs is injected as parameters, so the graph itself is
faked and no LangGraph or LLM runtime is required (offline-safe).
"""
from __future__ import annotations

import threading
import time
import unittest
from enum import Enum
from types import SimpleNamespace
from unittest import mock

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


def _base_kwargs(**overrides):
    kwargs = dict(
        system_prompt="sys",
        task="do the thing",
        session_id="session-1",
        temperature=0.3,
        language="vi",
        request_started=time.monotonic(),
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
            execute_approved=lambda t, p: {"success": True, "approved": True},
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
        validate_tool_call_fn=lambda name, params, models: (params, None),
        langsmith_trace_fn=lambda *a, **k: _LSContext(),
        span_fn=lambda *a, **k: _Span(),
        specialist_instruction_fn=lambda d: "instruction",
        is_coding_task_fn=lambda t: True,
        generate_action_id_fn=lambda: "id-123",
        save_pending_fn=lambda: None,
        LocalAgentGraph=_FakeGraph,
        Specialist=_Specialist,
    )
    kwargs.update(overrides)
    return kwargs


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
            run_langgraph_agent(**_base_kwargs(LocalAgentGraph=None))

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
        self.assertEqual(kwargs["pending_actions"]["rq-9"]["tool"], "application_action")

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
        self.assertEqual(kwargs["pending_actions"]["id-123"]["tool"], "run_command")
        self.assertEqual(result["steps"][0]["type"], "pending_approval")

    def test_graph_exception_propagates_and_trace_closed(self):
        _FakeGraph.run_error = RuntimeError("boom")
        ls = _LSContext()

        def langsmith_trace_fn(*a, **k):
            return ls

        with self.assertRaises(RuntimeError):
            self._run(langsmith_trace_fn=langsmith_trace_fn)
        self.assertEqual(len(ls.exited), 1)
        self.assertIs(ls.exited[0][0], RuntimeError)


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
            llm=SimpleNamespace(create_chat_completion=create_chat_completion),
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

    @mock.patch("ai_assistant.agents.runner.validate_action_params")
    def test_execute_application_action_canonicalises(self, validate):
        validate.return_value = ({"action": "viewer.load_2d"}, None)
        run_langgraph_agent(**_base_kwargs())
        execute = _FakeGraph._last.kwargs["execute"]
        res = execute("application_action", {"action": "viewer.load_2d"})
        self.assertEqual(res["params"]["action"], "viewer.load_2d")
        self.assertEqual(res["params"]["request_id"], "id-123")

    @mock.patch("ai_assistant.agents.runner.validate_action_params")
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
            A2ARouter=_A2ARouter,
        )
        run_langgraph_agent(**kwargs)
        execute = _FakeGraph._last.kwargs["execute"]
        res = execute("run_command", {"command": "ls"})
        self.assertEqual(res["source"], "remote")
        self.assertEqual(routed[0][1], "do the thing")
        self.assertEqual(routed[0][2]["idempotency_key"], "ik-1")

    def test_supervisor_route_defaults_and_helpers(self):
        run_langgraph_agent(**_base_kwargs())
        g = _FakeGraph._last
        self.assertEqual(g.run_kwargs["supervisor_route"], "supervisor")
        k = g.kwargs
        self.assertTrue(k["needs_approval"]("op_file"))
        self.assertFalse(k["needs_approval"]("run_command"))
        self.assertFalse(k["cancel_checker"]())

        # plan/reflect lambdas go through structured completion (openai branch)
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
        self.assertEqual(sel["instruction"], "instruction")
        self.assertEqual(k["select_specialist"]("_validation_error", {}), {})

        ver = k["verify_result"]("run_command", {}, {"success": True})
        self.assertEqual(ver["reason"], "verified")
        self.assertEqual(k["verify_result"]("_validation_error", {}, {"error": "e"})["passed"], False)

        ref = k["reflect_result"]("run_command", {}, {"success": True}, {"passed": True})
        self.assertEqual(ref["reason"], "reflected")

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
