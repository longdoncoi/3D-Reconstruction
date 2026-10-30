"""Guard-path coverage for the runner's invocation seams.

One line of ``run_langgraph_agent`` only runs for a specific caller: an
approval continuation carrying a token wires the single-use grant probe into
the graph. It is exercised here with the same fake-graph harness style as the
main runner tests, so no LangGraph or LLM runtime is required.
"""
from __future__ import annotations

import threading
import time
import unittest
from enum import Enum
from types import SimpleNamespace

from ai_assistant.agents.deps import RunDeps
from ai_assistant.agents.runner import run_langgraph_agent


class _Specialist(Enum):
    SUPERVISOR = "supervisor"
    CODE = "code"


class _LSContext:
    def __init__(self) -> None:
        self._ctx: dict = {}

    def __enter__(self):
        return self._ctx

    def __exit__(self, *args):
        return False


class _Span:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeGraph:
    """Minimal stand-in for the compiled LangGraph: already finished."""

    last: "_FakeGraph | None" = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        _FakeGraph.last = self

    def run(self, *args, **kwargs):
        return {"steps": [], "iteration": 0, "pending_tool": None, "cancelled": False}


def _delegation():
    return SimpleNamespace(
        specialist=_Specialist.CODE, remote_endpoint=None, idempotency_key="ik-1",
    )


def _deps(**overrides) -> RunDeps:
    kwargs = {
        "llm_runtime": None,
        "backend_mode_fn": lambda: "openai",
        "openai_compatible_fn": lambda messages, **kw: {
            "choices": [{"message": {"content": '{"kind": "final", "content": "done"}'}}],
            "usage": {},
        },
        "record_token_usage_fn": lambda i, o: None,
        "tool_registry": SimpleNamespace(
            get_openai_tools=lambda: [],
            grammar="{}",
            models={},
            get_all=lambda: [],
            get=lambda name: None,
        ),
        "pending_actions": {},
        "pending_lock": threading.Lock(),
        "task_coordinator": SimpleNamespace(
            is_cancelled=lambda sid: False,
            update=lambda sid, **kw: None,
            finish=lambda sid, **kw: None,
        ),
        "delegate_fn": lambda task, session_id, tool_name, params, prefer_code: _delegation(),
        "authorise_delegation_fn": lambda d, needs_approval: (True, ""),
        "audit_agent_fn": lambda kind, delegation, **kw: None,
        "verify_result_fn": lambda d, result: {"passed": True, "reason": "verified"},
        "reflect_result_fn": lambda d, result, verification: {"passed": True, "reason": "ok"},
        "record_tool_fn": lambda t, ok, duration: None,
        "record_schema_error_fn": lambda t: None,
        "langsmith_trace_fn": lambda *a, **k: _LSContext(),
        "span_fn": lambda *a, **k: _Span(),
        "is_coding_task_fn": lambda t: False,
        "save_pending_fn": lambda: None,
        "local_agent_graph": _FakeGraph,
        "tool_gateway": None,
        "a2a_router": None,
        "checkpointer": None,
    }
    kwargs.update(overrides)
    return RunDeps(**kwargs)


def _invoke(**overrides):
    kwargs = {
        "system_prompt": "sys",
        "task": "do the thing",
        "session_id": "session-1",
        "temperature": 0.3,
        "language": "vi",
        "request_started": time.monotonic(),
        "owner": "tenant-a",
        "deps": _deps(),
    }
    dep_overrides = {k: v for k, v in overrides.items() if k in RunDeps.__dataclass_fields__}
    run_overrides = {k: v for k, v in overrides.items() if k not in RunDeps.__dataclass_fields__}
    kwargs.update(run_overrides)
    if dep_overrides:
        kwargs["deps"] = _deps(**dep_overrides)
    return run_langgraph_agent(**kwargs)


class ApprovalTokenTests(unittest.TestCase):
    def test_a_token_wires_the_single_use_grant_probe(self):
        """An approval continuation hands the graph a live grant probe so the
        reason node re-asks as soon as the grant is spent."""
        gateway = SimpleNamespace(
            execute=lambda t, p: {"success": True},
            approval_covers=lambda t, p, tok: tok == "tok-1",
        )
        result = _invoke(approval_token="tok-1", tool_gateway=gateway)
        self.assertEqual(result["status"], "completed")
        # The probe handed to the graph really consults the gateway grant.
        covers = _FakeGraph.last.kwargs["approval_covers"]
        self.assertIsNotNone(covers)
        self.assertTrue(covers("read_file", {"path": "a"}))

    def test_a_token_without_a_covering_gateway_is_ignored(self):
        result = _invoke(approval_token="tok-1", tool_gateway=SimpleNamespace())
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(_FakeGraph.last.kwargs["approval_covers"])


if __name__ == "__main__":
    unittest.main()
