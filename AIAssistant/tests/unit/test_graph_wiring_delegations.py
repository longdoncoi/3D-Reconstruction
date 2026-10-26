"""Delegation memoisation on the run wiring (ADR 0044 §6).

``build_run_graph`` owns one supervisor-delegation memo per run. The parallel
tool batch precomputes a whole group's classifications in the dispatching
thread (``warm_delegations``) and then lets the branches read a warm cache, so
the cache is now reached concurrently and its read-modify-write needs a lock:
without it two branches of one group can both miss and pay twice for the same
classification.

The graph engine is faked (``test_runner``'s ``_FakeGraph``), so these tests
inspect the callbacks the wiring built without LangGraph or an LLM.
"""
from __future__ import annotations

import threading
import time
import unittest
from dataclasses import replace

from ai_assistant.agents.runner import run_langgraph_agent
from tests.unit.test_runner import _base_kwargs, _delegation, _FakeGraph


class GraphWiringDelegationTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeGraph._last = None

    def _graph_kwargs(self, *, parallel_max=4, **overrides):
        invocation = _base_kwargs(**overrides)
        invocation["deps"] = replace(invocation["deps"], parallel_max=parallel_max)
        run_langgraph_agent(**invocation)
        self.assertIsNotNone(_FakeGraph._last)
        return _FakeGraph._last.kwargs

    def test_the_batch_wiring_is_passed_to_the_graph(self) -> None:
        kwargs = self._graph_kwargs(parallel_max=6)
        self.assertEqual(kwargs["max_parallel"], 6)
        self.assertEqual(kwargs["requires_approval"], frozenset({"op_file"}))
        self.assertTrue(callable(kwargs["warm_delegations"]))

    def test_the_warm_up_classifies_each_distinct_call_once(self) -> None:
        classified: list[tuple[str, dict]] = []

        def _delegate(task, session_id, tool_name, params, prefer_code):
            classified.append((tool_name, params))
            return _delegation()

        warm = self._graph_kwargs(delegate_fn=_delegate)["warm_delegations"]
        warm([
            ("rag_search", {"query": "b"}),
            ("rag_search", {"query": "a"}),
            ("rag_search", {"query": "a"}),
        ])

        self.assertEqual(classified, [
            ("rag_search", {"query": "b"}),
            ("rag_search", {"query": "a"}),
        ])

    def test_two_branches_classify_a_shared_call_once(self) -> None:
        """The lock must cover the *lookup*, not just the write: a miss in two
        branches has to be resolved once by the first winner."""
        start = threading.Barrier(2, timeout=10)
        classified: list[str] = []

        def _delegate(task, session_id, tool_name, params, prefer_code):
            classified.append(tool_name)
            time.sleep(0.05)
            return _delegation()

        warm = self._graph_kwargs(delegate_fn=_delegate)["warm_delegations"]

        def _branch() -> None:
            start.wait()
            warm([("rag_search", {"query": "shared"})])

        threads = [threading.Thread(target=_branch) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())

        self.assertEqual(classified, ["rag_search"])


if __name__ == "__main__":
    unittest.main()
