"""Guard-path coverage for the orchestration graph wrapper.

``LocalAgentGraph`` has exactly two responsibilities that the node-level tests
never touch: streaming newly produced steps to the UI emitter, and restoring
``required_ui_actions`` from the LangGraph checkpoint when a run resumes. The
checkpoint read is explicitly best-effort — a corrupt or missing checkpoint
must degrade to "no pending UI actions" instead of killing the resumed run —
so both the success arm and the ``except`` arm are worth pinning.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from ai_assistant.orchestration.graph import LocalAgentGraph


def _graph(emit=None, checkpointer: Any | None = None) -> LocalAgentGraph:
    if checkpointer is not None:
        # LangGraph refuses an arbitrary object; a real in-memory saver keeps
        # the compile() call honest while still exercising the resume path.
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()
    return LocalAgentGraph(
        complete=lambda _messages, _temperature: "{}",
        parse=lambda _text: (None, None),
        execute=lambda _tool, _params: {},
        needs_approval=lambda _tool, _params: False,
        max_iterations=3,
        emit=emit,
        checkpointer=checkpointer,
    )


class _FakeCompiledGraph:
    """Stands in for ``builder.compile(...)`` so ``run`` can be driven alone."""

    def __init__(self, prior: Any = None, error: Exception | None = None) -> None:
        self._prior = prior
        self._error = error
        self.prior_reads = 0
        self.invoked_state: dict[str, Any] | None = None

    def get_state(self, _config) -> Any:
        self.prior_reads += 1
        if self._error is not None:
            raise self._error
        return self._prior

    def invoke(self, input_state, config=None) -> dict[str, Any]:
        self.invoked_state = dict(input_state)
        return {"done": True, "steps": []}


class StepStreamingTests(unittest.TestCase):
    """Each node may only tell the UI about steps it has not sent yet."""

    def test_steps_are_streamed_once_and_only_once(self):
        seen: list[dict[str, Any]] = []
        graph = _graph(emit=seen.append)
        pool = [
            {"type": "plan", "content": "first"},
            {"type": "plan", "content": "second"},
            {"type": "plan", "content": "third"},
        ]
        node = graph._traced("plan", lambda _state: {"steps": pool[:2], "done": False})
        state = {"iteration": 0, "routing_plan": "supervisor", "steps": []}

        node(dict(state))
        self.assertEqual([s["content"] for s in seen], ["first", "second"])

        # A later turn re-sends the full list; only the delta is new.
        node(dict(state, steps=pool[:2]))
        self.assertEqual(len(seen), 2)

        node(dict(state, steps=pool[:2]))
        self.assertEqual(len(seen), 2)

    def test_a_node_returning_no_steps_is_silent(self):
        seen: list[dict[str, Any]] = []
        graph = _graph(emit=seen.append)
        node = graph._traced("reason", lambda _state: {"done": True})
        node({"iteration": 1, "routing_plan": "supervisor", "steps": []})
        self.assertEqual(seen, [])


class CheckpointRestoreTests(unittest.TestCase):
    """A resumed run re-reads the stored UI actions, but never depends on it."""

    def _run(self, graph: LocalAgentGraph, fake: _FakeCompiledGraph):
        graph._graph = fake
        return graph.run(
            [{"role": "user", "content": "resume"}],
            session_id="sess-1",
            temperature=0.2,
            resume_with_reflection=True,
        )

    def test_a_saved_checkpoint_restores_the_pending_ui_actions(self):
        fake = _FakeCompiledGraph(
            prior=SimpleNamespace(values={"required_ui_actions": [{"action": "login"}]}),
        )
        graph = _graph(checkpointer=object())
        self._run(graph, fake)
        self.assertEqual(fake.prior_reads, 1)
        self.assertEqual(fake.invoked_state["required_ui_actions"], [{"action": "login"}])

    def test_a_checkpoint_without_saved_ui_actions_yields_an_empty_list(self):
        fake = _FakeCompiledGraph(prior=SimpleNamespace(values={}))
        graph = _graph(checkpointer=object())
        self._run(graph, fake)
        self.assertEqual(fake.invoked_state["required_ui_actions"], [])

    def test_an_absent_checkpoint_yields_an_empty_list(self):
        fake = _FakeCompiledGraph(prior=None)
        graph = _graph(checkpointer=object())
        self._run(graph, fake)
        self.assertEqual(fake.invoked_state["required_ui_actions"], [])

    def test_an_unreadable_checkpoint_does_not_abort_the_resumed_run(self):
        """The resume must proceed with no pending actions, not raise."""
        fake = _FakeCompiledGraph(error=RuntimeError("checkpointer gone"))
        graph = _graph(checkpointer=object())
        with self.assertLogs("ai_assistant.orchestration.graph", level="WARNING") as logged:
            self._run(graph, fake)
        self.assertTrue(
            any("khôi phục required_ui_actions" in line for line in logged.output),
            logged.output,
        )
        self.assertEqual(fake.invoked_state["required_ui_actions"], [])

    def test_explicit_ui_actions_skip_the_checkpoint_read(self):
        fake = _FakeCompiledGraph()
        graph = _graph(checkpointer=object())
        graph._graph = fake
        graph.run(
            [{"role": "user", "content": "resume"}],
            session_id="sess-1",
            temperature=0.2,
            resume_with_reflection=True,
            required_ui_actions=[{"action": "confirm"}],
        )
        self.assertEqual(fake.prior_reads, 0)
        self.assertEqual(fake.invoked_state["required_ui_actions"], [{"action": "confirm"}])

    def test_a_stateless_run_never_touches_the_checkpointer(self):
        """``checkpointer=None`` is the stateless path used by tests."""
        fake = _FakeCompiledGraph()
        graph = _graph()
        graph._graph = fake
        graph.run(
            [{"role": "user", "content": "hi"}],
            session_id="sess-2",
            temperature=0.2,
        )
        self.assertEqual(fake.prior_reads, 0)
        self.assertEqual(fake.invoked_state["required_ui_actions"], [])


if __name__ == "__main__":
    unittest.main()
