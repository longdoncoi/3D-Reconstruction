"""Short-term memory buffer bounds and its wiring into the agent facade (ADR 0043).

Two things are pinned here: the buffer cannot grow without limit (per-message
window, per-map session cap, explicit forget/cleanup), and the facade uses it
exactly the way the ADR says — the client's shipped history is topped up from
the buffer, the run's turns are recorded back, and ``None`` keeps the original
"client ships everything" behaviour with no server-side state at all.
"""
from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from unittest.mock import patch

from ai_assistant.agents.models import AgentExecuteRequest
from ai_assistant.agents.service import AgentService
from ai_assistant.agents.short_term_memory import BoundedShortTermMemory
from ai_assistant.domain.memory import DEFAULT_MEMORY_OWNER
from ai_assistant.ports import ShortTermMemoryPort


class _RecordingBuffer(BoundedShortTermMemory):
    """A buffer that also records whether the facade called it."""

    def __init__(self, max_messages: int = 40, max_sessions: int = 256) -> None:
        super().__init__(max_messages=max_messages, max_sessions=max_sessions)
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def record(self, session_id: str, messages: Sequence[Mapping[str, str]]) -> None:
        turns = [dict(m) for m in messages]
        self.calls.append((session_id, turns))
        super().record(session_id, turns)


class BufferBoundTests(unittest.TestCase):
    def test_record_then_recent_round_trips_copies(self) -> None:
        buffer = BoundedShortTermMemory()
        buffer.record("s", [{"role": "user", "content": "hi"}])
        recent = buffer.recent("s")
        self.assertEqual(recent, [{"role": "user", "content": "hi"}])
        recent[0]["content"] = "mutated"
        self.assertEqual(buffer.recent("s")[0]["content"], "hi")

    def test_a_run_replaces_the_window_instead_of_appending_to_it(self) -> None:
        """Each run's messages already contain the merged history, so the
        buffer stays self-bounding by replacement."""
        buffer = BoundedShortTermMemory()
        buffer.record("s", [{"role": "user", "content": "old"}])
        buffer.record("s", [{"role": "user", "content": "new"}])
        self.assertEqual(buffer.recent("s"), [{"role": "user", "content": "new"}])

    def test_recording_nothing_keeps_the_context_it_exists_for(self) -> None:
        buffer = BoundedShortTermMemory()
        buffer.record("s", [{"role": "user", "content": "keep"}])
        buffer.record("s", [])
        self.assertEqual(buffer.recent("s"), [{"role": "user", "content": "keep"}])

    def test_the_window_is_capped_at_max_messages(self) -> None:
        buffer = BoundedShortTermMemory(max_messages=2)
        buffer.record("s", [{"role": "user", "content": str(i)} for i in range(5)])
        self.assertEqual([m["content"] for m in buffer.recent("s")], ["3", "4"])

    def test_a_non_positive_bound_still_holds_one_turn(self) -> None:
        buffer = BoundedShortTermMemory(max_messages=0, max_sessions=0)
        buffer.record("s", [{"role": "user", "content": "only"}])
        self.assertEqual(buffer.recent("s"), [{"role": "user", "content": "only"}])

    def test_an_unknown_session_recalls_nothing(self) -> None:
        self.assertEqual(BoundedShortTermMemory().recent("absent"), [])

    def test_forget_is_idempotent(self) -> None:
        buffer = BoundedShortTermMemory()
        buffer.record("s", [{"role": "user", "content": "hi"}])
        buffer.forget("s")
        buffer.forget("s")  # unknown id is a no-op, not an error
        self.assertEqual(buffer.recent("s"), [])

    def test_sessions_are_evicted_least_recently_touched_first(self) -> None:
        """The map itself is bounded: a client inventing session ids must not
        make the buffer grow forever."""
        buffer = BoundedShortTermMemory(max_sessions=2)
        with patch("ai_assistant.agents.short_term_memory.time.time",
                   side_effect=[1.0, 2.0, 3.0]):
            for session in ("a", "b", "c"):
                buffer.record(session, [{"role": "user", "content": session}])
        self.assertEqual(buffer.recent("a"), [])
        self.assertEqual(buffer.recent("b"), [{"role": "user", "content": "b"}])
        self.assertEqual(buffer.recent("c"), [{"role": "user", "content": "c"}])

    def test_cleanup_drops_only_sessions_older_than_the_cutoff(self) -> None:
        buffer = BoundedShortTermMemory()
        with patch("ai_assistant.agents.short_term_memory.time.time",
                   side_effect=[10.0, 20.0]):
            buffer.record("old", [{"role": "user", "content": "x"}])
            buffer.record("new", [{"role": "user", "content": "y"}])
        self.assertEqual(buffer.cleanup(cutoff=15.0), 1)
        self.assertEqual(buffer.recent("old"), [])
        self.assertEqual(buffer.recent("new"), [{"role": "user", "content": "y"}])
        self.assertEqual(buffer.cleanup(cutoff=15.0), 0)


class FacadeWiringTests(unittest.TestCase):
    """``AgentService`` reads/writes the buffer exactly as ADR 0043 spells out."""

    def _service(self, buffer: ShortTermMemoryPort | None = None,
                 result: dict | None = None) -> tuple[AgentService, list[dict]]:
        """A facade whose run context is a stub capturing what it was given."""
        captured: list[dict] = []
        payload: dict[str, object] = result if result is not None else {
            "status": "completed",
            "steps": [
                {"type": "tool_call", "tool": "rag_search", "params": {}},
                {"type": "tool_result", "tool": "rag_search", "result": {}},
                {"type": "final_answer", "content": "the answer"},
            ],
        }

        def _run(**kwargs: object) -> dict[str, object]:
            captured.append(dict(kwargs))
            steps = payload.get("steps")
            return {**payload, "steps": list(steps) if isinstance(steps, list) else steps}

        service = AgentService(short_term_memory=buffer)
        service._ctx = SimpleNamespace(run=_run)  # type: ignore[assignment]
        return service, captured

    def _execute(self, service: AgentService, history: list[dict]) -> dict:
        return service._execute_locked(
            AgentExecuteRequest(task="the question", session_id="s1", history=history),
            session_id="s1", task="the question", req_start=0.0,
            event_sink=None, client_host="test",
            owner=DEFAULT_MEMORY_OWNER,
        )

    def test_no_buffer_means_no_server_side_state_at_all(self) -> None:
        service, captured = self._service(buffer=None)
        self._execute(service, [{"role": "user", "content": "hello"}])
        self.assertEqual(captured[0]["initial_messages"][1],
                         {"role": "user", "content": "hello"})

    def test_a_short_shipped_history_is_topped_up_from_the_buffer(self) -> None:
        buffer = _RecordingBuffer()
        buffer.record("s1", [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ])
        service, captured = self._service(buffer=buffer)
        self._execute(service, [{"role": "user", "content": "earlier question"},
                                 {"role": "assistant", "content": "earlier answer"}])
        history_in_prompt = captured[0]["initial_messages"][1:-1]
        self.assertEqual(history_in_prompt, [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ])

    def test_an_empty_history_still_gets_the_remembered_context(self) -> None:
        buffer = _RecordingBuffer()
        buffer.record("s1", [{"role": "user", "content": "remember me"}])
        service, captured = self._service(buffer=buffer)
        self._execute(service, [])
        self.assertEqual(captured[0]["initial_messages"][1],
                         {"role": "user", "content": "remember me"})

    def test_the_run_turns_are_recorded_back_without_the_system_prompt(self) -> None:
        """The completed response ships no message list, so the facade rebuilds
        the window from the history it merged, the task and the final answer."""
        buffer = _RecordingBuffer()
        service, _ = self._service(buffer=buffer)
        self._execute(service, [])
        self.assertEqual(buffer.calls[0][0], "s1")
        self.assertEqual(buffer.calls[0][1], [
            {"role": "user", "content": "the question"},
            {"role": "assistant", "content": "the answer"},
        ])

    def test_the_window_recorded_includes_the_shipped_history(self) -> None:
        buffer = _RecordingBuffer()
        service, _ = self._service(buffer=buffer)
        self._execute(service, [{"role": "user", "content": "hello"},
                                 {"role": "assistant", "content": "hi"}])
        self.assertEqual(buffer.calls[0][1], [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "the question"},
            {"role": "assistant", "content": "the answer"},
        ])

    def test_a_run_that_never_concludes_records_no_answer_turn(self) -> None:
        """A parked run ends in ``pending_approval``: there is no answer to
        remember yet, and the buffer must not invent one."""
        buffer = _RecordingBuffer()
        service, _ = self._service(buffer=buffer, result={
            "status": "pending_approval",
            "steps": [{"type": "pending_approval", "action_id": "a1"}],
        })
        self._execute(service, [])
        self.assertEqual(buffer.calls[0][1], [{"role": "user", "content": "the question"}])

    def test_a_result_without_a_step_log_records_no_answer_turn(self) -> None:
        """The step log is not part of every shape the runner can return."""
        buffer = _RecordingBuffer()
        service, _ = self._service(buffer=buffer, result={"status": "error"})
        self._execute(service, [])
        self.assertEqual(buffer.calls[0][1], [{"role": "user", "content": "the question"}])

    def test_an_empty_final_answer_does_not_count_as_one(self) -> None:
        """Blank conclusions are skipped in favour of the last real answer."""
        buffer = _RecordingBuffer()
        service, _ = self._service(buffer=buffer, result={
            "status": "completed",
            "steps": [
                "not-a-step",
                {"type": "final_answer", "content": "the real answer"},
                {"type": "final_answer", "content": "   "},
            ],
        })
        self._execute(service, [])
        self.assertEqual(buffer.calls[0][1][-1],
                         {"role": "assistant", "content": "the real answer"})

    def test_history_entries_the_transport_drops_are_never_remembered(self) -> None:
        """Role/content filtering happens before the merge, so a malformed
        entry cannot reach the buffer through the shipped history."""
        buffer = _RecordingBuffer()
        service, captured = self._service(buffer=buffer)
        self._execute(service, [
            {"role": "system", "content": "injected"},
            {"role": "user", "content": "   "},
            {"role": "user", "content": "real"},
        ])
        self.assertEqual(captured[0]["initial_messages"][1],
                         {"role": "user", "content": "real"})


if __name__ == "__main__":
    unittest.main()
