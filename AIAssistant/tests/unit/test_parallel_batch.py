"""Parallel tool-batch execution and merge in the tool node (ADR 0044).

``_plan_batch`` decides, fail-closed, whether the pending call heads an
eligible group; ``_run_batch`` fans the group out; ``_record_batch`` merges the
branches back into ordered call/result pairs. These tests drive every refusal
arm, the real concurrent path (a barrier proves two branches overlap), the
bounded width, the per-branch exception barrier, cooperative cancellation
inside the group, the delegation precompute and the circuit breaker.
"""
from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import Future
from typing import Any
from unittest.mock import patch

from ai_assistant.orchestration.helpers import completed_plan_steps
from ai_assistant.orchestration.nodes import tool as tool_module
from ai_assistant.orchestration.nodes.tool import ToolContext, _plan_batch, tool_node

_PLAN = ["Thông tin kỹ sư A", "vai trò của project manager"]
#: Message of the branch the exception-containment test makes explode.
_BRANCH_ERROR = "sibling boom"


def _spec(plan: list[str], kinds: list[str] | None = None,
          groups: list[Any] | None = None) -> dict[str, Any]:
    return {
        "steps": list(plan),
        "step_kinds": list(kinds if kinds is not None else ["rag_search"] * len(plan)),
        "parallel_groups": list(groups if groups is not None else []),
    }


def _rag_call(**extra: Any) -> dict[str, Any]:
    step: dict[str, Any] = {
        "type": "tool_call", "tool": "rag_search", "params": {"query": "head query"},
    }
    step.update(extra)
    return step


def _state(steps: list[dict[str, Any]], *, plan: list[str] | None = None,
           plan_spec: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "steps": list(steps),
        "iteration": 1,
        "error_count": 0,
        "messages": [{"role": "user", "content": "hỏi"}],
        "plan": list(plan or []),
        "plan_spec": plan_spec,
        "tool_call_count": 0,
        "done": False,
        "pending_tool": None,
    }
    state.update(extra)
    return state


def _group_state(*, groups: list[Any] | None = None,
                 kinds: list[str] | None = None,
                 plan: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    plan = list(plan if plan is not None else _PLAN)
    return _state(
        [_rag_call(plan_step_index=0)], plan=plan,
        plan_spec=_spec(plan, kinds, [[0, 1]] if groups is None else groups),
        **extra,
    )


def _ctx(execute=None, *, verify=None, cancel_checker=None,
         requires_approval: frozenset[str] = frozenset(),
         warm_delegations=None, max_parallel: int = 4) -> ToolContext:
    def _default_execute(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"documents": [f"evidence:{params.get('query')}"]}

    return ToolContext(
        execute=execute or _default_execute,
        verify_result=verify,
        cancel_checker=cancel_checker or (lambda: False),
        requires_approval=requires_approval,
        warm_delegations=warm_delegations,
        max_parallel=max_parallel,
    )


def _only(update: dict[str, Any], step_type: str) -> list[dict[str, Any]]:
    return [step for step in update["steps"] if step.get("type") == step_type]


class _RecordingPool:
    """A ThreadPoolExecutor stand-in that runs each branch inline."""

    instances: list["_RecordingPool"] = []

    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = max_workers
        self.shutdown_wait: bool | None = None
        self.submitted = 0
        _RecordingPool.instances.append(self)

    def submit(self, fn, *args, **kwargs):
        self.submitted += 1
        future: Future = Future()
        future.set_result(fn(*args, **kwargs))
        return future

    def shutdown(self, wait: bool = True) -> None:
        self.shutdown_wait = wait


class PlanBatchEligibilityTests(unittest.TestCase):
    """ADR 0044 §3: any doubt about a member sends the whole group sequential."""

    def _calls(self, ctx: ToolContext, state: dict[str, Any] | None = None,
               tool: str = "rag_search",
               params: dict[str, Any] | None = None):
        state = state if state is not None else _group_state()
        head = state["steps"][0]
        return _plan_batch(
            state, ctx, head, tool,
            {"query": "head query"} if params is None else params,
        )

    def test_a_run_without_a_plan_never_batches(self) -> None:
        state = _group_state()
        state["plan"] = []
        self.assertIsNone(self._calls(_ctx(), state))

    def test_a_plan_without_a_spec_never_batches(self) -> None:
        state = _group_state()
        state["plan_spec"] = None
        self.assertIsNone(self._calls(_ctx(), state))

    def test_a_head_without_a_plan_step_index_never_batches(self) -> None:
        state = _group_state()
        state["steps"][0].pop("plan_step_index")
        self.assertIsNone(self._calls(_ctx(), state))

    def test_a_non_rag_head_never_batches(self) -> None:
        self.assertIsNone(self._calls(_ctx(), tool="read_file"))

    def test_a_group_that_needs_approval_never_batches(self) -> None:
        ctx = _ctx(requires_approval=frozenset({"rag_search"}))
        self.assertIsNone(self._calls(ctx))

    def test_a_head_that_starts_no_group_never_batches(self) -> None:
        state = _group_state(plan=["a", "b", "c"], groups=[[1, 2]])
        self.assertIsNone(self._calls(_ctx(), state))

    def test_a_non_list_group_table_never_batches(self) -> None:
        state = _group_state()
        state["plan_spec"]["parallel_groups"] = "0,1"
        self.assertIsNone(self._calls(_ctx(), state))

    def test_an_invalid_group_from_the_checkpoint_never_batches(self) -> None:
        state = _group_state(groups=[[0]])
        self.assertIsNone(self._calls(_ctx(), state))

    def test_a_spec_without_step_kinds_never_batches(self) -> None:
        state = _group_state()
        state["plan_spec"]["step_kinds"] = "rag_search"
        self.assertIsNone(self._calls(_ctx(), state))

    def test_a_group_reaching_past_step_kinds_never_batches(self) -> None:
        state = _group_state(kinds=["rag_search"])
        self.assertIsNone(self._calls(_ctx(), state))

    def test_a_direct_member_never_batches(self) -> None:
        """The model owes a ``direct`` step its answer; a worker cannot."""
        state = _group_state(kinds=["rag_search", "direct"])
        self.assertIsNone(self._calls(_ctx(), state))

    def test_a_single_threaded_context_never_batches(self) -> None:
        self.assertIsNone(self._calls(_ctx(max_parallel=1)))

    def test_a_group_is_selected_by_its_head_not_its_position(self) -> None:
        state = _group_state(plan=["a", "b", "c"], groups=[[1, 2], [0, 1]])
        calls = self._calls(_ctx(), state)
        self.assertIsNotNone(calls)
        self.assertEqual([index for _name, _params, index in calls], [0, 1])

    def test_a_broken_group_before_the_matching_one_is_ignored(self) -> None:
        state = _group_state(groups=[[9, 10], [0, 1]])
        calls = self._calls(_ctx(), state)
        self.assertIsNotNone(calls)

    def test_the_group_keeps_the_head_call_and_derives_the_siblings(self) -> None:
        calls = self._calls(_ctx())
        self.assertIsNotNone(calls)
        self.assertEqual(calls[0], ("rag_search", {"query": "head query"}, 0))
        self.assertEqual([index for _name, _params, index in calls], [0, 1])
        self.assertEqual(calls[1][0], "rag_search")
        self.assertIsInstance(calls[1][1]["query"], str)
        self.assertTrue(calls[1][1]["query"])

    def test_a_project_role_sibling_carries_the_wider_top_k(self) -> None:
        calls = self._calls(_ctx())
        self.assertIsNotNone(calls)
        self.assertEqual(calls[1][1]["top_k"], 8)


class ParallelBatchRunTests(unittest.TestCase):
    def setUp(self) -> None:
        _RecordingPool.instances.clear()

    def test_two_members_merge_as_ordered_call_result_pairs(self) -> None:
        seen: list[dict[str, Any]] = []

        def _execute(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
            seen.append(params)
            return {"documents": [f"evidence:{params['query']}"]}

        update = tool_node(_group_state(), _ctx(_execute))

        calls = _only(update, "tool_call")
        results = _only(update, "tool_result")
        self.assertEqual([call["plan_step_index"] for call in calls], [0, 1])
        self.assertEqual([result["plan_step_index"] for result in results], [0, 1])
        self.assertEqual([result["branch_index"] for result in results], [0, 1])
        self.assertEqual([call["batch_id"] for call in calls], ["batch-0-1", "batch-0-1"])
        self.assertEqual(calls[0]["branch_index"], 0)
        self.assertEqual(results[0]["result"], {"documents": ["evidence:head query"]})
        # The head's query is honoured verbatim; the sibling's is derived.
        self.assertIn({"query": "head query"}, seen)
        self.assertEqual(update["tool_call_count"], 2)
        self.assertTrue(update["synthesize_after_rag"])

    def test_the_siblings_evidence_rides_in_one_extra_message(self) -> None:
        update = tool_node(_group_state(), _ctx())
        messages = update["messages"]
        self.assertIn("Bằng chứng của các bước còn lại", messages[-2]["content"])
        self.assertIn("vai trò của project manager", messages[-2]["content"])
        self.assertIn("RAG evidence đã được xác minh", messages[-1]["content"])

    def test_the_batch_does_not_advance_the_plan_on_its_own(self) -> None:
        """Reflections (not raw results) advance the plan; the head's synthesis
        turn still owns step 0 (ADR 0044 §5)."""
        update = tool_node(_group_state(), _ctx())
        self.assertEqual(completed_plan_steps(update["steps"], _PLAN), 0)

    def test_the_branches_really_overlap_and_stay_ordered(self) -> None:
        barrier = threading.Barrier(2, timeout=10)
        main_thread = threading.get_ident()
        threads: list[int] = []

        def _execute(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
            threads.append(threading.get_ident())
            barrier.wait()
            if params["query"] == "head query":
                time.sleep(0.05)
            return {"documents": [params["query"]]}

        update = tool_node(_group_state(), _ctx(_execute))

        self.assertEqual(len(set(threads)), 2)
        self.assertNotIn(main_thread, threads)
        results = _only(update, "tool_result")
        self.assertEqual([result["result"]["documents"][0] for result in results],
                         ["head query", results[1]["result"]["documents"][0]])
        self.assertNotIn("error", results[0]["result"])

    def test_the_pool_width_is_bounded_by_the_setting_and_the_group(self) -> None:
        with patch.object(tool_module, "ThreadPoolExecutor", _RecordingPool):
            tool_node(_group_state(), _ctx(max_parallel=8))
        pool = _RecordingPool.instances[0]
        self.assertEqual(pool.max_workers, 2)
        self.assertEqual(pool.submitted, 2)
        self.assertTrue(pool.shutdown_wait)

    def test_one_throwing_branch_does_not_kill_the_group(self) -> None:
        def _execute(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
            if params["query"] != "head query":
                raise RuntimeError(_BRANCH_ERROR)
            return {"documents": ["ok"]}

        update = tool_node(_group_state(), _ctx(_execute))

        results = _only(update, "tool_result")
        self.assertNotIn("error", results[0]["result"])
        self.assertIn(_BRANCH_ERROR, results[1]["result"]["error"])
        self.assertEqual(update["error_count"], 1)
        self.assertNotIn("done", update)
        self.assertTrue(update["synthesize_after_rag"])

    def test_each_member_is_verified_before_its_result(self) -> None:
        verified: list[str] = []

        def _verify(tool_name: str, params: dict[str, Any],
                    result: dict[str, Any]) -> dict[str, Any]:
            verified.append(params["query"])
            return {"passed": True, "reason": "ok"}

        update = tool_node(_group_state(), _ctx(verify=_verify))
        sequence = [step["type"] for step in update["steps"]]
        self.assertEqual(sequence, [
            "tool_call", "tool_call", "verification", "tool_result",
            "verification", "tool_result",
        ])
        self.assertEqual(len(verified), 2)

    def test_the_batch_is_classified_before_it_fans_out(self) -> None:
        warmed: list[list[tuple[str, dict[str, Any]]]] = []
        update = tool_node(
            _group_state(), _ctx(warm_delegations=lambda calls: warmed.append(calls)),
        )
        self.assertEqual(len(warmed), 1)
        self.assertEqual([name for name, _params in warmed[0]], ["rag_search", "rag_search"])
        self.assertEqual(warmed[0][0][1], {"query": "head query"})
        self.assertEqual(len(_only(update, "tool_result")), 2)

    def test_a_clean_group_resets_the_error_streak(self) -> None:
        update = tool_node(_group_state(error_count=2), _ctx())
        self.assertEqual(update["error_count"], 0)

    def test_a_group_can_trip_the_circuit_breaker(self) -> None:
        def _execute(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
            return {"error": "down"}

        update = tool_node(_group_state(error_count=1), _ctx(_execute))

        self.assertTrue(update["done"])
        self.assertEqual(update["error_count"], 3)
        self.assertIn("Circuit Breaker", update["steps"][-1]["content"])

    def test_a_cancelled_branch_becomes_an_error_and_ends_the_run(self) -> None:
        ran: list[str] = []
        gate = {"calls": 0}

        def _cancel() -> bool:
            gate["calls"] += 1
            # The dispatch gate passes; the second branch is skipped.
            return gate["calls"] >= 3

        def _execute(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
            ran.append(params["query"])
            return {"documents": ["d"]}

        with patch.object(tool_module, "ThreadPoolExecutor", _RecordingPool):
            update = tool_node(_group_state(), _ctx(_execute, cancel_checker=_cancel))

        self.assertTrue(update["done"])
        self.assertTrue(update["cancelled"])
        self.assertEqual(update["steps"][-1]["type"], "cancelled")
        results = _only(update, "tool_result")
        self.assertEqual(len(results), 2)
        self.assertIn("cancelled", results[1]["result"]["error"])
        self.assertEqual(ran, ["head query"])
        pool = _RecordingPool.instances[0]
        self.assertEqual(pool.submitted, 1)
        self.assertTrue(pool.shutdown_wait)

    def test_the_head_is_stamped_where_it_sits_in_the_log(self) -> None:
        """A batch may follow earlier, completed tool calls; only the pending
        call (by identity) is the head and only it gets the batch label."""
        state = _group_state()
        state["steps"] = [
            {"type": "tool_call", "tool": "read_file", "params": {"path": "x"}},
            {"type": "tool_result", "tool": "read_file", "result": {"content": "ok"}},
        ] + state["steps"]
        head = state["steps"][2]

        update = tool_node(state, _ctx())

        self.assertNotIn("batch_id", update["steps"][0])
        self.assertEqual(update["steps"][2]["batch_id"], "batch-0-1")
        self.assertEqual(update["steps"][2]["branch_index"], 0)
        self.assertIsNot(update["steps"][2], head)
        self.assertEqual(len(_only(update, "tool_result")), 3)

    def test_an_approval_gated_tool_runs_the_group_sequentially(self) -> None:
        ran: list[str] = []

        def _execute(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
            ran.append(params["query"])
            return {"documents": ["d"]}

        update = tool_node(
            _group_state(),
            _ctx(_execute, requires_approval=frozenset({"rag_search"})),
        )

        self.assertEqual(ran, ["head query"])
        self.assertEqual(update["tool_call_count"], 1)
        self.assertNotIn("batch_id", update["steps"][0])

    def test_a_single_threaded_context_runs_the_group_sequentially(self) -> None:
        ran: list[str] = []

        def _execute(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
            ran.append(params["query"])
            return {"documents": ["d"]}

        update = tool_node(_group_state(), _ctx(_execute, max_parallel=1))

        self.assertEqual(ran, ["head query"])
        self.assertEqual(update["tool_call_count"], 1)

    def test_a_group_that_no_longer_fits_the_plan_runs_sequentially(self) -> None:
        """The spec was written by the planner; a resumed run must re-validate
        it against the plan it actually restored."""
        state = _state(
            [_rag_call(plan_step_index=0)], plan=["chỉ còn một bước"],
            plan_spec=_spec(_PLAN, ["rag_search", "rag_search"], [[0, 1]]),
        )
        ran: list[str] = []

        def _execute(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
            ran.append(params["query"])
            return {"documents": ["d"]}

        update = tool_node(state, _ctx(_execute))

        self.assertEqual(ran, ["head query"])
        self.assertEqual(update["tool_call_count"], 1)


if __name__ == "__main__":
    unittest.main()
