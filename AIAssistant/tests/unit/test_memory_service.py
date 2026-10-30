"""Long-term memory service: fail-closed recall and gated consolidation (ADR 0043).

The service is the trust crossing of the memory subsystem, so every refusal
is pinned here rather than assumed: a missing store, a zero ``top_k``, a
throwing store, an unknown kind at the re-check, a credential-shaped record
that the redaction gate must drop whole, a write that raises. The facade
wiring is covered too: recall appends a block to the system prompt and
announces itself on the step log; consolidation runs for a completed run and
never for a parked one.
"""
from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from ai_assistant.agents import context as agents_context
from ai_assistant.agents.models import AgentExecuteRequest
from ai_assistant.agents.service import AgentService
from ai_assistant.application.memory import MemoryService
from ai_assistant.domain.memory import (
    DEFAULT_MEMORY_OWNER,
    MemoryRecord,
    task_signature,
)

NOW = 1_700_000_000.0
DAY = 86400.0


def _record(**overrides: object) -> MemoryRecord:
    base: dict[str, object] = {
        "id": "m1",
        "kind": "semantic",
        # Same default the service composes with, so the stub's owner filter
        # matches what a real store would be asked for.
        "owner": "default",
        "text": "tim file main.cpp",
        "created_at": NOW,
        "last_used_at": NOW,
        "salience": 0.5,
        "metadata": {},
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


class _StubStore:
    """An in-memory ``LongTermMemoryPort`` that can be made to fail."""

    def __init__(self, records: list[MemoryRecord] | None = None,
                 *, fail_on_add: bool = False,
                 fail_on_prune: bool = False) -> None:
        self.records = list(records or [])
        self.added: list[MemoryRecord] = []
        self.pruned: list[tuple[int, int]] = []
        self.fail_on_add = fail_on_add
        self.fail_on_prune = fail_on_prune

    def candidates(self, owner: str, kinds: frozenset[str],
                   *, limit: int = 200) -> list[MemoryRecord]:
        return [r for r in self.records
                if r["owner"] == owner and r["kind"] in kinds][:limit]

    def add(self, record: MemoryRecord) -> None:
        if self.fail_on_add:
            msg = "disk on fire"
            raise OSError(msg)
        self.added.append(record)

    def prune(self, retention_days: int, *, max_bytes: int = 0) -> int:
        if self.fail_on_prune:
            msg = "prune failed"
            raise OSError(msg)
        self.pruned.append((retention_days, max_bytes))
        return 0

    def close(self) -> None:  # pragma: no cover - never called by the service
        return None


class _ThrowingStore:
    """A store whose read path raises — the fail-closed recall case."""

    def candidates(self, owner: str, kinds: frozenset[str],
                   *, limit: int = 200) -> list[MemoryRecord]:
        msg = "db locked"
        raise RuntimeError(msg)


class _UnfilteredStore:
    """A store that returns candidates regardless of the kind filter.

    The port lets the store narrow by kind, but the service must not *trust*
    that narrowing: a buggy or hostile adapter can return a kind outside the
    vocabulary, and the re-check at the trust crossing is what drops it.
    """

    def __init__(self, records: list[MemoryRecord]) -> None:
        self.records = records

    def candidates(self, owner: str, kinds: frozenset[str],
                   *, limit: int = 200) -> list[MemoryRecord]:
        return list(self.records)

    def add(self, record: MemoryRecord) -> None:  # pragma: no cover - unused
        return None


class MemoryServiceRecallTests(unittest.TestCase):
    def test_no_store_means_disabled_and_no_recall(self) -> None:
        service = MemoryService(None, owner=DEFAULT_MEMORY_OWNER)
        self.assertFalse(service.enabled)
        self.assertEqual(service.recall("tim file main.cpp"), [])

    def test_a_zero_top_k_means_disabled(self) -> None:
        service = MemoryService(_StubStore([_record()]), top_k=0,
                                owner=DEFAULT_MEMORY_OWNER)
        self.assertFalse(service.enabled)
        self.assertEqual(service.recall("tim file main.cpp"), [])

    def test_a_enabled_store_ranks_and_truncates_to_top_k(self) -> None:
        store = _StubStore([
            _record(id="old", created_at=NOW - 60 * DAY, text="tim file main.cpp"),
            _record(id="new", created_at=NOW, text="tim file main.cpp"),
            _record(id="other", text="ve sinh o cung"),
        ])
        service = MemoryService(store, top_k=1, owner=DEFAULT_MEMORY_OWNER)
        self.assertTrue(service.enabled)
        found = service.recall("tim file main.cpp")
        # The no-overlap record is dropped, the fresher hit wins, top_k=1.
        self.assertEqual([r["id"] for r in found], ["new"])

    def test_a_throwing_store_recalls_nothing_instead_of_failing(self) -> None:
        """Memory must never fail a run: a broken store degrades to no memory."""
        service = MemoryService(_ThrowingStore(), owner=DEFAULT_MEMORY_OWNER)  # type: ignore[arg-type]
        self.assertEqual(service.recall("tim file main.cpp"), [])

    def test_a_candidate_with_an_unknown_kind_is_dropped_at_the_boundary(self) -> None:
        """The vocabulary is re-checked where the store's output is trusted —
        the port lets the store filter by kind, but a buggy or hostile adapter
        can still return something else."""
        store = _UnfilteredStore([
            _record(id="dream", kind="dream", text="tim file main.cpp"),
            _record(id="real", kind="semantic", text="tim file main.cpp"),
        ])
        service = MemoryService(store, top_k=5, owner=DEFAULT_MEMORY_OWNER)  # type: ignore[arg-type]
        found = service.recall("tim file main.cpp")
        self.assertEqual([r["id"] for r in found], ["real"])

    def test_construction_refuses_an_empty_owner(self) -> None:
        """The silent ``"default"`` was the multi-tenant defect: an explicit
        owner, or no service at all (ADR 0043)."""
        with self.assertRaises(ValueError):
            MemoryService(None, owner="  ")

    def test_a_per_call_owner_overrides_the_configured_namespace(self) -> None:
        store = _StubStore([
            _record(id="alice", owner="alice", text="tim file main.cpp"),
            _record(id="bob", owner="bob", text="tim file main.cpp"),
        ])
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        found = service.recall("tim file main.cpp", owner="alice")
        self.assertEqual([r["id"] for r in found], ["alice"])

    def test_a_blank_per_call_owner_is_refused(self) -> None:
        service = MemoryService(_StubStore(), owner=DEFAULT_MEMORY_OWNER)
        with self.assertRaises(ValueError):
            service.recall("tim file main.cpp", owner="")


class MemoryServiceConsolidationTests(unittest.TestCase):
    STEPS: list[dict[str, Any]] = [
        {"type": "tool_call", "tool": "rag_search", "params": {}},
        {"type": "final_answer", "content": "da tim thay"},
    ]

    def test_no_store_persists_nothing_and_reports_zero(self) -> None:
        service = MemoryService(None, owner=DEFAULT_MEMORY_OWNER)
        self.assertEqual(service.consolidate_run("task", "s1", self.STEPS), 0)

    def test_a_completed_run_persists_an_episodic_record_and_reports_the_count(self) -> None:
        store = _StubStore()
        service = MemoryService(store, owner="alice")
        stored = service.consolidate_run("tim file main.cpp", "s1", self.STEPS)
        self.assertEqual(stored, 1)
        self.assertEqual(len(store.added), 1)
        self.assertEqual(store.added[0]["kind"], "episodic")
        self.assertEqual(store.added[0]["owner"], "alice")
        self.assertIn("tim file main.cpp", store.added[0]["text"])
        self.assertIn("da tim thay", store.added[0]["text"])

    def test_a_token_shaped_task_is_dropped_by_the_redaction_gate(self) -> None:
        """ADR 0043 §5: redact first, then refuse what is still sensitive.
        ``scrub_pii`` masks keyword assignments (``password: …``) but has no
        pattern for a bare token shape, so an ``sk-…`` key survives redaction
        and the record must be dropped whole rather than stored."""
        store = _StubStore()
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        stored = service.consolidate_run(
            "dang nhap voi sk-abcdefghijklmnop de xem bao cao", "s1", self.STEPS,
        )
        self.assertEqual(stored, 0)
        self.assertEqual(store.added, [])

    def test_a_keyword_assignment_is_masked_and_the_record_still_lands(self) -> None:
        """The other half of the gate: ``scrub_pii`` already removed the
        secret, so the record is safe to keep with the masked text."""
        store = _StubStore()
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        stored = service.consolidate_run("password: hunter2secret", "s1", [])
        self.assertEqual(stored, 1)
        self.assertIn("[REDACTED_SECRET]", store.added[0]["text"])
        self.assertNotIn("hunter2secret", store.added[0]["text"])

    def test_pii_is_masked_before_the_record_is_stored(self) -> None:
        store = _StubStore()
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        stored = service.consolidate_run("gui mail cho nguyen@example.com", "s1", [])
        self.assertEqual(stored, 1)
        self.assertIn("[REDACTED_EMAIL]", store.added[0]["text"])
        self.assertNotIn("nguyen@example.com", store.added[0]["text"])

    def test_a_failing_write_is_contained_and_not_counted(self) -> None:
        store = _StubStore(fail_on_add=True)
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        self.assertEqual(service.consolidate_run("task", "s1", self.STEPS), 0)

    def test_one_bad_record_does_not_stop_the_others(self) -> None:
        """The gate decides per record: the episodic text embeds the run's
        answer (here a bare token shape that survives redaction, so that
        record is dropped whole), while the procedural plan from the same
        run still lands."""
        store = _StubStore()
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        steps = [
            {"type": "plan", "steps": ["buoc 1", "buoc 2"]},
            {"type": "final_answer", "content": "dang nhap voi sk-abcdefghijklmnop"},
        ]
        stored = service.consolidate_run("task binh thuong", "s1", steps)
        self.assertEqual(stored, 1)
        self.assertEqual(len(store.added), 1)
        self.assertEqual(store.added[0]["kind"], "procedural")

    def test_a_per_call_owner_scopes_the_written_records(self) -> None:
        """The A2A run writes under its tenant, not under the composition's
        default owner (ADR 0043)."""
        store = _StubStore()
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        service.consolidate_run("tim file main.cpp", "s1", self.STEPS,
                                owner="tenant-a")
        self.assertEqual([r["owner"] for r in store.added], ["tenant-a"])

    def test_a_successful_consolidation_runs_the_retention_policy(self) -> None:
        store = _StubStore()
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER,
                                retention_days=30, retention_max_bytes=1024)
        self.assertEqual(service.consolidate_run("task", "s1", self.STEPS), 1)
        self.assertEqual(store.pruned, [(30, 1024)])

    def test_retention_is_off_by_default(self) -> None:
        store = _StubStore()
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        service.consolidate_run("task", "s1", self.STEPS)
        self.assertEqual(store.pruned, [])

    def test_nothing_stored_means_no_retention_pass(self) -> None:
        store = _StubStore(fail_on_add=True)
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER, retention_days=30)
        self.assertEqual(service.consolidate_run("task", "s1", self.STEPS), 0)
        self.assertEqual(store.pruned, [])

    def test_a_failing_prune_is_contained(self) -> None:
        """Retention runs on the response path: a prune failure must not turn
        a succeeded run into a failed one."""
        store = _StubStore(fail_on_prune=True)
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER, retention_days=30)
        with self.assertLogs("ai_assistant.application.memory", level="WARNING"):
            stored = service.consolidate_run("task", "s1", self.STEPS)
        self.assertEqual(stored, 1)


class MemoryServiceProceduralTests(unittest.TestCase):
    """Procedural memory is an exact-key lookup, not a rank (ADR 0043 §7).

    The record was written under ``task_signature(task)``; only that
    signature reads it back. Every refusal is pinned: memory off, a store
    that raises, a row of the wrong kind (the store's filter is a hint),
    metadata that is not a mapping, text that is not a string or splits
    into nothing.
    """

    TASK = "tim file main.cpp"
    PLAN = "Buoc doc thu muc | Buoc tim ten file"

    def _record(self, **overrides: object) -> MemoryRecord:
        base: dict[str, object] = {
            "id": f"proc-{task_signature(self.TASK)[:24]}",
            "kind": "procedural",
            "owner": "default",
            "text": self.PLAN,
            "created_at": NOW,
            "last_used_at": NOW,
            "salience": 0.5,
            "metadata": {"task_signature": task_signature(self.TASK)},
        }
        base.update(overrides)
        return base  # type: ignore[return-value]

    def test_memory_off_means_no_remembered_plan(self) -> None:
        self.assertIsNone(MemoryService(None, owner=DEFAULT_MEMORY_OWNER).procedural_plan(self.TASK))

    def test_a_zero_top_k_disables_the_lookup_too(self) -> None:
        """``AGENT_MEMORY_LTM_TOP_K=0`` is the off switch for *both* halves."""
        service = MemoryService(_StubStore([self._record()]), top_k=0,
                                owner=DEFAULT_MEMORY_OWNER)
        self.assertIsNone(service.procedural_plan(self.TASK))

    def test_a_throwing_store_yields_no_plan_instead_of_failing(self) -> None:
        service = MemoryService(_ThrowingStore(), owner=DEFAULT_MEMORY_OWNER)  # type: ignore[arg-type]
        self.assertIsNone(service.procedural_plan(self.TASK))

    def test_the_matching_record_is_split_back_into_steps(self) -> None:
        store = _StubStore([self._record()])
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        self.assertEqual(service.procedural_plan(self.TASK),
                         ["Buoc doc thu muc", "Buoc tim ten file"])

    def test_the_lookup_survives_how_the_user_types_the_task(self) -> None:
        """Write and read key on the normalised signature, not the raw text."""
        store = _StubStore([self._record()])
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        self.assertEqual(len(service.procedural_plan("Tìm  FILE main.cpp") or []), 2)

    def test_a_record_for_another_task_is_not_reused(self) -> None:
        store = _StubStore([
            self._record(metadata={"task_signature": task_signature("ve sinh o cung")}),
        ])
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        self.assertIsNone(service.procedural_plan(self.TASK))

    def test_a_non_procedural_row_is_never_used_as_a_plan(self) -> None:
        """The store's kind filter is a hint, not a guarantee: a semantic
        fact whose metadata carries the same signature is not a recipe."""
        store = _UnfilteredStore([
            _record(kind="semantic", text="thich mau xanh",
                    metadata={"task_signature": task_signature(self.TASK)}),
        ])
        service = MemoryService(store, top_k=5, owner=DEFAULT_MEMORY_OWNER)  # type: ignore[arg-type]
        self.assertIsNone(service.procedural_plan(self.TASK))

    def test_metadata_that_is_not_a_mapping_is_skipped(self) -> None:
        store = _StubStore([self._record(metadata=None)])
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        self.assertIsNone(service.procedural_plan(self.TASK))

    def test_text_that_is_not_a_string_is_skipped(self) -> None:
        store = _StubStore([self._record(text=42)])
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        self.assertIsNone(service.procedural_plan(self.TASK))

    def test_text_that_splits_into_no_steps_yields_no_plan(self) -> None:
        store = _StubStore([self._record(text=" | ")])
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        self.assertIsNone(service.procedural_plan(self.TASK))

    def test_the_first_matching_record_wins(self) -> None:
        store = _StubStore([
            self._record(id="proc-a", text="Buoc mot | Buoc hai"),
            self._record(id="proc-b", text="Buoc khac"),
        ])
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        self.assertEqual(service.procedural_plan(self.TASK), ["Buoc mot", "Buoc hai"])

    def test_the_lookup_is_scoped_to_the_call_owner(self) -> None:
        store = _StubStore([
            self._record(id="a", owner="alice", text="Alice step"),
            self._record(id="b", owner="bob", text="Bob step"),
        ])
        service = MemoryService(store, owner=DEFAULT_MEMORY_OWNER)
        self.assertEqual(service.procedural_plan(self.TASK, owner="bob"), ["Bob step"])
        self.assertIsNone(service.procedural_plan(self.TASK))


class FacadeLongTermWiringTests(unittest.TestCase):
    """``AgentService`` consults memory the way ADR 0043 spells out."""

    COMPLETED: dict[str, Any] = {
        "status": "completed",
        "steps": [
            {"type": "tool_call", "tool": "rag_search", "params": {}},
            {"type": "final_answer", "content": "da tim thay"},
        ],
    }

    def _service(self, store: _StubStore,
                 result: dict[str, Any] | None = None,
                 *, top_k: int = 5) -> tuple[AgentService, list[dict[str, Any]]]:
        captured: list[dict[str, Any]] = []
        payload = result if result is not None else self.COMPLETED

        def _run(**kwargs: Any) -> dict[str, Any]:
            captured.append(dict(kwargs))
            return dict(payload)

        service = AgentService(memory_service=MemoryService(
            store, owner=DEFAULT_MEMORY_OWNER, top_k=top_k,
        ))
        # Patch the module-level runner so the *real* AgentRunContext (and its
        # consolidation decision) is exercised instead of a stub that would
        # hide a completion path.
        patcher = patch.object(agents_context, "run_langgraph_agent", side_effect=_run)
        patcher.start()
        self.addCleanup(patcher.stop)
        return service, captured

    def _execute(self, service: AgentService,
                 *, owner: str = DEFAULT_MEMORY_OWNER) -> dict[str, Any]:
        return service._execute_locked(
            AgentExecuteRequest(task="tim file main.cpp", session_id="s1", history=[]),
            session_id="s1", task="tim file main.cpp", req_start=0.0,
            event_sink=None, client_host="test", owner=owner,
        )

    def test_recalled_memory_is_appended_to_the_prompt_and_announced(self) -> None:
        store = _StubStore([_record(text="tim file main.cpp o thu muc src")])
        service, captured = self._service(store)
        self._execute(service)
        system_prompt = captured[0]["system_prompt"]
        self.assertIn("[Long-term memory]", system_prompt)
        self.assertIn("tim file main.cpp o thu muc src", system_prompt)
        self.assertEqual(captured[0]["initial_steps"],
                         [{"type": "memory_recall", "count": 1}])

    def test_an_empty_recall_leaves_the_prompt_untouched(self) -> None:
        store = _StubStore([])  # candidates exist but none overlap the query
        service, captured = self._service(store)
        self._execute(service)
        self.assertNotIn("[Long-term memory]", captured[0]["system_prompt"])
        self.assertIsNone(captured[0]["initial_steps"])

    def test_a_disabled_memory_service_is_never_consulted(self) -> None:
        store = _StubStore([_record(text="tim file main.cpp")])
        service, captured = self._service(store, top_k=0)
        self._execute(service)
        self.assertNotIn("[Long-term memory]", captured[0]["system_prompt"])
        self.assertEqual(store.added, [])

    def test_a_completed_run_is_consolidated_with_the_task_and_steps(self) -> None:
        store = _StubStore()
        service, _ = self._service(store)
        self._execute(service)
        self.assertEqual(len(store.added), 1)
        self.assertIn("tim file main.cpp", store.added[0]["text"])

    def test_a_parked_run_is_never_consolidated(self) -> None:
        store = _StubStore()
        service, _ = self._service(store, result={
            "status": "pending_approval",
            "steps": [{"type": "pending_approval", "action_id": "a1"}],
        })
        self._execute(service)
        self.assertEqual(store.added, [])

    def test_a_remembered_recipe_is_offered_to_the_runner(self) -> None:
        """The procedural half of memory reaches the graph as ``prior_plan``
        (ADR 0043 §7), keyed on the very task the run is about."""
        store = _StubStore([
            _record(kind="procedural", text="Buoc mot | Buoc hai",
                    metadata={"task_signature": task_signature("tim file main.cpp")}),
        ])
        service, captured = self._service(store)
        self._execute(service)
        self.assertEqual(captured[0]["prior_plan"], ["Buoc mot", "Buoc hai"])

    def test_no_matching_recipe_means_no_prior_plan(self) -> None:
        store = _StubStore([])
        service, captured = self._service(store)
        self._execute(service)
        self.assertIsNone(captured[0]["prior_plan"])

    def test_the_run_owner_scopes_both_memory_directions(self) -> None:
        """The owner a run was started with reaches recall and consolidation:
        one service, one store, two tenants, no shared namespace."""
        store = _StubStore([_record(owner="tenant-a", text="tim file main.cpp")])
        service, captured = self._service(store)
        self._execute(service, owner="tenant-a")
        self.assertIn("[Long-term memory]", captured[0]["system_prompt"])
        self.assertEqual([r["owner"] for r in store.added], ["tenant-a"])

    def test_another_owners_records_are_never_injected(self) -> None:
        store = _StubStore([_record(owner="tenant-a", text="tim file main.cpp")])
        service, captured = self._service(store)
        self._execute(service, owner="tenant-b")
        self.assertNotIn("[Long-term memory]", captured[0]["system_prompt"])
        self.assertEqual([r["owner"] for r in store.added], ["tenant-b"])

    def test_the_run_owner_scopes_the_procedural_lookup(self) -> None:
        store = _StubStore([
            _record(kind="procedural", owner="tenant-a", text="Buoc mot | Buoc hai",
                    metadata={"task_signature": task_signature("tim file main.cpp")}),
        ])
        service, captured = self._service(store)
        self._execute(service, owner="tenant-a")
        self.assertEqual(captured[0]["prior_plan"], ["Buoc mot", "Buoc hai"])


if __name__ == "__main__":
    unittest.main()
