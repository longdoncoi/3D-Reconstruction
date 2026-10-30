"""Guards for the memory domain vocabulary, ranking and short-term policy (ADR 0043).

``recall_rank`` is a pure decision, so every arm of it is pinned directly:
the "nothing" answer (``top_k <= 0``), the no-overlap drop (an unrelated
memory must never be injected), the empty-query recency browse, the decay
arithmetic and the deterministic tie-break. The runtime twin
``is_memory_kind`` re-derives from the vocabulary, so a rehydrated record of
an unknown kind is dropped instead of recalled.
"""
from __future__ import annotations

import unittest

from ai_assistant.domain.memory import (
    CONTEXT_TOTAL_LIMIT,
    DEFAULT_MEMORY_OWNER,
    MAX_SEMANTIC_PER_RUN,
    MEMORY_HALF_LIFE_DAYS,
    MEMORY_KINDS,
    build_run_records,
    conversation_turns,
    estimate_tokens,
    format_memory_block,
    is_memory_kind,
    merge_history,
    recall_rank,
    summarize_messages,
    task_signature,
    trim_history,
)

DAY = 86400.0
NOW = 1_700_000_000.0


def _record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "m1",
        "kind": "semantic",
        "owner": "u",
        "text": "nguoi dung thich mau xanh",
        "created_at": NOW - 10 * DAY,
        "last_used_at": NOW,
        "salience": 0.5,
        "metadata": {},
    }
    base.update(overrides)
    return base


class MemoryKindTests(unittest.TestCase):
    def test_vocabulary_is_exactly_the_three_documented_kinds(self) -> None:
        self.assertEqual(MEMORY_KINDS, frozenset({"semantic", "episodic", "procedural"}))

    def test_unknown_or_non_string_kind_is_rejected(self) -> None:
        """The runtime gate drops what the compiler cannot see: a JSON record
        may spell any string, and only the vocabulary is recallable."""
        self.assertFalse(is_memory_kind("dream"))
        self.assertFalse(is_memory_kind(7))
        self.assertFalse(is_memory_kind(None))
        self.assertTrue(is_memory_kind("episodic"))


class RecallRankTests(unittest.TestCase):
    def test_top_k_zero_returns_nothing_even_with_hits(self) -> None:
        self.assertEqual(recall_rank([_record()], "nguoi dung", 0, now=NOW), [])

    def test_non_matching_record_is_dropped_not_ranked_last(self) -> None:
        """A zero-overlap row is noise in the prompt — the guard that keeps
        recall from degenerating into "recency with extra steps"."""
        hit, miss = _record(id="hit", text="nguoi dung thich mau xanh"), _record(id="miss", text="mat khau cua may")
        ranked = recall_rank([miss, hit], "nguoi dung mau xanh", 5, now=NOW)
        self.assertEqual([r["id"] for r in ranked], ["hit"])

    def test_empty_query_browses_by_recency_and_salience(self) -> None:
        old = _record(id="old", created_at=NOW - 200 * DAY, salience=0.0)
        new = _record(id="new", created_at=NOW - 1 * DAY, salience=1.0)
        ranked = recall_rank([old, new], "", 5, now=NOW)
        self.assertEqual([r["id"] for r in ranked], ["new", "old"])

    def test_recency_decay_halves_the_keyword_weight_after_one_half_life(self) -> None:
        fresh = _record(id="fresh", created_at=NOW, text="chien luoc xay dung")
        stale = _record(id="stale", created_at=NOW - MEMORY_HALF_LIFE_DAYS * DAY, text="chien luoc xay dung")
        ranked = recall_rank([stale, fresh], "chien luoc", 2, now=NOW)
        self.assertEqual([r["id"] for r in ranked], ["fresh", "stale"])

    def test_ties_break_by_created_at_then_id_so_order_is_deterministic(self) -> None:
        b = _record(id="b", created_at=NOW - 5 * DAY, text="cong nghe moi")
        a = _record(id="a", created_at=NOW - 5 * DAY, text="cong nghe moi")
        newer = _record(id="z", created_at=NOW - 1 * DAY, text="cong nghe moi")
        ranked = recall_rank([b, a, newer], "cong nghe", 3, now=NOW)
        self.assertEqual([r["id"] for r in ranked], ["z", "a", "b"])

    def test_defaults_fill_in_for_a_sparse_record(self) -> None:
        """A record missing ``text``/``created_at``/``salience``/``id`` must
        rank without raising — unknown age and rank are neutral, never fatal."""
        sparse = {"text": "du an 3d"}
        ranked = recall_rank([sparse], "du an", 1, now=NOW)
        self.assertEqual(len(ranked), 1)

    def test_non_string_text_is_scored_as_its_string_form(self) -> None:
        ranked = recall_rank([{"text": 42, "id": "n"}], "42", 1, now=NOW)
        self.assertEqual(len(ranked), 1)

    def test_a_match_on_a_query_with_no_normalisable_tokens_browses(self) -> None:
        """Punctuation-only queries carry no tokens, so overlap cannot be
        computed; the ranker treats that like the browse case instead of
        dropping everything."""
        ranked = recall_rank([_record(id="x")], "!!!", 1, now=NOW)
        self.assertEqual(len(ranked), 1)

    def test_top_k_truncates_the_ranked_list(self) -> None:
        records = [_record(id=f"m{i}", text="du an 3d reconstruction") for i in range(3)]
        self.assertEqual(len(recall_rank(records, "du an", 2, now=NOW)), 2)

    def test_default_clock_is_used_when_now_is_omitted(self) -> None:
        ranked = recall_rank([_record(text="du an 3d")], "du an", 1)
        self.assertEqual(len(ranked), 1)


class ShortTermPolicyTests(unittest.TestCase):
    def test_estimate_tokens_uses_the_historical_ratio(self) -> None:
        self.assertEqual(estimate_tokens("a" * 22), 10)

    def test_trim_keeps_the_whole_window_when_it_already_fits(self) -> None:
        messages = [{"content": "x" * 10}]
        self.assertEqual(trim_history(messages, max_tokens=1000), messages)

    def test_trim_drops_oldest_until_the_budget_fits(self) -> None:
        messages = [{"content": "x" * 500} for _ in range(6)]
        trimmed = trim_history(messages, max_tokens=500)
        self.assertLess(len(trimmed), len(messages))
        self.assertGreaterEqual(len(trimmed), 1)

    def test_trim_never_empties_the_history(self) -> None:
        self.assertEqual(len(trim_history([{"content": "x" * 9999}], max_tokens=1)), 1)

    def test_trim_ignores_non_string_content_in_the_budget(self) -> None:
        """A message whose content is not a string contributes no tokens and
        must not be popped (``isinstance`` guard in the subtraction)."""
        messages = [{"content": 12}, {"content": "x" * 100}, {"content": "y" * 100}]
        trimmed = trim_history(messages, max_tokens=10)
        self.assertEqual(len(trimmed), 1)
        self.assertEqual(trimmed[0]["content"], "y" * 100)

    def test_short_conversation_under_the_total_budget_is_returned_untouched(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        self.assertIs(summarize_messages(messages), messages)

    def test_over_budget_history_is_compacted_into_the_total_limit(self) -> None:
        messages = (
            [{"role": "system", "content": "s" * 500}, {"role": "user", "content": "u" * 500}]
            + [{"role": "assistant", "content": "a" * 900} for _ in range(30)]
        )
        result = summarize_messages(messages)
        self.assertLessEqual(sum(len(m["content"]) for m in result), CONTEXT_TOTAL_LIMIT)


class ShortTermMergeTests(unittest.TestCase):
    """The buffer tops up the shipped history without ever dropping a turn."""

    U1 = {"role": "user", "content": "question 1"}
    A1 = {"role": "assistant", "content": "answer 1"}
    U2 = {"role": "user", "content": "question 2"}
    A2 = {"role": "assistant", "content": "answer 2"}

    def test_an_empty_buffer_passes_the_shipped_history_through(self) -> None:
        incoming = [self.U1]
        merged = merge_history([], incoming)
        self.assertEqual(merged, [self.U1])
        self.assertIsNot(merged[0], incoming[0])  # copies, never aliases

    def test_an_empty_shipped_history_keeps_the_buffered_context(self) -> None:
        self.assertEqual(merge_history([self.U1, self.A1], []), [self.U1, self.A1])

    def test_reshipping_a_shorter_window_of_the_same_conversation_does_not_duplicate(self) -> None:
        buffered = [self.U1, self.A1, self.U2]
        self.assertEqual(merge_history(buffered, [self.A1, self.U2]), buffered)

    def test_a_window_from_the_middle_of_the_buffer_does_not_duplicate(self) -> None:
        """A slice that neither starts nor ends at the buffer's tail has no
        overlap to key on — only the window rule keeps it from repeating."""
        buffered = [self.U1, self.A1, self.U2, self.A2]
        self.assertEqual(merge_history(buffered, [self.A1, self.U2]), buffered)

    def test_a_continuing_conversation_is_appended_exactly_once(self) -> None:
        merged = merge_history([self.U1, self.A1], [self.U1, self.A1, self.U2])
        self.assertEqual(merged, [self.U1, self.A1, self.U2])

    def test_an_interleaved_tail_continues_without_repeating_the_overlap(self) -> None:
        merged = merge_history([self.U1, self.A1], [self.A1, self.U2])
        self.assertEqual(merged, [self.U1, self.A1, self.U2])

    def test_a_wholly_new_conversation_is_kept_after_the_remembered_context(self) -> None:
        merged = merge_history([self.U1], [self.U2, self.A2])
        self.assertEqual(merged, [self.U1, self.U2, self.A2])

    def test_conversation_turns_keep_only_what_a_client_ships(self) -> None:
        """System prompts are rebuilt per run and a bare turn without content
        still counts (its content defaults to ``""`` instead of vanishing)."""
        turns = conversation_turns([
            {"role": "system", "content": "instructions"},
            {"role": "tool", "content": "observation"},
            {"role": "user"},
            {"role": "assistant", "content": "hi"},
            {"content": "roleless"},
        ])
        self.assertEqual(turns, [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "hi"},
        ])


class TaskSignatureTests(unittest.TestCase):
    def test_accent_case_and_separator_noise_share_one_signature(self) -> None:
        """The procedural key must survive how the user happens to type."""
        canonical = task_signature("Tìm file main.cpp")
        self.assertEqual(task_signature("tim file main.cpp"), canonical)
        self.assertEqual(task_signature("  TIM   FILE   main.cpp  "), canonical)

    def test_a_different_task_gets_a_different_signature(self) -> None:
        self.assertNotEqual(task_signature("Tìm file main.cpp"),
                            task_signature("Tạo file main.cpp"))

    def test_the_signature_is_64_hex_characters(self) -> None:
        signature = task_signature("task")
        self.assertEqual(len(signature), 64)
        self.assertTrue(all(ch in "0123456789abcdef" for ch in signature))


class BuildRunRecordTests(unittest.TestCase):
    def test_every_run_leaves_exactly_one_episodic_record(self) -> None:
        records = build_run_records("tim file", "s1", [], "u")
        self.assertEqual([r["kind"] for r in records], ["episodic"])
        self.assertIn("Task: tim file", records[0]["text"])

    def test_the_final_answer_is_part_of_the_episodic_text(self) -> None:
        records = build_run_records("tim file", "s1", [
            {"type": "final_answer", "content": "da tim thay trong src/"},
        ], "u")
        self.assertIn("da tim thay trong src/", records[0]["text"])

    def test_tool_calls_are_recorded_in_the_episodic_metadata(self) -> None:
        records = build_run_records("tim file", "s1", [
            {"type": "tool_call", "tool": "rag_search", "params": {}},
            {"type": "tool_call", "tool": "read_file", "params": {}},
        ], "u")
        self.assertEqual(records[0]["metadata"]["tools"], ["rag_search", "read_file"])
        self.assertEqual(records[0]["metadata"]["session_id"], "s1")

    def test_a_non_mapping_step_is_ignored_not_fatal(self) -> None:
        records = build_run_records("tim file", "s1", ["garbage"], "u")  # type: ignore[list-item]
        self.assertEqual(len(records), 1)

    def test_a_mapping_step_of_another_type_is_ignored(self) -> None:
        """A tool observation is neither answer, plan nor call: the elif
        chain must fall through to the next step, not misfile it."""
        records = build_run_records("tim file", "s1", [
            {"type": "tool_result", "tool": "rag_search", "result": {}},
        ], "u")
        self.assertEqual(records[0]["metadata"]["tools"], [])

    def test_a_tool_call_without_a_usable_name_is_not_recorded(self) -> None:
        """Only a non-empty string identifies a tool; anything else would
        put garbage into the episodic metadata."""
        records = build_run_records("tim file", "s1", [
            {"type": "tool_call", "tool": ""},
            {"type": "tool_call", "tool": 42},
        ], "u")
        self.assertEqual(records[0]["metadata"]["tools"], [])

    def test_a_plan_step_adds_a_procedural_record_keyed_by_the_signature(self) -> None:
        records = build_run_records("tim file", "s1", [
            {"type": "plan", "steps": ["doc thu muc", "tim ten file"]},
        ], DEFAULT_MEMORY_OWNER)
        procedural = [r for r in records if r["kind"] == "procedural"]
        self.assertEqual(len(procedural), 1)
        self.assertEqual(procedural[0]["id"],
                         f"proc-{task_signature('tim file')[:24]}")
        self.assertEqual(procedural[0]["text"], "doc thu muc | tim ten file")

    def test_a_non_default_owner_gets_an_owner_namespaced_id(self) -> None:
        """The id is the store's primary key and a rerun replaces on it.
        Without the owner in the id, two tenants running the same task would
        overwrite each other's procedural row (ADR 0043)."""
        records = build_run_records("tim file", "s1", [
            {"type": "plan", "steps": ["doc thu muc"]},
        ], "tenant-a")
        procedural = next(r for r in records if r["kind"] == "procedural")
        self.assertNotEqual(procedural["id"], f"proc-{task_signature('tim file')[:24]}")
        self.assertTrue(procedural["id"].startswith("proc-"))
        self.assertEqual(procedural["owner"], "tenant-a")

    def test_two_owners_running_the_same_task_share_no_record_id(self) -> None:
        """Every kind is namespaced: episodic, procedural and semantic rows
        written by different owners must be distinct primary keys even when
        the task, the plan and the fact are identical."""
        steps = [
            {"type": "plan", "steps": ["doc thu muc", "tim ten file"]},
            {"type": "final_answer", "content": "Toi thich mau xanh"},
        ]
        alice = build_run_records("tim file", "s1", steps, "alice", now=NOW)
        bob = build_run_records("tim file", "s2", steps, "bob", now=NOW)
        alice_ids = {record["id"] for record in alice}
        bob_ids = {record["id"] for record in bob}
        self.assertEqual(len(alice_ids), 3)
        self.assertEqual(len(bob_ids), 3)
        self.assertEqual(alice_ids & bob_ids, set())

    def test_the_default_owner_keeps_the_historical_episodic_id(self) -> None:
        """Rows written before ownership existed must remain addressable."""
        records = build_run_records("tim file", "s1", [], DEFAULT_MEMORY_OWNER,
                                    now=NOW)
        self.assertEqual(records[0]["id"],
                         f"ep-{task_signature('tim file')[:16]}-{int(NOW)}")

    def test_a_plan_step_without_a_list_adds_no_procedural_record(self) -> None:
        records = build_run_records("tim file", "s1", [
            {"type": "plan", "steps": "not-a-list"},
        ], "u")
        self.assertEqual([r["kind"] for r in records], ["episodic"])

    def test_preference_statements_become_semantic_records(self) -> None:
        records = build_run_records("Toi thich mau xanh", "s1", [], "u")
        semantic = [r for r in records if r["kind"] == "semantic"]
        self.assertEqual(len(semantic), 1)
        self.assertEqual(semantic[0]["text"], "mau xanh")

    def test_a_preference_in_the_final_answer_is_also_extracted(self) -> None:
        records = build_run_records("hoi mau", "s1", [
            {"type": "final_answer", "content": "Toi thich mau xanh"},
        ], "u")
        self.assertIn("mau xanh", [r["text"] for r in records if r["kind"] == "semantic"])

    def test_semantic_records_are_capped_per_run(self) -> None:
        task = ("Toi thich mau xanh. Toi thich cafe. "
                "Toi thich doc sach. Toi thich am nhac.")
        records = build_run_records(task, "s1", [], "u")
        semantic = [r for r in records if r["kind"] == "semantic"]
        self.assertEqual(len(semantic), MAX_SEMANTIC_PER_RUN)

    def test_the_same_fact_twice_keeps_one_record_id(self) -> None:
        """The id hashes the fact, so a repeated preference replaces instead
        of piling up (the store's ``INSERT OR REPLACE`` contract)."""
        first = build_run_records("Toi thich mau xanh", "s1", [], "u")
        second = build_run_records("Toi thich mau xanh", "s2", [], "u")
        sem_first = next(r for r in first if r["kind"] == "semantic")
        sem_second = next(r for r in second if r["kind"] == "semantic")
        self.assertEqual(sem_first["id"], sem_second["id"])

    def test_an_explicit_now_pins_every_timestamp(self) -> None:
        records = build_run_records("tim file", "s1", [], "u", now=NOW)
        self.assertTrue(all(r["created_at"] == NOW for r in records))


class FormatMemoryBlockTests(unittest.TestCase):
    def test_no_records_renders_the_empty_string(self) -> None:
        """An empty ``[Long-term memory]`` header would lie about memory."""
        self.assertEqual(format_memory_block([]), "")

    def test_records_render_as_a_kind_tagged_block(self) -> None:
        block = format_memory_block([_record(kind="semantic", text="thich mau xanh")])
        self.assertEqual(block, "[Long-term memory]\n- [semantic] thich mau xanh")

    def test_a_sparse_record_still_renders(self) -> None:
        block = format_memory_block([{}])
        self.assertEqual(block, "[Long-term memory]\n- [] ")


if __name__ == "__main__":
    unittest.main()
