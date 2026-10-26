"""Validation of planner-declared parallel groups and the fan-out bound.

ADR 0044 §2/§4: the planner may only *propose* a group; the parser validates
it fail-closed (a broken group is dropped, never repaired) and the width is
clamped at the settings boundary. These tests drive every refusal arm of
:func:`valid_parallel_group`, both arms of the group parser, and the clamp.
"""
from __future__ import annotations

import unittest

from ai_assistant.domain.parallel import (
    PARALLEL_MAX_DEFAULT,
    PARALLEL_MAX_HARD_CAP,
    clamp_parallel_max,
)
from ai_assistant.orchestration.helpers import (
    normalise_plan_payload,
    valid_parallel_group,
)


class ValidParallelGroupTests(unittest.TestCase):
    def test_two_ascending_indexes_inside_the_plan_are_valid(self) -> None:
        self.assertEqual(valid_parallel_group([0, 2], 3), [0, 2])

    def test_a_whole_plan_group_is_valid(self) -> None:
        self.assertEqual(valid_parallel_group([0, 1, 2, 3], 4), [0, 1, 2, 3])

    def test_a_non_list_is_refused(self) -> None:
        for raw in ("0,1", 1, {"first": 0}, None, (0, 1)):
            with self.subTest(raw=raw):
                self.assertIsNone(valid_parallel_group(raw, 3))

    def test_a_single_member_group_is_refused(self) -> None:
        self.assertIsNone(valid_parallel_group([], 3))
        self.assertIsNone(valid_parallel_group([1], 3))

    def test_a_non_integer_member_is_refused(self) -> None:
        self.assertIsNone(valid_parallel_group([0, "1"], 3))
        self.assertIsNone(valid_parallel_group([0, 1.0], 3))

    def test_a_boolean_member_is_refused(self) -> None:
        # ``True`` is an ``int`` in Python; a JSON ``true`` is not an index.
        self.assertIsNone(valid_parallel_group([0, True], 3))

    def test_a_negative_index_is_refused(self) -> None:
        self.assertIsNone(valid_parallel_group([-1, 1], 3))

    def test_an_index_at_or_past_the_plan_is_refused(self) -> None:
        self.assertIsNone(valid_parallel_group([0, 3], 3))

    def test_a_descending_group_is_refused(self) -> None:
        self.assertIsNone(valid_parallel_group([2, 1], 3))

    def test_a_duplicate_index_is_refused(self) -> None:
        self.assertIsNone(valid_parallel_group([1, 1], 3))


class NormalisePlanParallelGroupsTests(unittest.TestCase):
    def test_a_valid_group_survives_normalisation(self) -> None:
        payload = {
            "requires_plan": True,
            "steps": ["a", "b", "c"],
            "step_kinds": ["rag_search"] * 3,
            "parallel_groups": [[0, 2]],
        }
        _plan, spec = normalise_plan_payload(payload)
        self.assertEqual(spec["parallel_groups"], [[0, 2]])

    def test_an_invalid_group_is_dropped_while_a_valid_one_survives(self) -> None:
        payload = {
            "requires_plan": True,
            "steps": ["a", "b", "c"],
            "parallel_groups": [[0, 1], [1], [0, 9], "0,1"],
        }
        _plan, spec = normalise_plan_payload(payload)
        self.assertEqual(spec["parallel_groups"], [[0, 1]])

    def test_an_absent_group_table_means_no_groups(self) -> None:
        _plan, spec = normalise_plan_payload(
            {"requires_plan": True, "steps": ["a"]},
        )
        self.assertEqual(spec["parallel_groups"], [])

    def test_a_non_list_group_table_means_no_groups(self) -> None:
        _plan, spec = normalise_plan_payload({
            "requires_plan": True, "steps": ["a", "b"],
            "parallel_groups": "0,1",
        })
        self.assertEqual(spec["parallel_groups"], [])

    def test_a_group_reaching_past_the_shortened_plan_is_dropped(self) -> None:
        """The group indexes are validated against the plan as parsed, not as
        the planner intended it."""
        payload = {
            "requires_plan": True,
            "steps": ["a", "", "c"],
            "parallel_groups": [[0, 2]],
        }
        plan, spec = normalise_plan_payload(payload)
        self.assertEqual(plan, ["a", "c"])
        self.assertEqual(spec["parallel_groups"], [])


class ClampParallelMaxTests(unittest.TestCase):
    def test_the_documented_bounds(self) -> None:
        self.assertEqual(PARALLEL_MAX_DEFAULT, 4)
        self.assertEqual(PARALLEL_MAX_HARD_CAP, 8)

    def test_a_zero_or_negative_width_is_raised_to_one(self) -> None:
        self.assertEqual(clamp_parallel_max(0), 1)
        self.assertEqual(clamp_parallel_max(-3), 1)

    def test_a_width_above_the_hard_cap_is_capped(self) -> None:
        self.assertEqual(clamp_parallel_max(PARALLEL_MAX_HARD_CAP + 1), PARALLEL_MAX_HARD_CAP)

    def test_a_width_inside_the_bounds_is_kept(self) -> None:
        self.assertEqual(clamp_parallel_max(3), 3)


if __name__ == "__main__":
    unittest.main()
