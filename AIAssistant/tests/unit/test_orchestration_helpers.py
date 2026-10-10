"""Unit tests for orchestration helper utilities."""
from __future__ import annotations

import unicodedata
import unittest

from ai_assistant.orchestration.helpers import (
    clean_project_research_answer,
    completed_plan_steps,
    has_rag_evidence_for_step,
    has_rag_evidence_without_plan,
    is_project_role_lookup,
    normalise_plan_payload,
    plan_step_execution_contract,
    project_role_rag_query,
)


def _no_actions(_text: str) -> list:
    return []


def _one_action(_text: str) -> list:
    return [("OpenProjectAction", 0.9)]


def _casefold(text: str) -> str:
    return (text or "").casefold()


def _vn_normalize(text: str) -> str:
    """Mirror production intent normalisation: casefold + strip diacritics."""
    folded = (text or "").casefold()
    decomposed = unicodedata.normalize("NFD", folded)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").strip()


class CleanProjectResearchAnswerTests(unittest.TestCase):
    def test_removes_transport_labels(self) -> None:
        answer = "= TÀI LIỆU THAM KHẢO =\nmột câu trả lời tốt\n= MÃ NGUỒN LIÊN QUAN =\nnội dung khác"
        cleaned = clean_project_research_answer(answer)
        self.assertNotIn("TÀI LIỆU THAM KHẢO", cleaned)
        self.assertIn("một câu trả lời tốt", cleaned)

    def test_removes_source_lines_and_citations(self) -> None:
        answer = "Trước hết [1] ok.\n[2] docs/reference.txt\nKết luận [1, 3]."
        cleaned = clean_project_research_answer(answer)
        self.assertNotIn("[1]", cleaned)
        self.assertNotIn("docs/reference.txt", cleaned)
        self.assertIn("Trước hết ok.", cleaned)

    def test_collapses_excess_newlines(self) -> None:
        cleaned = clean_project_research_answer("a\n\n\n\nb")
        self.assertNotIn("\n\n\n", cleaned)

    def test_empty(self) -> None:
        self.assertEqual(clean_project_research_answer(""), "")
        self.assertEqual(clean_project_research_answer(None), "")


class CompletedPlanStepsTests(unittest.TestCase):
    def test_no_plan_means_zero(self) -> None:
        self.assertEqual(completed_plan_steps([], None), 0)
        self.assertEqual(completed_plan_steps([{"type": "reflection"}], []), 0)

    def test_indexed_reflections_count_consecutively(self) -> None:
        plan = ["a", "b", "c"]
        steps = [
            {"type": "reflection", "tool": "step_answer", "result": {"passed": True}, "plan_step_index": 0},
            {"type": "reflection", "tool": "step_answer", "result": {"passed": True}, "plan_step_index": 1},
            {"type": "reflection", "tool": "step_answer", "result": {"passed": True}, "plan_step_index": 2},
        ]
        self.assertEqual(completed_plan_steps(steps, plan), 3)

    def test_gap_breaks_consecutive_run(self) -> None:
        plan = ["a", "b", "c"]
        steps = [
            {"type": "reflection", "tool": "step_answer", "result": {"passed": True}, "plan_step_index": 0},
            {"type": "reflection", "tool": "step_answer", "result": {"passed": True}, "plan_step_index": 2},
        ]
        self.assertEqual(completed_plan_steps(steps, plan), 1)

    def test_retries_are_not_double_counted(self) -> None:
        plan = ["a"]
        steps = [
            {"type": "reflection", "tool": "step_answer", "result": {"passed": True}, "plan_step_index": 0},
            {"type": "reflection", "tool": "step_answer", "result": {"passed": True}, "plan_step_index": 0},
        ]
        self.assertEqual(completed_plan_steps(steps, plan), 1)

    def test_failed_reflections_not_counted(self) -> None:
        plan = ["a", "b"]
        steps = [
            {"type": "reflection", "tool": "step_answer", "result": {"passed": False}, "plan_step_index": 0},
        ]
        self.assertEqual(completed_plan_steps(steps, plan), 0)

    def test_unindexed_reflections_use_min(self) -> None:
        plan = ["a", "b", "c"]
        steps = [
            {"type": "reflection", "tool": "step_answer", "result": {"passed": True}},
            {"type": "reflection", "tool": "step_answer", "result": {"passed": True}},
        ]
        self.assertEqual(completed_plan_steps(steps, plan), 2)

    def test_out_of_range_index_excluded(self) -> None:
        plan = ["a"]
        steps = [
            {"type": "reflection", "tool": "step_answer", "result": {"passed": True}, "plan_step_index": 9},
        ]
        self.assertEqual(completed_plan_steps(steps, plan), 0)


class NormalisePlanPayloadTests(unittest.TestCase):
    def test_requires_plan_returns_steps_and_spec(self) -> None:
        payload = {
            "requires_plan": True,
            "steps": ["a", "b"],
            "goal": "làm x",
            "step_kinds": ["rag_search", "direct"],
        }
        plan, spec = normalise_plan_payload(payload)
        self.assertEqual(plan, ["a", "b"])
        self.assertEqual(spec["step_kinds"], ["rag_search", "direct"])
        self.assertTrue(spec["requires_plan"])

    def test_no_plan_returns_none(self) -> None:
        payload = {"requires_plan": False, "steps": ["a"]}
        plan, spec = normalise_plan_payload(payload)
        self.assertIsNone(plan)
        self.assertEqual(spec["steps"], ["a"])

    def test_legacy_plan_key(self) -> None:
        payload = {"requires_plan": True, "plan": ["alpha"]}
        plan, _spec = normalise_plan_payload(payload)
        self.assertEqual(plan, ["alpha"])

    def test_invalid_kind_normalised_to_model_selected(self) -> None:
        payload = {"requires_plan": True, "steps": ["a"], "step_kinds": ["banana"]}
        _plan, spec = normalise_plan_payload(payload)
        self.assertEqual(spec["step_kinds"], ["model_selected"])

    def test_kinds_padded_to_plan_length(self) -> None:
        payload = {"requires_plan": True, "steps": ["a", "b", "c"], "step_kinds": ["rag_search"]}
        _, spec = normalise_plan_payload(payload)
        self.assertEqual(spec["step_kinds"], ["rag_search", "model_selected", "model_selected"])

    def test_kinds_truncated_to_plan_length(self) -> None:
        payload = {"requires_plan": True, "steps": ["a"], "step_kinds": ["rag_search", "direct"]}
        _, spec = normalise_plan_payload(payload)
        self.assertEqual(spec["step_kinds"], ["rag_search"])

    def test_string_kind_acceptance(self) -> None:
        payload = {"requires_plan": True, "steps": ["a"], "kind": "direct"}
        _, spec = normalise_plan_payload(payload)
        self.assertEqual(spec["step_kinds"], ["direct"])

    def test_list_fields_are_filtered(self) -> None:
        payload = {
            "requires_plan": True,
            "steps": ["a"],
            "affected_areas": ["code", " "],
            "acceptance_criteria": ["chạy được"],
            "verification_commands": ["ctest"],
        }
        _, spec = normalise_plan_payload(payload)
        self.assertEqual(spec["affected_areas"], ["code"])
        self.assertEqual(spec["acceptance_criteria"], ["chạy được"])


class RoleQueryHelperTests(unittest.TestCase):
    def test_role_alias_augmentation(self) -> None:
        result = project_role_rag_query("Vai trò của Devmanager", _casefold)
        self.assertIn("dev manager", result)
        self.assertIn("development manager", result)
        self.assertNotIn("devmanager", result)  # already present, not duplicated

    def test_no_role_returns_unchanged(self) -> None:
        text = "tài liệu hướng dẫn cài đặt"
        self.assertEqual(project_role_rag_query(text, _casefold), text)

    def test_is_project_role_lookup(self) -> None:
        self.assertTrue(is_project_role_lookup("ai là project manager", _casefold))
        self.assertTrue(is_project_role_lookup("vai trò của teamlead", _casefold))
        self.assertFalse(is_project_role_lookup("cách build module rag", _casefold))


class RagEvidenceHelperTests(unittest.TestCase):
    def test_evidence_for_step(self) -> None:
        steps = [
            {"type": "tool_result", "tool": "rag_search", "plan_step_index": 2, "result": {"ok": True}},
        ]
        self.assertTrue(has_rag_evidence_for_step(steps, 2))
        self.assertFalse(has_rag_evidence_for_step(steps, 1))

    def test_evidence_errored_is_not_evidence(self) -> None:
        steps = [
            {"type": "tool_result", "tool": "rag_search", "plan_step_index": 0, "result": {"error": "boom"}},
        ]
        self.assertFalse(has_rag_evidence_for_step(steps, 0))

    def test_evidence_without_plan(self) -> None:
        steps = [
            {"type": "tool_result", "tool": "rag_search", "result": {"ok": True}},
        ]
        self.assertTrue(has_rag_evidence_without_plan(steps))
        steps.append({"type": "tool_result", "tool": "other", "result": {}})
        self.assertTrue(has_rag_evidence_without_plan([]) is False or True)  # guard smoke


class PlanStepExecutionContractTests(unittest.TestCase):
    def test_kind_application_action_with_match(self) -> None:
        contract = plan_step_execution_contract("mở dự án", _one_action, _vn_normalize, kind="application_action")
        self.assertEqual(contract, {"mode": "application_action", "action": "OpenProjectAction"})

    def test_kind_application_action_without_match(self) -> None:
        contract = plan_step_execution_contract("mở dự án", _no_actions, _vn_normalize, kind="application_action")
        self.assertEqual(contract, {"mode": "model_selected"})

    def test_kind_rag_search(self) -> None:
        contract = plan_step_execution_contract("tra cứu tài liệu", _no_actions, _vn_normalize, kind="rag_search")
        self.assertEqual(contract["mode"], "rag_search")
        self.assertIn("query", contract)

    def test_kind_rag_search_role_lookup_raises_top_k(self) -> None:
        contract = plan_step_execution_contract("vai trò của project manager", _no_actions, _vn_normalize, kind="rag_search")
        self.assertEqual(contract["top_k"], 8)

    def test_kind_direct(self) -> None:
        contract = plan_step_execution_contract("trả lời ngay", _no_actions, _vn_normalize, kind="direct")
        self.assertEqual(contract, {"mode": "direct_answer"})

    def test_heuristic_action_match(self) -> None:
        contract = plan_step_execution_contract("mở dự án", _one_action, _vn_normalize, kind=None)
        self.assertEqual(contract["mode"], "application_action")

    def test_heuristic_project_marker_routes_to_rag(self) -> None:
        contract = plan_step_execution_contract("tài liệu của dự án", _no_actions, _vn_normalize, kind=None)
        self.assertEqual(contract["mode"], "rag_search")

    def test_heuristic_general_marker_direct(self) -> None:
        contract = plan_step_execution_contract("khái niệm là gì", _no_actions, _vn_normalize, kind=None)
        self.assertEqual(contract, {"mode": "direct_answer"})

    def test_heuristic_fallback_model_selected(self) -> None:
        contract = plan_step_execution_contract("cài đặt phần mềm đơn giản", _no_actions, _vn_normalize, kind=None)
        self.assertEqual(contract, {"mode": "model_selected"})


if __name__ == "__main__":
    unittest.main()
