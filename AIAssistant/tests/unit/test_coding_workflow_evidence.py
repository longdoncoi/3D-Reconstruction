"""Guard-path coverage for the coding-workflow evidence classifier.

``specialists/code.py`` exists to stop the Reasoner from declaring a change
done before a diff and a verification command were actually observed. Its
reject arms — a non-mapping result, an explicit ``success: false``, a non-zero
exit code, a discovery result with no usable evidence — are the whole point of
the module, yet only the happy paths had tests. Each workflow kind's missing
list and both branches of the guidance formatter are pinned here too.
"""
from __future__ import annotations

import unittest

from ai_assistant.domain.tools import (
    tool_result_has_evidence as _result_has_evidence,
)
from ai_assistant.domain.tools import (
    tool_result_succeeded as _result_is_successful,
)
from ai_assistant.orchestration.specialists.code import (
    coding_workflow_guidance,
    coding_workflow_kind,
    coding_workflow_status,
)


def _result_step(tool: str, result: dict) -> dict:
    return {"type": "tool_result", "tool": tool, "result": result}


class ResultContractTests(unittest.TestCase):
    """A tool result only counts as progress if it really succeeded."""

    def test_a_non_mapping_result_is_never_successful(self):
        self.assertFalse(_result_is_successful("done"))
        self.assertFalse(_result_is_successful(None))
        self.assertFalse(_result_is_successful(0))

    def test_an_error_payload_is_not_successful(self):
        self.assertFalse(_result_is_successful({"error": "File not found"}))

    def test_an_explicit_success_flag_of_false_is_not_successful(self):
        """Absence of ``error`` is not enough; ``success: false`` is a failure."""
        self.assertFalse(_result_is_successful({"success": False}))

    def test_a_non_zero_exit_code_is_not_successful(self):
        """A command can exit without ``error`` and still have failed."""
        self.assertFalse(_result_is_successful({"return_code": 1, "stdout": "boom"}))

    def test_a_zero_exit_code_is_successful(self):
        self.assertTrue(_result_is_successful({"return_code": 0, "stdout": "ok"}))
        self.assertTrue(_result_is_successful({"content": "read"}))

    def test_a_failed_result_never_carries_evidence(self):
        self.assertFalse(_result_has_evidence("read_file", "not a mapping"))
        self.assertFalse(_result_has_evidence("read_file", {"error": "boom"}))
        self.assertFalse(_result_has_evidence("read_file", {"content": ""}))

    def test_blank_string_content_is_not_evidence(self):
        """``" "`` must not masquerade as a read result (shared table, #24/7)."""
        self.assertFalse(_result_has_evidence("read_file", {"content": "  \n"}))

    def test_an_unmapped_tool_is_trusted_on_success_alone(self):
        self.assertTrue(_result_has_evidence("run_command", {"success": True}))

    def test_git_diff_evidence_is_its_content_field(self):
        """``git_diff`` is listed in the shared table on both call sites."""
        self.assertFalse(_result_has_evidence("git_diff", {"success": True}))
        self.assertTrue(
            _result_has_evidence("git_diff", {"content": "diff --git a/x b/x"}),
        )

    def test_analyse_code_evidence_needs_one_of_its_three_fields(self):
        evidence = _result_has_evidence(
            "analyze_code", {"functions": ["main"], "classes": [], "includes": []},
        )
        self.assertTrue(evidence)
        self.assertFalse(
            _result_has_evidence(
                "analyze_code", {"functions": [], "classes": [], "includes": []},
            )
        )


class WorkflowKindTests(unittest.TestCase):
    def test_a_compile_request_is_a_verification_task(self):
        self.assertEqual(coding_workflow_kind("compile the project"), "verify")

    def test_a_patch_request_is_a_change_task(self):
        self.assertEqual(coding_workflow_kind("fix the null pointer bug"), "change")

    def test_a_request_without_evidence_verbs_is_investigation(self):
        self.assertEqual(coding_workflow_kind("what happens on startup"), "investigate")


class MissingStageTests(unittest.TestCase):
    def test_a_verification_task_without_evidence_is_incomplete_twice(self):
        status = coding_workflow_status("compile the project", [])
        self.assertEqual(status.kind, "verify")
        self.assertEqual(
            status.missing, ("project_discovery", "verification_command"),
        )

    def test_an_investigation_with_no_discovery_is_incomplete(self):
        status = coding_workflow_status("what happens on startup", [])
        self.assertEqual(status.missing, ("project_discovery",))

    def test_an_investigation_that_inspected_files_is_complete(self):
        """The ``elif`` guard is False here, so nothing is reported missing."""
        steps = [
            _result_step("read_file", {"content": "# header"}),
            _result_step("list_directory", {"entries": ["a.py", "b.py"]}),
        ]
        status = coding_workflow_status("what happens on startup", steps)
        self.assertTrue(status.source_evidence)
        self.assertEqual(status.missing, ())

    def test_an_analysis_with_source_evidence_is_complete(self):
        """Only the fully-evidenced arm reaches the no-work-required message."""
        status = coding_workflow_status(
            "phân tích hàm này", [_result_step("read_file", {"content": "x"})],
        )
        self.assertEqual(status.kind, "analysis")
        self.assertEqual(status.missing, ())

    def test_an_analysis_without_source_evidence_asks_for_it(self):
        status = coding_workflow_status("phân tích hàm này", [])
        self.assertEqual(status.missing, ("source_evidence",))

    def test_a_change_without_a_mutation_asks_for_one(self):
        status = coding_workflow_status(
            "add a --verbose flag", [_result_step("read_file", {"content": "x"})],
        )
        self.assertEqual(status.kind, "change")
        self.assertIn("approved_change", status.missing)

    def test_non_result_steps_and_failed_results_are_ignored(self):
        """``tool_call`` entries and failed results must not become evidence."""
        steps = [
            {"type": "tool_call", "tool": "read_file", "params": {"path": "a"}},
            _result_step("read_file", {"error": "boom"}),
            _result_step("read_file", {"content": "real evidence"}),
        ]
        status = coding_workflow_status("what happens on startup", steps)
        self.assertTrue(status.source_evidence)
        self.assertEqual(status.missing, ())

    def test_a_verify_task_that_already_discovered_and_ran_is_complete(self):
        steps = [
            _result_step("list_directory", {"entries": ["a.py"]}),
            _result_step("run_command", {"return_code": 0, "stdout": "ok"}),
        ]
        status = coding_workflow_status("compile the project", steps)
        self.assertEqual(status.kind, "verify")
        self.assertEqual(status.missing, ())
        self.assertTrue(status.execution_observed)


class GuidanceTests(unittest.TestCase):
    def test_a_complete_workflow_reports_no_next_stage(self):
        status = coding_workflow_status(
            "phân tích hàm này", [_result_step("read_file", {"content": "x"})],
        )
        self.assertEqual(
            coding_workflow_guidance(status),
            "The coding workflow has observable evidence for every required stage.",
        )

    def test_an_incomplete_workflow_names_the_first_missing_stage(self):
        status = coding_workflow_status("compile the project", [])
        guidance = coding_workflow_guidance(status)
        self.assertTrue(guidance.startswith("Next required coding stage: "))
        self.assertIn("discover the relevant project files", guidance)


if __name__ == "__main__":
    unittest.main()
