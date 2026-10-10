"""End-to-end regression coverage for generic Coding Agent completion rules."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_assistant.orchestration.supervisor import delegate, reflect_result, verify_result
from LangGraphAgent import LocalAgentGraph


class CodingAgentCompletionTests(unittest.TestCase):
    @staticmethod
    def _verifier(task: str, tool: str, params: dict, result: dict) -> dict:
        assignment = delegate(task, "completion-test", tool, params, prefer_code=True)
        return verify_result(assignment, result)

    def _run_task(self, task: str, actions: list[str], results: dict[str, dict]) -> dict:
        responses = iter([*actions, "DONE"])

        def parse(token: str) -> tuple[str | None, dict | None]:
            tool_by_token = {
                "READ": ("read_file", {"path": "src/example.cpp"}),
                "PATCH": ("patch_file", {"path": "src/example.cpp"}),
                "DIFF": ("git_diff", {"path": "src/example.cpp"}),
                "CHECK": ("run_command", {"command": "ctest -R focused"}),
                "EMPTY": ("search_text", {"query": "missing", "path": "src"}),
            }
            return tool_by_token.get(token, (None, None))

        def execute(tool: str, _params: dict) -> dict:
            return results[tool]

        def verify(tool: str, params: dict, result: dict) -> dict:
            return self._verifier(task, tool, params, result)

        def reflect(tool: str, params: dict, result: dict, verification: dict) -> dict:
            assignment = delegate(task, "completion-test", tool, params, prefer_code=True)
            return reflect_result(assignment, result, verification)

        graph = LocalAgentGraph(
            complete=lambda _messages, _temperature: next(responses),
            parse=parse,
            execute=execute,
            needs_approval=lambda _tool: False,
            max_iterations=len(actions) + 2,
            verify_result=verify,
            reflect_result=reflect,
            plan_complete=lambda _messages, _temperature: (
                '{"requires_plan": true, "plan": ['
                '"inspect source", "implement approved change", '
                '"review the resulting diff", "verify the result", "report"]}'
            ),
            reflect_complete=lambda _messages, _temperature: (
                '{"passed": true, "decision": "continue", "reason": "evidence accepted"}'
            ),
        )
        return graph.run(
            [{"role": "system", "content": "test"}, {"role": "user", "content": task}],
            "completion-test-" + str(abs(hash(task))),
            0.1,
            enforce_plan_completion=True,
        )

    def test_empty_discovery_is_rejected_and_agent_recovers_with_source_evidence(self) -> None:
        task = "Phân tích và giải thích chức năng của hàm example trong src/example.cpp"
        state = self._run_task(
            task,
            ["EMPTY", "READ"],
            {
                "search_text": {"query": "missing", "count": 0, "truncated": False, "results": []},
                "read_file": {"path": "src/example.cpp", "content": "void example() {}"},
            },
        )
        reflections = [step["result"] for step in state["steps"] if step.get("type") == "reflection"]
        self.assertFalse(reflections[0]["passed"])
        self.assertTrue(reflections[-1]["passed"])
        self.assertEqual(state["steps"][-1]["type"], "final_answer")

    def test_requested_coding_tasks_complete_from_observed_lifecycle_evidence(self) -> None:
        tasks = (
            ("Phân tích và giải thích chức năng của hàm _build_code_citation_result trong "
             "src/ai_assistant/adapters/legacy_agent.py", ["READ"]),
            ("Fix lỗi khi tải ảnh DICOM trong project, bổ sung unit test và kiểm tra build.",
             ["READ", "PATCH", "DIFF", "CHECK"]),
            ("Bổ sung chức năng export kết quả reconstruction sang file OBJ.",
             ["READ", "PATCH", "DIFF", "CHECK"]),
        )
        results = {
            "read_file": {"path": "src/example.cpp", "content": "void example() {}"},
            "patch_file": {"success": True, "path": "src/example.cpp", "replacements": 1},
            "git_diff": {"path": "src/example.cpp", "content": "+ changed source"},
            "run_command": {"return_code": 0, "stdout": "all checks passed"},
        }
        for task, actions in tasks:
            with self.subTest(task=task):
                state = self._run_task(task, actions, results)
                self.assertEqual(state["steps"][-1]["type"], "final_answer")
                self.assertFalse(any(step.get("type") == "coding_incomplete" for step in state["steps"]))
                if actions == ["READ"]:
                    self.assertEqual(state["steps"][-1]["content"], "DONE")


if __name__ == "__main__":
    unittest.main()
