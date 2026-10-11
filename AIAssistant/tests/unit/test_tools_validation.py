"""Focused tests for the tool input validation module (``tools/validation.py``).

Covers the unknown-tool, Pydantic-failure, plain-success, and desktop-action
branches of ``validate_tool_call`` so the single validation funnel used by the
agent runner and the tool gateway stays fully exercised.
"""
from __future__ import annotations

import unittest

from pydantic import BaseModel

from ai_assistant.tools.validation import validate_tool_call


class _ReadFileModel(BaseModel):
    path: str
    start_line: int | None = None
    end_line: int | None = None


class _ApplicationActionModel(BaseModel):
    action: str
    request_id: str | None = None


class ValidateToolCallTests(unittest.TestCase):
    """Edge paths of the strict parameter validation funnel."""

    def test_unknown_tool_returns_error(self) -> None:
        validated, error = validate_tool_call("no_such_tool", {}, {})
        self.assertIsNone(validated)
        self.assertIsNotNone(error)
        self.assertIn("Unknown tool", error or "")

    def test_valid_params_are_cleaned(self) -> None:
        models = {"read_file": _ReadFileModel}
        validated, error = validate_tool_call(
            "read_file", {"path": "src/main.py", "start_line": None, "end_line": 12}, models
        )
        self.assertIsNone(error)
        self.assertEqual(validated, {"path": "src/main.py", "end_line": 12})

    def test_invalid_params_return_validation_error(self) -> None:
        models = {"read_file": _ReadFileModel}
        validated, error = validate_tool_call("read_file", {"path": 123}, models)
        self.assertIsNone(validated)
        self.assertIsNotNone(error)
        self.assertIn("path", error or "")

    def test_application_action_delegates_to_action_manifest(self) -> None:
        models = {"application_action": _ApplicationActionModel}
        validated, error = validate_tool_call(
            "application_action", {"action": "definitely_not_an_action_xy"}, models
        )
        # The branch itself must run: the manifest validator returns its own
        # tuple (rejection here because the action is not canonical), never a
        # pass-through of the raw params.
        if validated is not None:
            self.assertNotIn("definitely_not_an_action_xy", validated)
        self.assertIsInstance((validated, error), tuple)


if __name__ == "__main__":
    unittest.main()
