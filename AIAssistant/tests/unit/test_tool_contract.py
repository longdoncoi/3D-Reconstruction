"""Unit tests for ``ai_assistant.tools.tool_contract``.

The contract layer maps plain tool dictionaries onto ``ToolDefinition`` policy
records and renders the pydantic model variants used by the agent. These tests
exercise the mapping and override tables without a running model. The JSON
schema/OpenAI/grammar renderings now live on ``ToolDefinition``/``tools.schema``
and are pinned through ``test_tool_registry``.
"""
from __future__ import annotations

import unittest

from ai_assistant.tools.tool_contract import _to_tool_specs, build_tool_models


def _tool(name: str = "read_file", **overrides) -> dict:
    tool = {
        "name": name,
        "description": "Read a file",
        "parameters": {
            "path": {"type": "string", "description": "Path", "required": True},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
    }
    tool.update(overrides)
    return tool


class ToolSpecMappingTests(unittest.TestCase):
    def test_default_contract_values(self):
        spec = _to_tool_specs([_tool()])[0]
        self.assertEqual(spec.timeout_seconds, 10)
        self.assertEqual(spec.policy, "read_only")
        self.assertFalse(spec.requires_approval)

    def test_write_file_override(self):
        spec = _to_tool_specs([_tool("write_file")])[0]
        self.assertEqual(spec.policy, "code_write")
        self.assertTrue(spec.requires_approval)

    def test_run_command_override(self):
        spec = _to_tool_specs([_tool("run_command")])[0]
        self.assertEqual(spec.timeout_seconds, 120)
        self.assertEqual(spec.policy, "code_execute")
        self.assertTrue(spec.requires_approval)


class BuildToolModelsTests(unittest.TestCase):
    def test_builds_pydantic_model_with_strict_validation(self):
        models = build_tool_models([_tool()])
        model = models["read_file"]
        instance = model(path="some/file.txt", limit=5)
        self.assertEqual(instance.path, "some/file.txt")
        with self.assertRaises(Exception):
            model(path="some/file.txt", unknown=1)

    def test_unknown_parameter_type_defaults_to_any(self):
        models = build_tool_models([_tool(parameters={"blob": {"type": "weird"}})])
        self.assertIsNotNone(models["read_file"])


if __name__ == "__main__":
    unittest.main()
