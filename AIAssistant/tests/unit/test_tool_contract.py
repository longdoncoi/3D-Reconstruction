"""Unit tests for ``ai_assistant.tools.tool_contract``.

The contract layer maps plain tool dictionaries onto ``ToolDefinition`` policy
records and renders the JSON-schema/OpenAI/grammar variants used by the agent.
These tests exercise the mapping and override tables without a running model.
"""
from __future__ import annotations

import json
import unittest

from ai_assistant.tools.tool_contract import (
    _to_tool_specs,
    build_tool_models,
    enrich_tool_definitions,
    grammar_schema,
    json_schema,
    openai_tools,
)


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
        self.assertTrue(spec.idempotent)

    def test_write_file_override(self):
        spec = _to_tool_specs([_tool("write_file")])[0]
        self.assertEqual(spec.policy, "code_write")
        self.assertTrue(spec.requires_approval)

    def test_run_command_override(self):
        spec = _to_tool_specs([_tool("run_command")])[0]
        self.assertEqual(spec.timeout_seconds, 120)
        self.assertEqual(spec.policy, "code_execute")
        self.assertTrue(spec.requires_approval)


class JsonSchemaTests(unittest.TestCase):
    def test_renders_required_and_constraints(self):
        schema = json_schema(_tool())
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], ["path"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["limit"], {"type": "integer", "minimum": 1, "maximum": 100})
        # "required" is not part of the rendered property definition.
        self.assertNotIn("required", schema["properties"]["path"])

    def test_rejects_missing_name(self):
        with self.assertRaises(KeyError):
            json_schema({"description": "x", "parameters": {}})


class EnrichTests(unittest.TestCase):
    def test_adds_contract_and_schema_fields(self):
        enriched = enrich_tool_definitions([_tool()])[0]
        for key in ("timeout_seconds", "policy", "requires_approval", "idempotent", "schema"):
            self.assertIn(key, enriched)
        self.assertEqual(enriched["schema"]["type"], "object")

    def test_application_action_gets_action_enum(self):
        tool = _tool(
            "application_action",
            parameters={"action": {"type": "string", "description": "Action"}},
        )
        enriched = enrich_tool_definitions([tool])[0]
        self.assertIsInstance(enriched["parameters"]["action"]["enum"], list)

    def test_non_action_tool_has_no_enum_injection(self):
        enriched = enrich_tool_definitions([_tool()])[0]
        self.assertNotIn("enum", enriched["parameters"]["path"])


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


class OpenAiToolsTests(unittest.TestCase):
    def test_openai_format(self):
        tools = openai_tools([_tool()])
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "read_file")
        self.assertEqual(tools[0]["function"]["parameters"]["type"], "object")


class GrammarSchemaTests(unittest.TestCase):
    def test_grammar_contains_variants(self):
        payload = json.loads(grammar_schema([_tool()]))
        self.assertIn("oneOf", payload)
        kinds = [variant["properties"]["kind"]["const"] for variant in payload["oneOf"]]
        self.assertIn("final", kinds)
        self.assertIn("tool", kinds)


if __name__ == "__main__":
    unittest.main()
