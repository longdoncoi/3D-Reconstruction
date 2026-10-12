"""Cache and dynamic-enum coverage for the tool registry.

``tools/registry.py`` is the single catalogue the prompt, grammar and OpenAI
tool-call layers all render from. Two properties need pinning:

* the compiled ``models``/``grammar`` are memoised and invalidated by
  registration — a stale grammar would advertise a tool that no longer exists;
* ``application_action`` is *dynamic*: its enum and description are injected
  from the action manifest at render time, so the registry must never ship the
  frozen enum that happens to be in the static definition.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_assistant.tools.definition import ToolDefinition
from ai_assistant.tools.registry import ToolRegistry


def _tool(name: str, **extra) -> ToolDefinition:
    parameters = extra.pop("parameters", {"path": {"type": "string"}})
    description = extra.pop("description", f"the {name} tool")
    return ToolDefinition(name=name, description=description,
                          parameters=parameters, **extra)


class RegistrationTests(unittest.TestCase):
    def test_register_makes_the_tool_reachable_by_name(self):
        registry = ToolRegistry()
        registry.register(_tool("read_file"))
        self.assertIsNotNone(registry.get("read_file"))
        self.assertIsNone(registry.get("write_file"))
        self.assertEqual(registry.names(), ["read_file"])
        self.assertEqual([t.name for t in registry.get_all()], ["read_file"])

    def test_register_all_registers_every_tool_in_order(self):
        registry = ToolRegistry()
        registry.register_all([_tool("a"), _tool("b")])
        self.assertEqual(registry.names(), ["a", "b"])

    def test_reregistering_a_name_replaces_the_definition(self):
        registry = ToolRegistry()
        registry.register(_tool("read_file", description="first"))
        registry.register(_tool("read_file", description="second"))
        self.assertEqual(registry.get("read_file").description, "second")
        self.assertEqual(len(registry.get_all()), 1)

    def test_registration_invalidates_the_cached_grammar(self):
        """The memoised grammar must not outlive the tool it describes."""
        registry = ToolRegistry()
        registry.register(_tool("a"))
        self.assertIn("a", registry.grammar)
        registry.register(_tool("b"))
        self.assertIn("b", registry.grammar)


class ModelAndGrammarCacheTests(unittest.TestCase):
    def test_models_are_compiled_once_and_reused(self):
        registry = ToolRegistry()
        registry.register(_tool("read_file"))
        first = registry.models
        self.assertIs(registry.models, first)
        # ``build_tool_models`` keys the compiled classes by *tool name*.
        self.assertIn("read_file", first)

    def test_models_are_recompiled_after_a_registration(self):
        registry = ToolRegistry()
        registry.register(_tool("a"))
        registry.models
        registry.register(_tool("b"))
        self.assertIn("b", registry.models)

    def test_grammar_is_built_once_and_reused(self):
        registry = ToolRegistry()
        registry.register(_tool("read_file"))
        self.assertIs(registry.grammar, registry.grammar)

    def test_an_empty_registry_yields_an_empty_catalogue(self):
        registry = ToolRegistry()
        self.assertEqual(registry.models, {})
        self.assertEqual(registry.names(), [])


class DynamicActionEnumTests(unittest.TestCase):
    """``application_action`` is rendered from the live manifest, not the schema."""

    @staticmethod
    def _with_action(enum: list[str], description: str = "static description") -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(_tool(
            "application_action",
            description=description,
            parameters={"action": {"type": "string", "enum": list(enum)}},
        ))
        return registry

    def test_models_compile_the_dynamic_action_tool(self):
        """The special case must at minimum yield a usable model.

        ``build_tool_models`` reads only ``type``/``required``/``minimum``/
        ``maximum``, so the injected enum is not part of the annotation — the
        observable contract is that the tool still compiles cleanly and the
        stored definition is left alone.
        """
        from ai_assistant.tools.action_manifest import action_ids
        registry = self._with_action(["stale_action"])
        model = registry.models["application_action"]
        self.assertIn("action", model.model_fields)
        self.assertTrue(action_ids())
        self.assertEqual(registry.get("application_action").parameters["action"]["enum"],
                         ["stale_action"])

    def test_openai_tools_inject_the_manifest_enum(self):
        from ai_assistant.tools.action_manifest import action_ids
        registry = self._with_action(["stale_action"])
        rendered = registry.get_openai_tools()[0]["function"]
        self.assertEqual(rendered["parameters"]["properties"]["action"]["enum"],
                         sorted(action_ids()))

    def test_openai_tools_appends_the_catalog_to_the_description(self):
        from ai_assistant.tools.action_manifest import action_catalog
        registry = self._with_action(["x"], description="static description")
        rendered = registry.get_openai_tools()[0]["function"]
        self.assertTrue(rendered["description"].startswith("static description"))
        self.assertIn(action_catalog(), rendered["description"])

    def test_a_static_tool_is_passed_through_untouched(self):
        registry = ToolRegistry()
        registry.register(_tool("read_file", description="plain"))
        rendered = registry.get_openai_tools()[0]["function"]
        self.assertEqual(rendered["description"], "plain")
        self.assertEqual(rendered["name"], "read_file")

    def test_a_dynamic_action_definition_does_not_mutate_the_registry(self):
        """Rendering refreshes a copy; the stored definition stays static."""
        registry = self._with_action(["stale_action"])
        registry.get_openai_tools()
        stored = registry.get("application_action")
        self.assertEqual(stored.parameters["action"]["enum"], ["stale_action"])

    def test_an_action_tool_without_an_action_parameter_is_not_rewritten(self):
        registry = ToolRegistry()
        registry.register(_tool("application_action",
                                parameters={"other": {"type": "string"}}))
        rendered = registry.get_openai_tools()[0]["function"]
        self.assertEqual(rendered["description"], "the application_action tool")


if __name__ == "__main__":
    unittest.main()
