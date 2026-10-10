"""Unit tests for the agent system-prompt builder.

``build_agent_system_prompt`` is pure (no I/O), so a fake tool registry is
enough to render the prompt deterministically.
"""
from __future__ import annotations

import unittest
from unittest import mock

from ai_assistant.agents.prompts import build_agent_system_prompt


class _Spec:
    def __init__(self, name: str, description: str, parameters: dict | None = None) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters or {}


class _Registry:
    def __init__(self, specs: list[_Spec]) -> None:
        self._specs = specs

    def get_all(self) -> list[_Spec]:
        return self._specs


class AgentSystemPromptTest(unittest.TestCase):
    def test_builds_prompt_with_tool_descriptions(self) -> None:
        registry = _Registry(
            [
                _Spec(
                    "read_file",
                    "Read a file",
                    {"path": {"type": "string", "required": True, "description": "file path"}},
                ),
                _Spec("list_directory", "List a directory", {}),
            ]
        )
        prompt = build_agent_system_prompt(registry, language="en")
        self.assertIn("read_file", prompt)
        self.assertIn("path: string (required) — file path", prompt)
        self.assertIn("list_directory", prompt)
        self.assertIn("Respond to the user in English", prompt)

    def test_optional_parameter_is_marked(self) -> None:
        registry = _Registry(
            [_Spec("search_text", "Search", {"query": {"type": "string", "required": False}})]
        )
        prompt = build_agent_system_prompt(registry)
        self.assertIn("query: string (optional)", prompt)

    def test_default_registry_created_when_none(self) -> None:
        registry = _Registry([_Spec("demo", "Demo tool", {})])
        with mock.patch(
            "ai_assistant.tools.factory.create_tool_registry", return_value=registry
        ):
            prompt = build_agent_system_prompt()
        self.assertIn("demo", prompt)
        self.assertIn("Respond to the user in Vietnamese", prompt)

    def test_string_registry_argument_sets_language(self) -> None:
        registry = _Registry([_Spec("demo", "Demo tool", {})])
        with mock.patch(
            "ai_assistant.tools.factory.create_tool_registry", return_value=registry
        ):
            prompt = build_agent_system_prompt("en")
        self.assertIn("Respond to the user in English", prompt)


if __name__ == "__main__":
    unittest.main()
