"""Regression checks for grammar-constrained tool envelopes.

Originally driven through ``adapters.legacy_agent`` — a compatibility bridge
that built a second ``AgentService`` and a second tool catalog at import time
and was removed in ADR 0013. The canonical decoding path is
``agents.completion.parse_tool_call`` over the real registry, which is exactly
what ``agents.runner`` wires in production.
"""
from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ai_assistant.agents.completion import constrained_completion, parse_tool_call
from ai_assistant.observability import record_schema_error
from ai_assistant.tools.factory import create_tool_registry
from ai_assistant.tools.tool_contract import validate_tool_call

_TOOL_REGISTRY = create_tool_registry()


def _parse(response_text: str) -> tuple[str | None, dict | None]:
    return parse_tool_call(response_text, _TOOL_REGISTRY.models, validate_tool_call, record_schema_error)


class _StepAnswerModel:
    def create_chat_completion(self, **_kwargs):
        return {"choices": [{"message": {"content":
                '{"kind":"step_answer","content":"Teamlead điều phối dự án."}'}}]}


class _FakeRuntime:
    """Minimal LLMRuntime stand-in: concrete ``llm`` + ``llm_lock``."""

    def __init__(self, model) -> None:
        self.llm = model
        self.llm_lock = threading.Lock()


class ParseToolCallTests(unittest.TestCase):
    def test_canonical_tool_call(self) -> None:
        tool, params = _parse(
            '{"kind": "tool", "tool": "application_action",'
            ' "params": {"action": "reconstruction.load_images"}}'
        )
        self.assertEqual(tool, "application_action")
        self.assertEqual((params or {}).get("action"), "reconstruction.load_images")

    def test_app_action_alias_is_canonicalised(self) -> None:
        tool, params = _parse(
            '{"kind": "tool", "tool": "app_action_reconstruction",'
            ' "params": {"action": "reconstruction.load_images"}}'
        )
        self.assertEqual(tool, "application_action")
        self.assertEqual((params or {}).get("action"), "reconstruction.load_images")

    def test_manifest_action_alias_is_canonicalised(self) -> None:
        tool, params = _parse(
            '{"kind": "tool", "tool": "application_action",'
            ' "params": {"action": "mail.inbox"}}'
        )
        self.assertEqual(tool, "application_action")
        self.assertEqual((params or {}).get("action"), "mail.open")

    def test_final_envelope_is_not_a_tool_call(self) -> None:
        self.assertEqual(_parse('{"kind": "final", "content": "Done"}'), (None, None))

    def test_invalid_parameters_become_validation_error(self) -> None:
        tool, params = _parse(
            '{"kind": "tool", "tool": "application_action",'
            ' "params": {"action": "language.change", "language": "fr"}}'
        )
        self.assertEqual(tool, "_validation_error")
        self.assertIn("error", params or {})

    def test_start_reconstruction(self) -> None:
        tool, params = _parse(
            '{"kind": "tool", "tool": "application_action",'
            ' "params": {"action": "reconstruction.start_reconstruction"}}'
        )
        self.assertEqual(tool, "application_action")
        self.assertEqual((params or {}).get("action"), "reconstruction.start_reconstruction")


class ConstrainedCompletionTests(unittest.TestCase):
    def test_step_answer_envelope_is_preserved(self) -> None:
        """The llama.cpp adapter must keep a step-answer envelope intact.

        Otherwise LangGraph receives raw text and repeatedly executes the same
        plan step.
        """
        runtime = _FakeRuntime(_StepAnswerModel())
        with patch("ai_assistant.agents.completion.load_agent_runtime_settings",
                   return_value=SimpleNamespace(native_tool_calls=True)):
            response = constrained_completion(
                [], 32, 0.0, runtime, lambda: "llama_cpp", lambda messages, **kw: {},
                [], "{}", lambda i, o: None,
            )
        self.assertEqual(
            response,
            '{"kind": "step_answer", "content": "Teamlead điều phối dự án."}',
        )
        self.assertEqual(_parse(response)[0], "_step_answer")


if __name__ == "__main__":
    unittest.main()
