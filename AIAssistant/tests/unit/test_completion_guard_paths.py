"""Guard-path coverage for the LLM completion adapters.

``agents/completion.py`` is where a model's raw bytes become either a typed
tool envelope or plain prose. Every malformed shape it rejects is a shape the
runtime really does receive from a quantised local model, so each rejection
arm is a contract worth pinning: the XML-envelope parser, ``parse_tool_call``'s
non-object arms, the constrained decoder's non-string / JSON-array /
tool-envelope exits and the structured decoder's non-string guard. The
happy-path envelope tests cover none of them.
"""
from __future__ import annotations

import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from ai_assistant.agents.completion import (
    CRITIC_JSON_SCHEMA,
    PLANNER_JSON_SCHEMA,
    _extract_tool_call_xml,
    constrained_completion,
    parse_tool_call,
    structured_completion,
)
from ai_assistant.observability import record_schema_error
from ai_assistant.tools.factory import create_tool_registry
from ai_assistant.tools.tool_contract import validate_tool_call

_REGISTRY = create_tool_registry()

# The reasoning tag and the tool-call terminator, spelled out so the
# literals cannot drift from ``agents/completion.py``.
THINK_TAG = "<think>"
THINK_CLOSE = "</think>"
TOOL_OPEN = "<tool_call>"
TOOL_END = "</tool_call>"


def _parse(text: str):
    return parse_tool_call(text, _REGISTRY.models, validate_tool_call, record_schema_error)


class _FakeRuntime:
    """Minimal stand-in for the llama.cpp runtime adapter."""

    def __init__(self, response: dict) -> None:
        self.llm = SimpleNamespace(create_chat_completion=MagicMock(return_value=response))
        self.llm_lock = threading.Lock()


def _llama_response(content) -> dict:
    return {"choices": [{"message": {"content": content}}], "usage": {}}


def _run(response: dict) -> str:
    return constrained_completion(
        [{"role": "user", "content": "hi"}],
        max_tokens=128,
        temperature=0.0,
        llm_runtime=_FakeRuntime(response),
        backend_mode_fn=lambda: "llama_cpp",
        openai_compatible_fn=lambda *a, **kw: response,
        openai_tools=[],
        grammar_schema=json.dumps({"type": "object"}),
        record_token_usage_fn=lambda *a, **kw: None,
    )


def _envelope(payload: str) -> str:
    """Wrap a payload the way a Qwen-style model emits a tool call."""
    return TOOL_OPEN + payload + TOOL_END


class XmlEnvelopeRejectionTests(unittest.TestCase):
    """``_extract_tool_call_xml`` must never raise on a stray envelope.

    The extractor is regex-driven and non-greedy, so it only recovers a
    *single-level* JSON object — every payload below is deliberately flat,
    which is also the only shape the real Qwen envelope can carry.
    """

    def test_text_without_an_envelope_is_not_a_call(self):
        self.assertIsNone(_extract_tool_call_xml("just prose about tool calls"))

    def test_an_envelope_holding_malformed_json_is_not_a_call(self):
        """The regex matches the braces, but the captured body is not JSON."""
        self.assertIsNone(_extract_tool_call_xml(_envelope("{not valid json}")))

    def test_an_envelope_without_its_open_tag_is_not_a_call(self):
        self.assertIsNone(_extract_tool_call_xml("{broken" + TOOL_END))

    def test_string_arguments_that_are_not_json_reject_the_call(self):
        """``arguments`` arrives as an unparsed JSON *string* from some models."""
        payload = '{"kind":"tool","name":"read_file","arguments":"not json"}'
        self.assertIsNone(_extract_tool_call_xml(_envelope(payload)))

    def test_a_missing_tool_name_rejects_the_call(self):
        payload = '{"kind":"tool","name":"","params":null}'
        self.assertIsNone(_extract_tool_call_xml(_envelope(payload)))

    def test_non_mapping_arguments_reject_the_call(self):
        payload = '{"kind":"tool","name":"read_file","arguments":42}'
        self.assertIsNone(_extract_tool_call_xml(_envelope(payload)))

    def test_a_tool_without_positional_arguments_defaults_to_an_empty_map(self):
        """No ``arguments`` and no ``params`` still yields a runnable call."""
        parsed = json.loads(_extract_tool_call_xml(_envelope('{"kind":"tool","name":"list_directory"}')))
        self.assertEqual(parsed["tool"], "list_directory")
        self.assertEqual(parsed["params"], {})

    def test_a_tool_name_from_the_alternative_key_is_accepted(self):
        """Some models emit ``tool`` where others emit ``name``."""
        parsed = json.loads(_extract_tool_call_xml(_envelope('{"kind":"tool","tool":"read_file"}')))
        self.assertEqual(parsed["tool"], "read_file")


class ParseToolCallRejectionTests(unittest.TestCase):
    def test_a_json_array_is_not_an_envelope(self):
        self.assertEqual(_parse("[1, 2, 3]"), (None, None))

    def test_non_mapping_params_are_not_an_envelope(self):
        self.assertEqual(
            _parse('{"kind":"tool","tool":"read_file","params":"nope"}'), (None, None)
        )

    def test_a_tool_call_without_params_is_not_an_envelope(self):
        self.assertEqual(_parse('{"kind":"tool","tool":"read_file"}'), (None, None))


class ConstrainedDecoderExitTests(unittest.TestCase):
    def test_non_string_content_is_a_hard_failure(self):
        """The decoder contract is text; anything else is a bug, not prose."""
        with self.assertRaises(RuntimeError) as caught:
            _run(_llama_response(42))
        self.assertIn("no text content", str(caught.exception))

    def test_a_structured_tool_envelope_is_re_emitted(self):
        """A decoded ``tool`` envelope must survive round-tripping unchanged."""
        raw = '{"kind":"tool","tool":"read_file","params":{"path":"a.py"}}'
        self.assertEqual(json.loads(_run(_llama_response(raw)))["kind"], "tool")

    def test_a_json_array_after_think_tags_is_returned_as_text(self):
        """No ``{`` means no object to decode, so the array is the answer."""
        content = THINK_TAG + " hmm " + THINK_CLOSE + " [1, 2, 3]"
        self.assertEqual(_run(_llama_response(content)), "[1, 2, 3]")

    def test_a_tagged_but_unparseable_envelope_is_returned_as_text(self):
        """The open tag is present so the XML path runs, but extraction fails
        and the JSON fallback fails too — the raw text is the answer."""
        content = TOOL_OPEN + "{broken, unterminated" + TOOL_END
        self.assertEqual(_run(_llama_response(content)), content)

    def test_a_broken_json_object_after_prose_is_returned_as_text(self):
        """The raw decoder fails, so the prose wins rather than raising."""
        content = 'Here: {"kind": "final", "content": unterminated'
        self.assertEqual(_run(_llama_response(content)), content)

    def test_an_envelope_without_a_tool_name_falls_through_to_the_json_path(self):
        """The terminator is present but there is no tool to report, so the
        XML extractor declines and the JSON path gets its turn."""
        content = '{"kind": "final", "content": "ok"} ' + TOOL_END
        self.assertEqual(_run(_llama_response(content)), "ok")


class StructuredDecoderGuardTests(unittest.TestCase):
    def test_non_string_content_is_a_hard_failure(self):
        import os
        from unittest import mock

        # The remote branch needs the ADR 0018 opt-in: the policy gate is
        # hoisted above the decode wrapper, so a bare local_only default
        # refuses before any decoding happens.
        with mock.patch.dict(os.environ, {"AGENT_HYBRID_POLICY": "cloud_allowed"}):
            with self.assertRaises(RuntimeError) as caught:
                structured_completion(
                    [], 64, 0.0, {"type": "object"}, _FakeRuntime(_llama_response(None)),
                    backend_mode_fn=lambda: "openai",
                    openai_compatible_fn=lambda *a, **kw: _llama_response(None),
                )
        self.assertIn("no text content", str(caught.exception))

    def test_think_tags_are_stripped_from_the_structured_answer(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"AGENT_HYBRID_POLICY": "cloud_allowed"}):
            out = structured_completion(
                [], 64, 0.0, {"type": "object"},
                _FakeRuntime(_llama_response("")),
                backend_mode_fn=lambda: "openai",
                openai_compatible_fn=lambda *a, **kw: _llama_response(
                    THINK_TAG + " checking " + THINK_CLOSE + '{"passed": true}'
                ),
            )
        self.assertEqual(out, '{"passed": true}')



# ── Parser and schema structure pins (moved from test_coverage_boost) ──

class CompletionTests(unittest.TestCase):
    """Test completion and parsing coverage."""

    def test_parse_tool_call_valid_json(self) -> None:
        """parse_tool_call handles valid JSON tool calls."""
        text = '{"kind": "tool", "tool": "read_file", "params": {"path": "test.txt"}}'
        tool, params = parse_tool_call(
            text, {},
            lambda *a, **kw: (a[1], None),  # validate_fn returns (params, error)
            lambda *a, **kw: None,
        )
        self.assertEqual(tool, "read_file")
        self.assertEqual(params.get("path"), "test.txt")

    def test_parse_tool_call_survives_a_validator_that_returns_none(self) -> None:
        """A ``None`` from the validator is a validation error, never a crash.

        Defensive arm added after a real crash: unpacking ``None`` as a
        ``(validated, error)`` pair raised instead of degrading.
        """
        text = '{"kind": "tool", "tool": "read_file", "params": {"path": "test.txt"}}'
        tool, params = parse_tool_call(text, {}, lambda *a, **kw: None, lambda *a, **kw: None)
        self.assertEqual(tool, "_validation_error")
        self.assertIn("validate_tool_call returned None", params["error"])

    def test_parse_tool_call_no_tool(self) -> None:
        """parse_tool_call returns None for non-tool text."""
        tool, _ = parse_tool_call("Just a regular message", {}, lambda *a, **kw: None, lambda *a, **kw: None)
        self.assertIsNone(tool)

    def test_planner_json_schema_structure(self) -> None:
        """PLANNER_JSON_SCHEMA has correct structure."""
        self.assertIn("type", PLANNER_JSON_SCHEMA)
        self.assertEqual(PLANNER_JSON_SCHEMA["type"], "object")

    def test_critic_json_schema_structure(self) -> None:
        """CRITIC_JSON_SCHEMA has correct structure."""
        self.assertIn("type", CRITIC_JSON_SCHEMA)
        self.assertEqual(CRITIC_JSON_SCHEMA["type"], "object")


class HybridPolicyGateTests(unittest.TestCase):
    """ADR 0018: the remote branch refuses under the default ``local_only``.

    ``backend_mode`` is only a configuration *read*; the decision to send
    bytes off-box is made here, in ``constrained_completion``, before the
    transport is touched — and again inside ``openai_compatible_completion``.
    """

    def _call(self, backend: str):
        return constrained_completion(
            [{"role": "user", "content": "hi"}],
            max_tokens=8,
            temperature=0.0,
            llm_runtime=_FakeRuntime(_llama_response("x")),
            backend_mode_fn=lambda: backend,
            openai_compatible_fn=MagicMock(return_value=_llama_response("x")),
            openai_tools=[],
            grammar_schema=json.dumps({"type": "object"}),
            record_token_usage_fn=lambda *a, **kw: None,
        )

    def test_remote_backend_is_blocked_without_cloud_opt_in(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}):
            os.environ.pop("AGENT_HYBRID_POLICY", None)
            with self.assertRaises(RuntimeError) as caught:
                self._call("openai")
        self.assertIn("AGENT_HYBRID_POLICY", str(caught.exception))

    def test_remote_backend_runs_once_cloud_is_opted_in(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"AGENT_HYBRID_POLICY": "cloud_allowed"}):
            result = self._call("openai")
        self.assertEqual(result, "x")

    def _call_structured(self, backend: str):
        return structured_completion(
            [{"role": "user", "content": "hi"}],
            8, 0.0, {"type": "object"},
            _FakeRuntime(_llama_response("x")),
            backend_mode_fn=lambda: backend,
            openai_compatible_fn=MagicMock(return_value=_llama_response("x")),
        )

    def test_structured_remote_backend_is_blocked_without_cloud_opt_in(self):
        """The planner/critic decoder runs the same gate, hoisted above the
        decode wrapper: the refusal must not be relabelled as a decoding
        failure (finding 16)."""
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}):
            os.environ.pop("AGENT_HYBRID_POLICY", None)
            with self.assertRaises(RuntimeError) as caught:
                self._call_structured("openai")
        self.assertIn("AGENT_HYBRID_POLICY", str(caught.exception))
        self.assertNotIn("decoding failed", str(caught.exception))

    def test_structured_local_backend_is_unaffected_by_the_policy(self):
        import os
        import sys
        from unittest import mock

        fake_llama_cpp = SimpleNamespace(
            LlamaGrammar=SimpleNamespace(from_json_schema=lambda schema: object())
        )
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("AGENT_HYBRID_POLICY", None)
            with mock.patch.dict(sys.modules, {"llama_cpp": fake_llama_cpp}):
                self.assertEqual(self._call_structured("llama_cpp"), "x")

    def test_local_backend_is_unaffected_by_the_policy(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}):
            os.environ.pop("AGENT_HYBRID_POLICY", None)
            self.assertEqual(self._call("llama_cpp"), "x")


if __name__ == "__main__":
    unittest.main()
