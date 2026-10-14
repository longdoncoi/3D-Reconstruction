"""Happy-path coverage for the completion adapters.

``agents/completion.py`` turns raw model bytes into typed envelopes. The
guard-path file pins every rejection; this file pins the shapes that must
keep working: remote and local tool-call envelopes, the XML envelope, final
and step-answer envelopes, plain prose, the planner/critic decoder on both
transports and the grammar precompile cache. (Moved verbatim from
``test_coverage_boost2.py`` during the coverage-file split.)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai_assistant.agents import completion as _completion
from ai_assistant.agents.completion import (
    _get_precompiled_grammar,
    constrained_completion,
    structured_completion,
)


class _FakeRuntime:
    """Minimal LLMRuntime double exposing ``llm`` and ``llm_lock``."""

    def __init__(self, response: dict | None = None, *, has_llm: bool = True) -> None:
        self.llm = None if not has_llm else MagicMock()
        if self.llm is not None and response is not None:
            self.llm.create_chat_completion.return_value = response
        self.llm_lock = threading.Lock()


def _remote_message(*, content: str | None = None, tool_calls: list | None = None) -> dict:
    message: dict = {}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if content is not None:
        message["content"] = content
    return message


class ConstrainedCompletionEnvelopeTests(unittest.TestCase):
    """Coverage for the constrained tool envelope decoder."""

    def _run(self, response: dict, *, llama: bool = False) -> str:
        runtime = _FakeRuntime(response)
        invoke = lambda: constrained_completion(  # noqa: E731 - local helper
            [{"role": "user", "content": "hi"}],
            max_tokens=128,
            temperature=0.0,
            llm_runtime=runtime,
            backend_mode_fn=lambda: "llama_cpp" if llama else "openai",
            openai_compatible_fn=lambda *a, **kw: response,
            openai_tools=[],
            grammar_schema=json.dumps({"type": "object"}),
            record_token_usage_fn=lambda *a, **kw: None,
        )
        if llama:
            return invoke()
        # ADR 0018: the remote leg is refused unless the operator opts in.
        # These tests exercise the decoder, so they opt in explicitly instead of
        # inheriting whatever the host environment happens to say.
        with patch.dict(os.environ, {"AGENT_HYBRID_POLICY": "cloud_allowed"}):
            return invoke()

    def test_remote_leg_is_refused_without_the_opt_in(self) -> None:
        """The default ``local_only`` policy must stop the remote call itself."""
        with patch.dict(os.environ, {}):
            os.environ.pop("AGENT_HYBRID_POLICY", None)
            with self.assertRaises(RuntimeError) as caught:
                constrained_completion(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=8,
                    temperature=0.0,
                    llm_runtime=_FakeRuntime(None, has_llm=False),
                    backend_mode_fn=lambda: "openai",
                    openai_compatible_fn=MagicMock(),
                    openai_tools=[],
                    grammar_schema=json.dumps({"type": "object"}),
                    record_token_usage_fn=lambda *a, **kw: None,
                )
        self.assertIn("AGENT_HYBRID_POLICY", str(caught.exception))

    def test_local_transport_failure_is_wrapped_with_context(self) -> None:
        """A dead llama.cpp backend surfaces as a runtime error, not a TypeError."""
        runtime = _FakeRuntime(None)
        runtime.llm.create_chat_completion.side_effect = OSError("model unloaded")
        with self.assertRaises(RuntimeError) as caught:
            constrained_completion(
                [{"role": "user", "content": "hi"}],
                max_tokens=8,
                temperature=0.0,
                llm_runtime=runtime,
                backend_mode_fn=lambda: "llama_cpp",
                openai_compatible_fn=MagicMock(),
                openai_tools=[],
                grammar_schema=json.dumps({"type": "object"}),
                record_token_usage_fn=lambda *a, **kw: None,
            )
        self.assertIn("Constrained tool decoding failed", str(caught.exception))
        self.assertIn("model unloaded", str(caught.exception))

    def test_remote_tool_call_envelope(self) -> None:
        response = {
            "choices": [{"message": _remote_message(tool_calls=[
                {"function": {"name": "read_file", "arguments": '{"path": "x.py"}'}}])}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        out = self._run(response)
        self.assertEqual(json.loads(out)["kind"], "tool")
        self.assertEqual(json.loads(out)["tool"], "read_file")

    def test_remote_plain_content(self) -> None:
        out = self._run({"choices": [{"message": _remote_message(content="plain answer")}], "usage": {}})
        self.assertEqual(out, "plain answer")

    def test_local_tool_call_envelope(self) -> None:
        response = {
            "choices": [{"message": _remote_message(tool_calls=[
                {"function": {"name": "search_text", "arguments": '{"query": "main"}'}}])}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }
        out = self._run(response, llama=True)
        self.assertEqual(json.loads(out)["tool"], "search_text")

    def test_local_xml_tool_call(self) -> None:
        xml = '<tool_call>{"kind":"tool","tool":"read_file","params":{"path":"a"}}</tool_call>'
        out = self._run(
            {"choices": [{"message": _remote_message(content=xml)}], "usage": {}},
            llama=True,
        )
        self.assertIn("read_file", out)

    def test_local_final_envelope(self) -> None:
        out = self._run(
            {"choices": [{"message": _remote_message(content='{"kind":"final","content":"Done"}')}], "usage": {}},
            llama=True,
        )
        self.assertEqual(out, "Done")

    def test_local_step_answer_envelope(self) -> None:
        out = self._run(
            {"choices": [{"message": _remote_message(content='{"kind":"step_answer","content":"ok"}')}], "usage": {}},
            llama=True,
        )
        self.assertEqual(json.loads(out)["kind"], "step_answer")

    def test_local_non_json_text(self) -> None:
        out = self._run(
            {"choices": [{"message": _remote_message(content="just prose")}], "usage": {}},
            llama=True,
        )
        self.assertEqual(out, "just prose")

    def test_unsupported_envelope_treated_as_text(self) -> None:
        out = self._run(
            {"choices": [{"message": _remote_message(content='{"kind":"other","x":1}')}], "usage": {}},
            llama=True,
        )
        self.assertIn("kind", out)

    def test_decoder_error_wrapped(self) -> None:
        with self.assertRaises(RuntimeError):
            constrained_completion(
                [], 0, 0.0, _FakeRuntime({}),
                backend_mode_fn=lambda: "openai",
                openai_compatible_fn=lambda *a, **kw: (_ for _ in ()).throw(OSError("boom")),
                openai_tools=[], grammar_schema="", record_token_usage_fn=lambda *a, **kw: None,
            )


class StructuredCompletionTests(unittest.TestCase):
    """Coverage for the planner/critic JSON decoder."""

    def test_remote_path(self) -> None:
        response = {"choices": [{"message": {"content": "  {\"passed\": true}  "}}]}
        out = structured_completion(
            [], 64, 0.0, {"type": "object"},
            _FakeRuntime(response),
            backend_mode_fn=lambda: "openai",
            openai_compatible_fn=lambda *a, **kw: response,
        )
        self.assertIn("passed", out)

    def test_llama_path_with_fake_grammar(self) -> None:
        fake_llama_cpp = SimpleNamespace(
            LlamaGrammar=SimpleNamespace(from_json_schema=MagicMock(return_value=object()))
        )
        response = {"choices": [{"message": {"content": '{"decision":"continue"}'}}]}
        runtime = _FakeRuntime(response)
        with patch.dict(sys.modules, {"llama_cpp": fake_llama_cpp}):
            out = structured_completion(
                [], 64, 0.0, {"type": "object"},
                runtime,
                backend_mode_fn=lambda: "llama_cpp",
                openai_compatible_fn=lambda *a, **kw: response,
            )
        self.assertIn("decision", out)

    def test_llama_path_decoder_error_wrapped(self) -> None:
        runtime = _FakeRuntime({})
        runtime.llm.create_chat_completion.side_effect = ValueError("bad completion")
        with self.assertRaises(RuntimeError):
            structured_completion(
                [], 64, 0.0, {"type": "object"},
                runtime,
                backend_mode_fn=lambda: "llama_cpp",
                openai_compatible_fn=lambda *a, **kw: {},
            )


class GrammarCacheTests(unittest.TestCase):
    """Coverage for the module-level grammar precompile cache.

    The cache is module-level global state, so each test saves and restores it.
    That keeps compile/hit coverage deterministic instead of depending on test
    ordering or on a real ``llama_cpp`` being installed (it is a manual
    heavyweight dependency that is absent from the offline CI test environment,
    so the success path must be exercised through a stand-in).
    """

    def setUp(self) -> None:
        self._saved_grammar = _completion._PRECOMPILED_GRAMMAR
        self._saved_hash = _completion._GRAMMAR_SCHEMA_HASH
        _completion._PRECOMPILED_GRAMMAR = None
        _completion._GRAMMAR_SCHEMA_HASH = None

    def tearDown(self) -> None:
        _completion._PRECOMPILED_GRAMMAR = self._saved_grammar
        _completion._GRAMMAR_SCHEMA_HASH = self._saved_hash

    @staticmethod
    def _working_llama_cpp() -> SimpleNamespace:
        return SimpleNamespace(
            LlamaGrammar=SimpleNamespace(
                from_json_schema=MagicMock(return_value=object())
            )
        )

    def test_cache_missing_llama_cpp_returns_none(self) -> None:
        # When llama_cpp cannot provide LlamaGrammar the cache falls back to None.
        fake = SimpleNamespace()  # no LlamaGrammar attribute -> ImportError on import
        with patch.dict(sys.modules, {"llama_cpp": fake}):
            self.assertIsNone(_get_precompiled_grammar("{}"))

    def test_successful_compile_is_then_served_from_cache(self) -> None:
        """A working llama_cpp compiles once, then the cache answers for free."""
        fake = self._working_llama_cpp()
        with patch.dict(sys.modules, {"llama_cpp": fake}):
            first = _get_precompiled_grammar('{"type":"object"}')
            second = _get_precompiled_grammar('{"type":"object"}')
        self.assertIsNotNone(first)
        self.assertIs(first, second)  # second call is a cache hit, no recompile
        fake.LlamaGrammar.from_json_schema.assert_called_once()

    def test_constrained_completion_forwards_the_precompiled_grammar(self) -> None:
        """The llama path passes the compiled grammar into the model kwargs."""
        fake = self._working_llama_cpp()
        response = {"choices": [{"message": {"content": '{"decision":"continue"}'}}]}
        runtime = _FakeRuntime(response)
        with patch.dict(sys.modules, {"llama_cpp": fake}):
            out = constrained_completion(
                [{"role": "user", "content": "hi"}],
                max_tokens=8,
                temperature=0.0,
                llm_runtime=runtime,
                backend_mode_fn=lambda: "llama_cpp",
                openai_compatible_fn=MagicMock(),
                openai_tools=[],
                grammar_schema=json.dumps({"type": "object"}),
                record_token_usage_fn=lambda *a, **kw: None,
            )
        _, kwargs = runtime.llm.create_chat_completion.call_args
        self.assertIn("grammar", kwargs)
        self.assertEqual(json.loads(out)["decision"], "continue")


if __name__ == "__main__":
    unittest.main()
