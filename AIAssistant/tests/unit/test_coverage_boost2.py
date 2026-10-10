"""Second coverage-boost batch: completion, A2A client, model registry, app factory, domain errors, builtin tools.

Targets modules that were below 75% after the first batch. Network access is
fully mocked; no external services are contacted.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai_assistant.adapters.a2a_client import RemoteAgentCard, TrustedA2AClient
from ai_assistant.agents.completion import (
    _get_precompiled_grammar,
    constrained_completion,
    structured_completion,
)
from ai_assistant.bootstrap.app_factory import create_app
from ai_assistant.config.model_registry import ModelDefinition, ModelRegistry, load_model_registry
from ai_assistant.domain.errors import (
    ApprovalRequiredError,
    AuthorizationError,
    ConfigurationError,
    ContextWindowExceededError,
    ModelLoadError,
    ModelNotLoadedError,
    NotFoundError,
    PlatformError,
    RAGNotReadyError,
    SandboxViolationError,
    ServiceUnavailableError,
    TaskCancelledError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
    UnprocessableRequestError,
)
from ai_assistant.domain.security import DataClassification
from ai_assistant.settings import ArchitectureSettings
from ai_assistant.tools.builtin import desktop_tools, transfer_tools


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


class ConstrainedCompletionTests(unittest.TestCase):
    """Coverage for the constrained tool envelope decoder."""

    def _run(self, response: dict, *, llama: bool = False) -> str:
        runtime = _FakeRuntime(response)
        return constrained_completion(
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
    """Coverage for the module-level grammar precompile cache."""

    def test_cache_missing_llama_cpp_returns_none(self) -> None:
        # When llama_cpp cannot provide LlamaGrammar the cache falls back to None.
        fake = SimpleNamespace()  # no LlamaGrammar attribute -> ImportError on import
        with patch.dict(sys.modules, {"llama_cpp": fake}):
            self.assertIsNone(_get_precompiled_grammar("{}"))


class TrustedA2AClientSubmitTests(unittest.TestCase):
    """Coverage for TrustedA2AClient submit flow."""

    def _client(self) -> TrustedA2AClient:
        return TrustedA2AClient(
            trusted_endpoints=frozenset({"https://trusted.example.com"}),
            timeout_seconds=5,
        )

    def test_submit_happy_path(self) -> None:
        client = self._client()
        card = RemoteAgentCard(
            endpoint="https://trusted.example.com", name="Agent", version="1.0.0",
            skills=frozenset({"supervisor"}), streaming=False,
        )
        response_body = b'{"task": {"id": "t-1", "status": {"state": "completed"}}}'
        with patch.object(client, "_read_card", return_value=card), patch(
            "ai_assistant.adapters.a2a_client.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value.read.return_value = response_body
            mock_urlopen.return_value = mock_resp

            result = client.submit("https://trusted.example.com", "supervisor", "hello")

        self.assertEqual(result["routing"], "remote_selected")
        self.assertEqual(result["task"]["id"], "t-1")

    def test_submit_rejects_restricted_classification(self) -> None:
        client = self._client()
        result = client.submit(
            "https://trusted.example.com", "supervisor", "secret",
            classification=DataClassification.RESTRICTED,
        )
        self.assertEqual(result["routing"], "rejected_policy")

    def test_submit_rejects_missing_capability(self) -> None:
        client = self._client()
        card = RemoteAgentCard(
            endpoint="https://trusted.example.com", name="Agent", version="1.0.0",
            skills=frozenset({"research"}), streaming=False,
        )
        with patch.object(client, "_read_card", return_value=card):
            result = client.submit("https://trusted.example.com", "supervisor", "hello")
        self.assertEqual(result["routing"], "rejected_no_eligible_agent")

    def test_submit_transport_failure_degraded(self) -> None:
        client = self._client()
        card = RemoteAgentCard(
            endpoint="https://trusted.example.com", name="Agent", version="1.0.0",
            skills=frozenset({"supervisor"}), streaming=False,
        )
        with patch.object(client, "_read_card", return_value=card), patch(
            "ai_assistant.adapters.a2a_client.urlopen",
            side_effect=OSError("network down"),
        ):
            result = client.submit("https://trusted.example.com", "supervisor", "hello")
        self.assertEqual(result["routing"], "degraded_dependency")

    def test_discover_skips_failed_endpoints(self) -> None:
        client = TrustedA2AClient(
            trusted_endpoints=frozenset({"https://a.example.com", "https://b.example.com"}),
            timeout_seconds=5,
        )
        good = RemoteAgentCard(
            endpoint="https://a.example.com", name="A", version="1", skills=frozenset(), streaming=True,
        )

        def _read_card(endpoint: str) -> RemoteAgentCard:
            if endpoint.endswith("b.example.com"):
                raise ValueError("bad card")
            return good

        with patch.object(client, "_read_card", side_effect=_read_card):
            cards = client.discover()
        self.assertEqual(set(cards), {"https://a.example.com"})

    def test_read_card_invalid_payload(self) -> None:
        client = self._client()
        with patch("ai_assistant.adapters.a2a_client.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value.read.return_value = b'{"name": 123}'
            mock_urlopen.return_value = mock_resp
            with self.assertRaises(ValueError):
                client._read_card("https://trusted.example.com")

    def test_read_card_no_capabilities(self) -> None:
        client = self._client()
        with patch("ai_assistant.adapters.a2a_client.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value.read.return_value = (
                b'{"name": "A", "skills": [{"id": 123}]}'
            )
            mock_urlopen.return_value = mock_resp
            with self.assertRaises(ValueError):
                client._read_card("https://trusted.example.com")


class ModelRegistryTests(unittest.TestCase):
    """Coverage for the model registry."""

    def test_get_by_index_edges(self) -> None:
        fallback = ModelDefinition("fallback/repo", "f.gguf", "fallback")
        m0 = ModelDefinition("repo/a", "a.gguf", "a")
        m1 = ModelDefinition("repo/b", "b.gguf", "b")
        registry = ModelRegistry(models=(m0, m1), fallback=fallback)
        self.assertIs(registry.get_by_index(0), m0)
        self.assertIs(registry.get_by_index(1), m1)
        self.assertIs(registry.get_by_index(99), m0)   # out of bounds -> first
        self.assertIs(registry.get_by_index(-2), m0)   # negative -> first

    def test_get_by_index_empty_returns_fallback(self) -> None:
        fallback = ModelDefinition("fallback/repo", "f.gguf", "fallback")
        registry = ModelRegistry(models=(), fallback=fallback)
        self.assertIs(registry.get_by_index(0), fallback)

    def test_load_model_registry_missing_config_uses_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = load_model_registry(tmpdir)
        self.assertGreater(len(registry.models), 0)
        self.assertTrue(registry.models[0].is_vision is False or registry.models[0].is_vision is True)

    def test_load_model_registry_from_toml(self) -> None:
        toml_text = (
            "[[models]]\n"
            'repo_id = "test/repo"\n'
            'filename = "m.gguf"\n'
            'desc = "Test"\n'
            "is_vision = true\n"
            "[[models]]\n"
            'repo_id = "test/repo2"\n'
            'filename = "m2.gguf"\n'
            'desc = "Test 2"\n'
            "[fallback]\n"
            'repo_id = "fallback/repo"\n'
            'filename = "f.gguf"\n'
            'desc = "Fallback"\n'
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "models.toml").write_text(toml_text, encoding="utf-8")
            registry = load_model_registry(tmpdir)
        self.assertEqual(len(registry.models), 2)
        self.assertTrue(registry.models[0].is_vision)
        self.assertEqual(registry.fallback.repo_id, "fallback/repo")


class AppFactoryTests(unittest.TestCase):
    """Coverage for the FastAPI composition boundary."""

    def _settings(self, tmpdir: str) -> ArchitectureSettings:
        return ArchitectureSettings(
            profile="desktop",
            data_dir=Path(tmpdir),
            enable_mcp=True,
            enable_a2a=True,
            allow_remote_a2a=False,
            trusted_a2a_endpoints=frozenset(),
            agent_capabilities=frozenset({"supervisor"}),
            capability_scopes={},
            allowed_plugins=frozenset(),
            allowed_origins=("*",),
        )

    def test_create_app_registers_handlers_and_cors(self) -> None:
        from fastapi.middleware.cors import CORSMiddleware

        async def _lifespan(app):
            yield

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(self._settings(tmpdir), _lifespan)
        self.assertIn(NotFoundError, app.exception_handlers)
        self.assertIn(AuthorizationError, app.exception_handlers)
        self.assertIn(ModelNotLoadedError, app.exception_handlers)
        self.assertIn(ServiceUnavailableError, app.exception_handlers)
        self.assertTrue(any(m.cls is CORSMiddleware for m in app.user_middleware))


class DomainErrorTests(unittest.TestCase):
    """Coverage for typed domain exceptions."""

    def test_error_hierarchy_and_messages(self) -> None:
        self.assertTrue(issubclass(ConfigurationError, PlatformError))
        self.assertTrue(issubclass(ModelLoadError, PlatformError))
        self.assertTrue(issubclass(ContextWindowExceededError, PlatformError))
        self.assertTrue(issubclass(ToolNotFoundError, PlatformError))
        self.assertTrue(issubclass(SandboxViolationError, PlatformError))
        self.assertTrue(issubclass(RAGNotReadyError, PlatformError))
        self.assertTrue(issubclass(TaskCancelledError, PlatformError))
        self.assertTrue(issubclass(UnprocessableRequestError, PlatformError))

    def test_structured_error_attributes(self) -> None:
        validation = ToolValidationError("read_file", "missing path")
        self.assertEqual(validation.tool_name, "read_file")
        self.assertIn("missing path", str(validation))

        execution = ToolExecutionError("run_command", ValueError("boom"))
        self.assertIn("boom", str(execution))

        approval = ApprovalRequiredError("write_file", "act-42")
        self.assertEqual(approval.action_id, "act-42")
        self.assertIn("act-42", str(approval))

        unprocessable = UnprocessableRequestError("bad request shape")
        self.assertEqual(str(unprocessable), "bad request shape")


class BuiltinToolTests(unittest.TestCase):
    """Coverage for the transfer and desktop builtin tools."""

    def test_desktop_action_valid(self) -> None:
        result = desktop_tools.tool_application_action({"action": "viewer.load_2d"})
        self.assertTrue(result["pending_ui_ack"])
        self.assertEqual(result["action"], "viewer.load_2d")

    def test_desktop_action_invalid(self) -> None:
        result = desktop_tools.tool_application_action({})
        self.assertIn("error", result)

    def test_transfer_tools(self) -> None:
        self.assertEqual(transfer_tools.tool_transfer_to_code_agent({"intent": "fix"})["status"], "transferred_to_code")
        self.assertEqual(
            transfer_tools.tool_transfer_to_toolapp_agent({"intent": "ui"})["status"], "transferred_to_toolapp"
        )
        self.assertEqual(
            transfer_tools.tool_transfer_to_chatbot_agent({"intent": "chat"})["status"], "transferred_to_chatbot"
        )


if __name__ == "__main__":
    unittest.main()
