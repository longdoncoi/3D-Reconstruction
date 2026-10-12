"""Unit tests for the centralised ``ai_assistant.settings`` accessors.

ADR 0001 requires every environment read to flow through this module, so these
tests pin the parsing semantics (defaults, bool/int coercion, filtering) for the
runtime toggles consumed by the rest of the platform. They pass explicit
``env`` mappings and never touch the real process environment.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_assistant.settings import (
    AgentRuntimeSettings,
    ArchitectureSettings,
    a2a_card_name,
    a2a_card_url,
    a2a_enabled,
    a2a_remote_agent_urls,
    agent_write_roots,
    apply_startup_environment_defaults,
    force_langgraph_agent,
    langsmith_settings,
    load_agent_runtime_settings,
    load_local_dotenv,
    lsp_binaries,
    observability_enabled,
    use_langgraph_agent,
    vision_llm_enabled,
)


class ArchitectureSettingsTests(unittest.TestCase):
    def _write(self, base: Path, name: str, body: str) -> None:
        config = base / "config"
        config.mkdir(parents=True, exist_ok=True)
        (config / name).write_text(body, encoding="utf-8")

    def test_defaults_without_config_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "AIAssistant"
            base.mkdir()
            with mock.patch.dict("os.environ", {}, clear=True):
                settings = ArchitectureSettings.load(base)
            self.assertEqual(settings.profile, "desktop")
            self.assertTrue(settings.enable_mcp)
            self.assertTrue(settings.enable_a2a)
            self.assertFalse(settings.allow_remote_a2a)
            self.assertFalse(settings.trusted_a2a_endpoints)
            self.assertEqual(settings.allowed_plugins, frozenset({"builtin.legacy-tools"}))
            self.assertEqual(settings.allowed_origins, ())
            self.assertEqual(settings.capability_scopes, {})

    def test_reads_and_merges_config_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "AIAssistant"
            base.mkdir()
            self._write(base, "base.toml", """
[a2a]
trusted_endpoints = ["http://a", "http://b"]

[agents]
capabilities = ["chat", "code"]
[agents.scopes]
chat = ["read", "write"]

[api]
allowed_origins = ["http://localhost"]

[plugins]
enabled = ["builtin.legacy-tools", "custom.tools"]
""")
            # Tables are merged per top-level key, so the profile file owns
            # the entire ``[runtime]`` table.
            self._write(base, "desktop.toml", """
[runtime]
enable_mcp = false
enable_a2a = false
allow_remote_a2a = true
""")
            with mock.patch.dict("os.environ", {"AI_ASSISTANT_PROFILE": "Desktop"}, clear=True):
                settings = ArchitectureSettings.load(base)
            self.assertEqual(settings.profile, "desktop")
            self.assertFalse(settings.enable_mcp)
            self.assertFalse(settings.enable_a2a)
            self.assertTrue(settings.allow_remote_a2a)
            self.assertEqual(settings.trusted_a2a_endpoints, frozenset({"http://a", "http://b"}))
            self.assertEqual(settings.agent_capabilities, frozenset({"chat", "code"}))
            self.assertEqual(settings.capability_scopes["chat"], frozenset({"read", "write"}))
            self.assertEqual(settings.allowed_origins, ("http://localhost",))
            self.assertIn("custom.tools", settings.allowed_plugins)

    def test_ignores_non_list_scope_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "AIAssistant"
            base.mkdir()
            self._write(base, "base.toml", """
[agents.scopes]
good = ["read"]
bad = "oops"
""")
            with mock.patch.dict("os.environ", {}, clear=True):
                settings = ArchitectureSettings.load(base)
            self.assertEqual(set(settings.capability_scopes), {"good"})

    def test_app_data_dir_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "AIAssistant"
            base.mkdir()
            data = Path(tmp) / "data"
            with mock.patch.dict("os.environ", {"APP_DATA_DIR": str(data)}, clear=True):
                settings = ArchitectureSettings.load(base)
            self.assertEqual(settings.data_dir, data / "AIAssistant")


class AgentRuntimeSettingsTests(unittest.TestCase):
    def test_defaults(self):
        settings = AgentRuntimeSettings.from_env({})
        self.assertTrue(settings.sandbox_enabled)
        self.assertEqual(settings.sandbox_runtime, "local")
        self.assertEqual(settings.sandbox_image, "python:3.11-slim")
        self.assertEqual(settings.max_write_bytes, 1048576)
        self.assertEqual(settings.checkpoint_backend, "memory")
        self.assertEqual(settings.checkpoint_url, "")
        self.assertFalse(settings.native_tool_calls)

    def test_overrides_and_casefold(self):
        env = {
            "AGENT_SANDBOX_ENABLED": "0",
            "AGENT_SANDBOX_RUNTIME": "DOCKER",
            "AGENT_MAX_WRITE_BYTES": "2048",
            "AGENT_CHECKPOINT_BACKEND": "SQLITE",
            "AGENT_NATIVE_TOOL_CALLS": "1",
        }
        settings = load_agent_runtime_settings(env)
        self.assertFalse(settings.sandbox_enabled)
        self.assertEqual(settings.sandbox_runtime, "docker")
        self.assertEqual(settings.max_write_bytes, 2048)
        self.assertEqual(settings.checkpoint_backend, "sqlite")
        self.assertTrue(settings.native_tool_calls)

    def test_from_env_uses_process_environment(self):
        with mock.patch.dict("os.environ", {"AGENT_MAX_WRITE_BYTES": "99"}):
            self.assertEqual(load_agent_runtime_settings().max_write_bytes, 99)


class UseLanggraphAgentTests(unittest.TestCase):
    def test_default_enabled(self):
        self.assertTrue(use_langgraph_agent({}))

    def test_disabled_only_for_zero(self):
        self.assertFalse(use_langgraph_agent({"USE_LANGGRAPH_AGENT": "0"}))
        self.assertTrue(use_langgraph_agent({"USE_LANGGRAPH_AGENT": "1"}))


class ObservabilityToggleTests(unittest.TestCase):
    def test_default_disabled(self):
        self.assertFalse(observability_enabled({}))

    def test_enabled(self):
        self.assertTrue(observability_enabled({"AGENT_OBSERVABILITY": "1"}))
        self.assertFalse(observability_enabled({"AGENT_OBSERVABILITY": "true"}))


class LangsmithSettingsTests(unittest.TestCase):
    def test_disabled_returns_none(self):
        self.assertIsNone(langsmith_settings({}))
        self.assertIsNone(langsmith_settings({"LANGSMITH_TRACING": "true"}))

    def test_requires_api_key(self):
        self.assertIsNone(langsmith_settings({"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": ""}))

    def test_returns_project_and_endpoint(self):
        result = langsmith_settings({
            "LANGSMITH_TRACING": "YES",
            "LANGSMITH_API_KEY": "secret",
            "LANGSMITH_PROJECT": "proj",
            "LANGSMITH_ENDPOINT": "http://ls",
        })
        self.assertEqual(result, {"project": "proj", "api_key": "secret", "endpoint": "http://ls"})

    def test_defaults(self):
        result = langsmith_settings({"LANGSMITH_TRACING": "1", "LANGSMITH_API_KEY": "k"})
        self.assertEqual(result["project"], "3d-reconstruction")
        self.assertEqual(result["endpoint"], "https://api.smith.langchain.com")


class A2ASettingsTests(unittest.TestCase):
    def test_enabled(self):
        self.assertFalse(a2a_enabled({}))
        self.assertTrue(a2a_enabled({"A2A_ENABLED": "1"}))

    def test_remote_agent_urls_filters_blanks(self):
        self.assertEqual(a2a_remote_agent_urls({}), ())
        self.assertEqual(
            a2a_remote_agent_urls({"A2A_REMOTE_AGENTS": " http://a , ,http://b "}),
            ("http://a", "http://b"),
        )

    def test_card_name_and_url(self):
        self.assertEqual(a2a_card_name({}), "3D-Reconstruction AI Assistant")
        self.assertEqual(a2a_card_name({"A2A_CARD_NAME": "X"}), "X")
        self.assertEqual(a2a_card_url({}), "http://127.0.0.1:8080")
        self.assertEqual(a2a_card_url({"A2A_CARD_URL": "http://x"}), "http://x")


class AgentWriteRootsTests(unittest.TestCase):
    def test_default_roots(self):
        roots = agent_write_roots({})
        self.assertIn("src", roots)
        self.assertIn("tests", roots)

    def test_custom_roots_are_trimmed(self):
        self.assertEqual(
            agent_write_roots({"AGENT_WRITE_ALLOWLIST": "a, b , ,c"}),
            frozenset({"a", "b", "c"}),
        )


class LspBinariesTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(lsp_binaries({}), ("clangd", "pylsp", 15))

    def test_overrides(self):
        env = {"AGENT_CLANGD_BIN": "c", "AGENT_PYLSP_BIN": "p", "AGENT_LSP_TIMEOUT": "30"}
        self.assertEqual(lsp_binaries(env), ("c", "p", 30))

    def test_invalid_timeout_falls_back(self):
        self.assertEqual(lsp_binaries({"AGENT_LSP_TIMEOUT": "not-int"})[2], 15)


class StartupEnvironmentBootstrapTests(unittest.TestCase):
    """The process-wide bootstrap that used to be inlined in ``legacy/config.py``.

    ADR 0001 moved the ``.env`` parser, the library-default env keys and the
    warning filters into ``settings`` so the legacy shim is no longer a second
    environment owner. These tests pin each of those behaviours.
    """

    def test_vision_llm_defaults_on(self):
        self.assertTrue(vision_llm_enabled({}))

    def test_vision_llm_parses_every_truthy_spelling(self):
        for value in ("1", "true", "TRUE", " Yes ", "on"):
            self.assertTrue(vision_llm_enabled({"AI_ENABLE_VISION_LLM": value}), value)

    def test_vision_llm_rejects_every_other_value(self):
        for value in ("0", "no", "off", "", "maybe"):
            self.assertFalse(vision_llm_enabled({"AI_ENABLE_VISION_LLM": value}), value)

    def test_force_langgraph_defaults_on(self):
        self.assertTrue(force_langgraph_agent({}))

    def test_force_langgraph_reads_only_the_explicit_toggle(self):
        self.assertFalse(force_langgraph_agent({"FORCE_LANGGRAPH_AGENT": "0"}))
        self.assertTrue(force_langgraph_agent({"FORCE_LANGGRAPH_AGENT": "1"}))
        # ``USE_LANGGRAPH_AGENT`` must not bleed into the force toggle.
        self.assertTrue(force_langgraph_agent({"USE_LANGGRAPH_AGENT": "0"}))

    def test_load_local_dotenv_parses_keys_and_ignores_junk(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, ".env").write_text(
                "# a comment\n"
                "AI_DOTENV_A=value_a\n"
                "AI_DOTENV_B='quoted'\n"
                "line without an equals sign\n"
                "=missing_key\n"
                "  AI_DOTENV_C = spaced  \n",
                encoding="utf-8",
            )
            with mock.patch.dict("os.environ", {}, clear=False):
                load_local_dotenv(tmp)
                self.assertEqual(os.environ["AI_DOTENV_A"], "value_a")
                self.assertEqual(os.environ["AI_DOTENV_B"], "quoted")
                self.assertEqual(os.environ["AI_DOTENV_C"], "spaced")
                self.assertNotIn("missing_key", os.environ)

    def test_load_local_dotenv_never_overwrites_existing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, ".env").write_text("AI_DOTENV_KEEP=from_file\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {"AI_DOTENV_KEEP": "preset"}):
                load_local_dotenv(tmp)
                self.assertEqual(os.environ["AI_DOTENV_KEEP"], "preset")

    def test_load_local_dotenv_treats_a_missing_file_as_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict("os.environ", {}, clear=False):
                # Must not raise: the .env file is a local convenience, not a
                # required input.
                load_local_dotenv(Path(tmp) / "nope")
                self.assertNotIn("AI_DOTENV_ABSENT", os.environ)

    def test_startup_defaults_apply_the_library_quieting_keys(self):
        expected = {
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "TF_ENABLE_ONEDNN_OPTS": "0",
        }
        with mock.patch.dict("os.environ", {}, clear=True):
            apply_startup_environment_defaults(reload=True)
            for key, value in expected.items():
                self.assertEqual(os.environ[key], value)

    def test_startup_defaults_never_clobber_an_operator_override(self):
        with mock.patch.dict("os.environ", {"TF_CPP_MIN_LOG_LEVEL": "0"}):
            apply_startup_environment_defaults(reload=True)
            self.assertEqual(os.environ["TF_CPP_MIN_LOG_LEVEL"], "0")

    def test_startup_defaults_are_idempotent(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            apply_startup_environment_defaults(reload=True)
            os.environ["TOKENIZERS_PARALLELISM"] = "operator_value"
            apply_startup_environment_defaults()  # second call is a no-op
            self.assertEqual(os.environ["TOKENIZERS_PARALLELISM"], "operator_value")


if __name__ == "__main__":
    unittest.main()
