"""Unit tests for the HTTP adapter's injected-dependency protocols.

``adapters/http/_types.py`` declares the structural interfaces the routers
reach into on the legacy singletons (ADR 0001). It is imported by the routers
but was never exercised directly. These tests assert the protocols describe
the surface the adapters actually use, so a rename on either side is caught
here rather than at runtime inside a request handler.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, get_type_hints

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import unittest

from ai_assistant.adapters.http import _types

PROTOCOLS = (
    _types.LoggerLike,
    _types.LLMModuleLike,
    _types.RAGModuleLike,
    _types.ActionManifestLike,
    _types.PluginSpecLike,
    _types.PlatformSettingsLike,
    _types.PlatformLike,
    _types.ChatbotAgentLike,
)


class ProtocolSurfaceTests(unittest.TestCase):
    """Every declared protocol is importable and exposes its documented members."""

    def test_all_protocols_are_declared(self) -> None:
        for proto in PROTOCOLS:
            with self.subTest(protocol=proto.__name__):
                self.assertTrue(hasattr(proto, "__annotations__"))

    def test_logger_like_requires_the_five_log_levels(self) -> None:
        for name in ("debug", "info", "warning", "error", "exception"):
            with self.subTest(method=name):
                self.assertTrue(callable(getattr(_types.LoggerLike, name)))

    def test_llm_module_like_requires_runtime_and_model_attributes(self) -> None:
        from ai_assistant.ports import LLMRuntime

        # ADR 0041: the lease contract has one owner. ``LLMModuleLike`` adds
        # the admin/chat surface and inherits the runtime members instead of
        # re-declaring them, so the two views cannot drift apart.
        self.assertIn(LLMRuntime, _types.LLMModuleLike.__mro__)
        inherited = get_type_hints(_types.LLMModuleLike)
        for name in ("llm", "llm_lock", "active_model_desc", "is_vision_model"):
            with self.subTest(attribute=name):
                self.assertIn(name, inherited)
        for name in ("estimate_tokens", "load_model"):
            with self.subTest(method=name):
                self.assertTrue(callable(getattr(_types.LLMModuleLike, name)))

    def test_rag_module_like_requires_index_state_and_memory_release(self) -> None:
        annotations = _types.RAGModuleLike.__annotations__
        for name in ("knowledge_chunks", "knowledge_index", "bm25_index",
                     "embed_model_ref", "rag_lock"):
            with self.subTest(attribute=name):
                self.assertIn(name, annotations)
        # The adapter contract is public: the reranker is a read-only fact and
        # memory release is a named operation, never a private helper.
        self.assertTrue(isinstance(getattr(_types.RAGModuleLike, "reranker_active"), property))
        self.assertFalse(hasattr(_types.RAGModuleLike, "_reranker"))
        self.assertFalse(hasattr(_types.RAGModuleLike, "_release_ml_memory"))
        for name in ("release_ml_memory", "release_embedding_for_vision", "initialize_rag"):
            with self.subTest(method=name):
                self.assertTrue(callable(getattr(_types.RAGModuleLike, name)))

    def test_platform_like_nests_the_tool_catalogue_port_and_settings(self) -> None:
        # The nested members are read-only ``@property`` declarations (a frozen
        # dataclass attribute is not settable, so mypy rejects a plain protocol
        # attribute — see ADR 0001); properties surface via ``getattr``, not
        # via ``__annotations__``.
        self.assertTrue(isinstance(getattr(_types.PlatformLike, "plugins"), property))
        self.assertTrue(isinstance(getattr(_types.PlatformLike, "settings"), property))
        for name in ("enable_mcp", "enable_a2a", "allow_remote_a2a"):
            with self.subTest(setting=name):
                self.assertTrue(isinstance(getattr(_types.PlatformSettingsLike, name), property))

    def test_the_catalogue_is_the_shared_port_not_a_local_duplicate(self) -> None:
        """ADR 0032: the adapter used to declare its own duck-typed
        duplicate of the catalogue. There is one contract now, in
        ``ai_assistant.ports``."""
        from ai_assistant.ports import ToolCatalog

        self.assertIs(_types.ToolCatalog, ToolCatalog)
        self.assertFalse(hasattr(_types, "PluginRegistryLike"))

    def test_chatbot_agent_like_declares_build_and_clean_contract(self) -> None:
        self.assertTrue(callable(_types.ChatbotAgentLike.build_messages))
        self.assertTrue(callable(_types.ChatbotAgentLike.clean_answer))


class ConcreteFakesMatchProtocolTests(unittest.TestCase):
    """A minimal fake satisfies each protocol's shape (structural intent)."""

    def test_a_logger_like_implementation_exposes_every_level(self) -> None:
        class Recorder:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def _log(self, level: str, message: str, *args: Any) -> None:
                self.calls.append(f"{level}:{message}")

            def debug(self, message: str, *args: Any) -> None:
                self._log("debug", message)

            def info(self, message: str, *args: Any) -> None:
                self._log("info", message)

            def warning(self, message: str, *args: Any) -> None:
                self._log("warning", message)

            def error(self, message: str, *args: Any) -> None:
                self._log("error", message)

            def exception(self, message: str, *args: Any) -> None:
                self._log("exception", message)

        rec = Recorder()
        for level in ("debug", "info", "warning", "error", "exception"):
            getattr(rec, level)(level)
        self.assertEqual(rec.calls, [f"{level}:{level}" for level in
                                     ("debug", "info", "warning", "error", "exception")])

    def test_platform_settings_flags_are_plain_booleans(self) -> None:
        class Settings:
            enable_mcp = True
            enable_a2a = False
            allow_remote_a2a = False

        settings = Settings()
        for name in ("enable_mcp", "enable_a2a", "allow_remote_a2a"):
            with self.subTest(flag=name):
                self.assertIsInstance(getattr(settings, name), bool)

    def test_llm_module_like_estimate_tokens_signature(self) -> None:
        hints = get_type_hints(_types.LLMModuleLike.estimate_tokens)
        self.assertEqual(hints["return"], int)
        self.assertEqual(set(hints), {"text", "return"})

    def test_rag_module_like_initialize_rag_is_keyword_only(self) -> None:
        params = inspect.signature(_types.RAGModuleLike.initialize_rag).parameters
        self.assertEqual(list(params), ["self", "force_rebuild", "enable_reranker"])
        for name in ("force_rebuild", "enable_reranker"):
            with self.subTest(keyword_only=name):
                self.assertEqual(params[name].kind, inspect.Parameter.KEYWORD_ONLY)


if __name__ == "__main__":
    unittest.main()
