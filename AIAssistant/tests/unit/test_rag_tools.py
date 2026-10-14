"""Unit tests for the ``rag_search`` builtin tool.

The handler receives its retrieval port from the composition root
(``create_tool_registry(rag_context=…)``), so a test injects a fake provider
directly. The previous version of this tool reached into
``ai_assistant.bootstrap.server`` at call time, which forced every test to fake
an entire module in ``sys.modules`` — the service-locator smell this refactor
removed.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from ai_assistant.tools.builtin import rag_search_tool
from ai_assistant.tools.factory import create_tool_registry


def _provider(contexts: tuple[str, str, list]):
    """A retrieval port that always answers with ``contexts``."""
    return SimpleNamespace(get_context=lambda query, result_k: contexts)


class RagSearchToolTests(unittest.TestCase):
    def test_empty_query_returns_error(self):
        tool = rag_search_tool(_provider(("doc", "code", [])))
        self.assertIn("trống", tool({"query": "  "})["error"])

    def test_unavailable_index_returns_not_ready(self):
        tool = rag_search_tool(None)
        self.assertIn("chưa sẵn sàng", tool({"query": "tìm kiếm"})["error"])

    def test_returns_doc_and_code_evidence(self):
        tool = rag_search_tool(_provider(("doc ctx", "code ctx", [])))
        result = tool({"query": "câu hỏi", "top_k": 3})
        self.assertTrue(result["found"])
        self.assertEqual(result["top_k"], 3)
        self.assertEqual([r["source"] for r in result["results"]], ["documentation", "source_code"])

    def test_no_matches_reports_not_found(self):
        tool = rag_search_tool(_provider(("", "", [])))
        result = tool({"query": "không có"})
        self.assertFalse(result["found"])

    def test_top_k_is_clamped(self):
        tool = rag_search_tool(_provider(("ctx", "", [])))
        result = tool({"query": "q", "top_k": 99})
        self.assertEqual(result["top_k"], 10)

    def test_engine_error_is_reported(self):
        def boom(query, result_k):
            msg = "index crashed"
            raise RuntimeError(msg)

        tool = rag_search_tool(SimpleNamespace(get_context=boom))
        self.assertIn("index crashed", tool({"query": "q"})["error"])

    def test_the_factory_binds_the_port_into_the_registered_handler(self):
        """The port is closed over at registration time: no module lookup
        happens when the agent actually calls the tool."""
        registry = create_tool_registry(rag_context=_provider(("doc ctx", "", [])))
        spec = registry.get("rag_search")
        self.assertIsNotNone(spec.handler)
        result = spec.handler({"query": "q"})
        self.assertTrue(result["found"])
        self.assertEqual(result["results"][0]["source"], "documentation")

    def test_a_registry_built_without_a_port_keeps_reporting_not_ready(self):
        registry = create_tool_registry()
        spec = registry.get("rag_search")
        self.assertIn("chưa sẵn sàng", spec.handler({"query": "q"})["error"])


if __name__ == "__main__":
    unittest.main()
