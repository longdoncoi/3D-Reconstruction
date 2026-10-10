"""Unit tests for the ``tool_rag_search`` builtin tool.

The tool forwards to the legacy ``rag_module`` compatibility shim, whose real
module pulls in optional ML dependencies. A fake is injected into
``sys.modules`` instead, so these tests run in the offline CI environment.
"""
from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import ai_assistant.legacy as legacy_pkg
from ai_assistant.tools.builtin import rag_tools

_FAKE_RAG_PATH = "ai_assistant.legacy.rag_module"


class RagSearchToolTests(unittest.TestCase):
    def _patch_rag(self, fake):
        # Patch both the sys.modules entry (fresh process) and the package
        # attribute (once another test has imported the real shim) so the
        # function-level ``from ai_assistant.legacy import rag_module`` always
        # sees the fake regardless of test ordering.
        patchers = [
            mock.patch.dict(sys.modules, {_FAKE_RAG_PATH: fake}),
            mock.patch.object(legacy_pkg, "rag_module", fake, create=True),
        ]
        for patcher in patchers:
            patcher.start()
        self.addCleanup(lambda: [p.stop() for p in patchers])

    def test_empty_query_returns_error(self):
        self.assertIn("trống", rag_tools.tool_rag_search({"query": "  "})["error"])

    def test_unavailable_rag_module_returns_not_ready(self):
        fake = SimpleNamespace()
        self._patch_rag(fake)
        self.assertIn(
            "chưa sẵn sàng",
            rag_tools.tool_rag_search({"query": "tìm kiếm"})["error"],
        )

    def test_returns_doc_and_code_evidence(self):
        fake = SimpleNamespace(get_context=lambda query, result_k: ("doc ctx", "code ctx", []))
        self._patch_rag(fake)
        result = rag_tools.tool_rag_search({"query": "câu hỏi", "top_k": 3})
        self.assertTrue(result["found"])
        self.assertEqual(result["top_k"], 3)
        self.assertEqual([r["source"] for r in result["results"]], ["documentation", "source_code"])

    def test_no_matches_reports_not_found(self):
        fake = SimpleNamespace(get_context=lambda query, result_k: ("", "", []))
        self._patch_rag(fake)
        result = rag_tools.tool_rag_search({"query": "không có"})
        self.assertFalse(result["found"])

    def test_top_k_is_clamped(self):
        fake = SimpleNamespace(get_context=lambda query, result_k: ("ctx", "", []))
        self._patch_rag(fake)
        result = rag_tools.tool_rag_search({"query": "q", "top_k": 99})
        self.assertEqual(result["top_k"], 10)

    def test_engine_error_is_reported(self):
        def boom(query, result_k):
            raise RuntimeError("index crashed")

        fake = SimpleNamespace(get_context=boom)
        self._patch_rag(fake)
        self.assertIn("index crashed", rag_tools.tool_rag_search({"query": "q"})["error"])


if __name__ == "__main__":
    unittest.main()
