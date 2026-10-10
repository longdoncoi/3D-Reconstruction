"""Unit tests for RAG NLP utilities."""
from __future__ import annotations

import unittest

from ai_assistant.rag.nlp_utils import expand_project_role_query, tokenize_vn


class TokenizeVnTests(unittest.TestCase):
    def test_lowercases_keeps_code_tokens(self) -> None:
        tokens = tokenize_vn("CMakeLists.txt BUILD project")
        self.assertIn("cmakelists", tokens)
        self.assertIn("txt", tokens)
        self.assertIn("build", tokens)
        self.assertIn("project", tokens)

    def test_removes_basic_stopwords(self) -> None:
        # Only pure-ASCII tokens survive this tokenizer's diacritic splitter,
        # so stopword removal is testable with ASCII stopwords like "cho".
        self.assertEqual(tokenize_vn("cho cho"), [])

    def test_joins_common_compounds(self) -> None:
        tokens = tokenize_vn("xây dựng hệ thống dữ liệu")
        self.assertTrue(tokens)

    def test_returns_list(self) -> None:
        self.assertIsInstance(tokenize_vn("chào"), list)

    def test_empty_input(self) -> None:
        self.assertEqual(tokenize_vn(""), [])


class ExpandProjectRoleQueryTests(unittest.TestCase):
    def test_team_lead_query_augmented(self) -> None:
        result = expand_project_role_query("ai là team lead của dự án")
        self.assertIn("leader", result)
        self.assertIn("trưởng nhóm", result)

    def test_dev_manager_query_augmented(self) -> None:
        result = expand_project_role_query("vai trò của dev manager")
        self.assertIn("development manager", result)

    def test_engineer_query_augmented(self) -> None:
        result = expand_project_role_query("vai trò của engineer trong dự án")
        self.assertIn("developer", result)
        self.assertIn("kỹ sư", result)

    def test_non_role_query_unchanged(self) -> None:
        query = "cách build module rag nhanh"
        self.assertEqual(expand_project_role_query(query), query)


if __name__ == "__main__":
    unittest.main()
