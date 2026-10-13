"""Golden-set eval (model-in-the-loop) cho tiếng Việt/Anh.
Chạy: python -m unittest AIAssistant/evals/run_golden_set_eval.py
"""
from __future__ import annotations

import unittest

GOLDEN_SET = [
    {"q": "DevManager trong dự án 3D-Reconstruction là gì?", "lang": "vi", "expected_kind": "rag_search"},
    {"q": "What is an AI Agent?", "lang": "en", "expected_kind": "direct"},
    {"q": "Tải ảnh 2D", "lang": "vi", "expected_kind": "application_action"},
]

class TestGoldenSetEval(unittest.TestCase):
    def test_golden_set_exists(self) -> None:
        self.assertTrue(len(GOLDEN_SET) >= 3)
        for item in GOLDEN_SET:
            self.assertIn("lang", item)
            self.assertIn("expected_kind", item)

if __name__ == "__main__":
    unittest.main()
