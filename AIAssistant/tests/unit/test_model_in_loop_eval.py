"""Unit tests for the model-in-the-loop eval's grading arithmetic.

``run_model_in_loop_eval.run_golden`` is the only place the SCORECARD #1 signal
("tỉ lệ eval có model") is computed, and it is deliberately injectable via
``run_fn`` so the arithmetic is testable without a 4 GB model on the machine.
These tests pin the two numbers it reports:

* ``accuracy`` — fraction of cases whose predicted step kind matched;
* ``model_coverage`` — fraction of cases the model produced a gradeable
  (non-``None``) kind for. A backend that returns empty/unparseable
  ``step_kinds`` for some prompts drops coverage below 100% even when a model
  loaded, because those cases were never really exercised.

The eval script lives outside ``src/`` (it is a runner, not library code) and
has no ``__init__.py``, so it is loaded here via ``importlib`` by path.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_EVALS = Path(__file__).resolve().parents[2] / "evals"


def _load_eval():
    spec = importlib.util.spec_from_file_location(
        "run_model_in_loop_eval", _EVALS / "run_model_in_loop_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CASES = [
    {"id": "c1", "lang": "en", "query": "q1", "expected_kind": "direct"},
    {"id": "c2", "lang": "en", "query": "q2", "expected_kind": "rag_search"},
]


class RunGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.eval = _load_eval()

    def _run(self, kinds_by_query: dict[str, object]):
        def run_fn(query: str):
            return kinds_by_query[query], 0.1
        return self.eval.run_golden(run_fn, cases=_CASES)

    def test_all_correct_scores_full_accuracy_and_coverage(self) -> None:
        result = self._run({"q1": "direct", "q2": "rag_search"})
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["model_coverage"], 1.0)
        self.assertEqual(result["passed"], 2)
        self.assertEqual(result["total"], 2)

    def test_wrong_answers_still_count_as_model_graded(self) -> None:
        """A wrong kind was still exercised by the model → coverage stays 1.0."""
        result = self._run({"q1": "rag_search", "q2": "rag_search"})
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["model_coverage"], 1.0)

    def test_ungradeable_answers_drop_model_coverage(self) -> None:
        """Empty step_kinds (None) means that case was never really graded."""
        result = self._run({"q1": "direct", "q2": None})
        self.assertEqual(result["model_coverage"], 0.5)
        # The graded case was correct, the ungradeable one was not a match.
        self.assertEqual(result["accuracy"], 0.5)

    def test_every_case_ungradeable_is_zero_coverage(self) -> None:
        result = self._run({"q1": None, "q2": None})
        self.assertEqual(result["model_coverage"], 0.0)
        self.assertEqual(result["accuracy"], 0.0)

    def test_an_empty_golden_set_reports_zero_not_a_divide_error(self) -> None:
        # ``run_golden`` falls back to ``GOLDEN`` for a falsy ``cases``, so the
        # zero-division guard is only reachable when the golden set itself is
        # empty; stub it out to exercise that path honestly.
        saved = self.eval.GOLDEN
        self.eval.GOLDEN = []
        try:
            result = self.eval.run_golden(lambda query: ("direct", 0.1))
        finally:
            self.eval.GOLDEN = saved
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["model_coverage"], 0.0)
        self.assertEqual(result["p95_latency"], 0.0)


if __name__ == "__main__":
    unittest.main()
