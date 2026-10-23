"""Unit tests for the model-in-the-loop eval's grading arithmetic and gate.

``run_model_in_loop_eval.run_golden`` is the only place the SCORECARD #1 signal
("tỉ lệ eval có model") is computed, and it is deliberately injectable via
``run_fn`` so the arithmetic is testable without a 4 GB model on the machine.
These tests pin the two numbers it reports:

* ``accuracy`` — fraction of cases whose predicted step kind matched;
* ``model_coverage`` — fraction of cases the model produced a gradeable
  (non-``None``) kind for. A backend that returns empty/unparseable
  ``step_kinds`` for some prompts drops coverage below 100% even when a model
  loaded, because those cases were never really exercised.

The CI gate adds three more surfaces that must not drift: the payload parser
mirrors the production decode (an unparseable answer grades as "no kind", not
a crash), ``_gate_exit`` turns measured floors into an exit code, and ``main``
fails a requested gate when no model ran at all — a green run that graded
nothing would make the gate a lie.

The eval script lives outside ``src/`` (it is a runner, not library code) and
has no ``__init__.py``, so it is loaded here via ``importlib`` by path.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

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


class PlannerPayloadGradingTests(unittest.TestCase):
    """The payload parser mirrors ``orchestration/nodes/plan.py``: what would
    grade in production grades here, what would fall back there is ungradeable
    here — never a crash mid-run."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.eval = _load_eval()

    def test_a_valid_answer_declares_its_kind(self) -> None:
        self.assertEqual(
            self.eval._grade_planner_payload('{"step_kinds": ["rag_search"]}'),
            "rag_search",
        )

    def test_thinking_tags_are_stripped_before_parsing(self) -> None:
        raw = "<think>think first</think>```json\n{\"step_kinds\": [\"direct\"]}\n```"
        # The content after the think tags still has to decode on its own:
        # production json.loads would see the same bytes.
        self.assertIsNone(self.eval._grade_planner_payload(raw))

    def test_prose_that_is_not_json_declares_no_kind(self) -> None:
        self.assertIsNone(self.eval._grade_planner_payload("Tôi sẽ tìm kiếm dự án."))

    def test_a_non_object_json_declares_no_kind(self) -> None:
        self.assertIsNone(self.eval._grade_planner_payload('["rag_search"]'))

    def test_a_malformed_kinds_field_declares_no_kind(self) -> None:
        for raw in ('{"step_kinds": "direct"}', '{"step_kinds": []}',
                    '{"step_kinds": [3]}', '{"requires_plan": true}'):
            with self.subTest(raw=raw):
                self.assertIsNone(self.eval._grade_planner_payload(raw))


class _StubBackend:
    """Backend answering every planner call with one fixed content string."""

    model_description = "stub — test"

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    def generate(self, **kwargs) -> dict:
        self.calls += 1
        return {"choices": [{"message": {"content": self._content}}]}


class PlannerRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.eval = _load_eval()

    def test_a_gradeable_answer_returns_kind_and_latency(self) -> None:
        backend = _StubBackend('{"step_kinds": ["application_action"]}')
        kind, latency = self.eval.planner_run("query", backend)
        self.assertEqual(kind, "application_action")
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(backend.calls, 1)

    def test_an_unparseable_answer_is_ungradeable_not_a_crash(self) -> None:
        kind, _latency = self.eval.planner_run("query", _StubBackend("no json at all"))
        self.assertIsNone(kind)


class GateExitTests(unittest.TestCase):
    """Measured floors decide the exit code — and the log states both numbers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.eval = _load_eval()

    def _result(self, coverage: float, accuracy: float) -> dict:
        return {"model_coverage": coverage, "accuracy": accuracy, "passed": 0, "total": 4}

    def test_without_a_gate_every_case_must_pass(self) -> None:
        args = self.eval._parse_args([])
        self.assertEqual(self.eval._gate_exit({"passed": 4, "total": 4}, args), 0)
        self.assertEqual(self.eval._gate_exit({"passed": 3, "total": 4}, args), 1)

    def test_coverage_below_the_floor_fails_and_prints_the_measured_number(self) -> None:
        args = self.eval._parse_args(["--min-model-coverage", "0.9"])
        out = io.StringIO()
        with redirect_stdout(out):
            code = self.eval._gate_exit(self._result(0.75, 1.0), args)
        self.assertEqual(code, 1)
        self.assertIn("model_coverage=75%", out.getvalue())
        self.assertIn("required 90%", out.getvalue())

    def test_accuracy_below_the_floor_fails_even_with_full_coverage(self) -> None:
        args = self.eval._parse_args(["--min-model-coverage", "0.9",
                                      "--min-accuracy", "0.5"])
        with redirect_stdout(io.StringIO()):
            code = self.eval._gate_exit(self._result(1.0, 0.25), args)
        self.assertEqual(code, 1)

    def test_meeting_both_floors_passes(self) -> None:
        args = self.eval._parse_args(["--min-model-coverage", "0.9",
                                      "--min-accuracy", "0.5"])
        with redirect_stdout(io.StringIO()):
            code = self.eval._gate_exit(self._result(1.0, 0.75), args)
        self.assertEqual(code, 0)

    def test_coverage_gate_does_not_secretly_gate_accuracy(self) -> None:
        """CI's gate is model_coverage; accuracy stays reported, not enforced."""
        args = self.eval._parse_args(["--min-model-coverage", "0.9"])
        with redirect_stdout(io.StringIO()):
            code = self.eval._gate_exit(self._result(1.0, 0.0), args)
        self.assertEqual(code, 0)


class MainFlowTests(unittest.TestCase):
    """``main``: skip without a model, gate the skip when one was demanded,
    and write the graded report the CI artifact is built from."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.eval = _load_eval()

    def test_no_backend_without_a_gate_is_an_honest_skip(self) -> None:
        with patch.object(self.eval, "_load_backend", return_value=None):
            out = io.StringIO()
            with redirect_stdout(out):
                code = self.eval.main([])
        self.assertEqual(code, 0)
        self.assertIn("SKIP", out.getvalue())
        self.assertIn("model_coverage=0.00", out.getvalue())

    def test_no_backend_with_a_gate_fails_instead_of_passing_emptily(self) -> None:
        with patch.object(self.eval, "_load_backend", return_value=None):
            with redirect_stdout(io.StringIO()):
                code = self.eval.main(["--min-model-coverage", "0.9"])
        self.assertEqual(code, 1)

    def test_a_graded_run_writes_the_report_and_gates_on_coverage(self) -> None:
        backend = _StubBackend('{"step_kinds": ["direct"]}')  # only en_001 matches
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with (
                patch.object(self.eval, "_load_backend", return_value=backend),
                patch.object(self.eval, "REPORT_PATH", report),
            ):
                with redirect_stdout(io.StringIO()):
                    code = self.eval.main(["--model-path", "ignored-by-stub",
                                           "--min-model-coverage", "0.9"])
                self.assertEqual(code, 0)
                raw = report.read_text(encoding="utf-8")
                payload = json.loads(raw)
        self.assertEqual(payload["model_coverage"], 1.0)
        direct_cases = sum(1 for case in self.eval.GOLDEN
                           if case["expected_kind"] == "direct")
        self.assertEqual(payload["accuracy"], direct_cases / len(self.eval.GOLDEN))
        self.assertIn('"model": "ignored-by-stub"', raw)

    def test_model_path_loads_through_the_backend_class(self) -> None:
        # Load the real module first so the package __init__ has already
        # resolved its submodules: patching only sys.modules then cannot leave
        # a fake class behind in the parent package's namespace.
        importlib.import_module("ai_assistant.llm.llama_cpp_backend")

        created: dict = {}

        class _FakeBackend:
            def __init__(self, model_path: str, **kwargs) -> None:
                created["model_path"] = model_path

        fake_module = ModuleType("ai_assistant.llm.llama_cpp_backend")
        fake_module.LlamaCppBackend = _FakeBackend  # type: ignore[attr-defined]
        args = self.eval._parse_args(["--model-path", "cached.gguf"])
        with patch.dict(sys.modules, {"ai_assistant.llm.llama_cpp_backend": fake_module}):
            backend = self.eval._load_backend(args)
        self.assertEqual(created["model_path"], "cached.gguf")
        self.assertIsInstance(backend, _FakeBackend)

    def test_without_model_path_there_is_no_implicit_backend(self) -> None:
        args = self.eval._parse_args([])
        self.assertIsNone(self.eval._load_backend(args))


class BaselineRegressionTests(unittest.TestCase):
    """``--baseline`` turns an earlier report into a model-version regression gate."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.eval = _load_eval()

    def _result(self, coverage: float = 1.0, accuracy: float = 1.0) -> dict:
        return {
            "model_coverage": coverage, "accuracy": accuracy,
            "passed": 7, "total": 7,
        }

    def test_no_regression_produces_no_failures(self) -> None:
        self.assertEqual(
            self.eval._baseline_failures(
                self._result(), {"accuracy": 1.0, "model_coverage": 1.0}, 0.05),
            [],
        )

    def test_a_drop_within_the_tolerance_still_passes(self) -> None:
        failures = self.eval._baseline_failures(
            self._result(accuracy=0.95), {"accuracy": 1.0}, 0.05)
        # Exactly at the tolerance is not "more than" it.
        self.assertEqual(failures, [])

    def test_a_drop_beyond_the_tolerance_fails_every_metric(self) -> None:
        failures = self.eval._baseline_failures(
            self._result(0.8, 0.75),
            {"accuracy": 1.0, "model_coverage": 1.0}, 0.05)
        self.assertEqual(len(failures), 2)
        text = " ".join(failures)
        self.assertIn("accuracy=75%", text)
        self.assertIn("model_coverage=80%", text)
        self.assertIn("baseline 100%", text)

    def test_metrics_missing_from_the_baseline_are_ignored(self) -> None:
        self.assertEqual(
            self.eval._baseline_failures(self._result(0.1, 0.1), {"accuracy": 0.0}, 0.05),
            [],
        )

    def test_baseline_failures_fail_the_exit_code_even_when_all_cases_pass(self) -> None:
        args = self.eval._parse_args([])
        out = io.StringIO()
        with redirect_stdout(out):
            code = self.eval._gate_exit(
                self._result(), args, ["accuracy=50% is 40% below the baseline"])
        self.assertEqual(code, 1)
        self.assertIn("GATE FAIL: accuracy=50%", out.getvalue())

    def _run_main(self, baseline: Path, *extra: str):
        backend = _StubBackend('{"step_kinds": ["direct"]}')
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with (
                patch.object(self.eval, "_load_backend", return_value=backend),
                patch.object(self.eval, "REPORT_PATH", report),
            ):
                out = io.StringIO()
                with redirect_stdout(out):
                    code = self.eval.main(
                        ["--model-path", "ignored-by-stub", "--baseline", str(baseline), *extra])
        return code, out.getvalue()

    def test_main_fails_when_accuracy_drops_below_a_stored_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            baseline.write_text(json.dumps({"accuracy": 0.9, "model_coverage": 0.9}),
                                encoding="utf-8")
            code, output = self._run_main(baseline)
        self.assertEqual(code, 1)
        self.assertIn("below the baseline", output)

    def test_a_missing_baseline_file_exits_2_before_loading_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.json"
            with patch.object(self.eval, "_load_backend") as load:
                with redirect_stdout(io.StringIO()):
                    code = self.eval.main(["--baseline", str(missing)])
        self.assertEqual(code, 2)
        load.assert_not_called()

    def test_a_non_object_baseline_file_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            baseline.write_text("[]", encoding="utf-8")
            with patch.object(self.eval, "_load_backend") as load:
                with redirect_stdout(io.StringIO()):
                    code = self.eval.main(["--baseline", str(baseline)])
        self.assertEqual(code, 2)
        load.assert_not_called()

    def test_a_baseline_counts_as_a_gate_without_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            baseline.write_text("{}", encoding="utf-8")
            with patch.object(self.eval, "_load_backend", return_value=None):
                with redirect_stdout(io.StringIO()):
                    code = self.eval.main(["--baseline", str(baseline)])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
