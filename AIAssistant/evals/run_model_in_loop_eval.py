"""Model-in-the-loop eval over a fixed golden set (Sprint 2).

Honest by construction (ADR 0015):

* the golden prompts are literals in this file, written against the planner's
  documented three-way vocabulary (``direct`` / ``rag_search`` /
  ``application_action``), not derived from the code under test;
* grading runs the *real* planner — ``PLANNER_PROMPT`` through the loaded
  backend — and parses the JSON the planner is asked for, instead of a rule
  classifier;
* with no backend loaded the script prints ``SKIP`` and exits 0 **without**
  reporting an accuracy figure. The previous version called a rule classifier
  (since deleted as dead code) whose vocabulary (``coding``/``rag``/``direct``)
  could never equal these expectations, so every run produced a number that was
  wrong in one direction or the other.

Tool selection is graded separately by ``run_tool_eval.py``; this file grades
only the planner's step kind and its p95 latency.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
# The package lives under ``src/`` (ADR 0010 layout).
sys.path.insert(0, str(ROOT / "src"))

GOLDEN = [
    {"id": "vn_001", "lang": "vi", "query": "Dự án này có chức năng gì?",
     "expected_kind": "rag_search"},
    {"id": "vn_002", "lang": "vi", "query": "Tìm lỗi trong file src/app/AppConstants.h",
     "expected_kind": "rag_search"},
    {"id": "en_001", "lang": "en", "query": "What is the benefit of AI Agent?",
     "expected_kind": "direct"},
    {"id": "en_002", "lang": "en", "query": "Build the 3D reconstruction pipeline",
     "expected_kind": "application_action"},
]

# ``run_fn(query) -> (kind, latency_seconds)``: injected so the grading logic
# itself is testable without a 4 GB model on the machine.
RunFn = Callable[[str], tuple[Any, float]]


def _response_content(response: Any) -> str:
    """Read ``choices[0].message.content`` from a dict or an object payload."""
    if isinstance(response, dict):
        choices = response.get("choices") or [{}]
        message = choices[0].get("message", {})
        return str(message.get("content", ""))
    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    return str(getattr(message, "content", "") or "")


def planner_run(query: str, backend: Any, *, max_tokens: int = 384,
                temperature: float = 0.0) -> tuple[Any, float]:
    """Ask the loaded backend the planner question and parse its step kinds."""
    from ai_assistant.llm.inference import strip_think_tags
    from ai_assistant.orchestration.prompts import PLANNER_PROMPT

    started = time.monotonic()
    response = backend.generate(
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": f"Yêu cầu hiện tại: {query}"},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    latency = time.monotonic() - started

    payload = json.loads(strip_think_tags(_response_content(response)))
    kinds = payload.get("step_kinds") or []
    # A single-step answer still has to declare its kind; an empty list is an
    # ungradeable answer and is reported as ``None`` rather than guessed.
    return (kinds[0] if kinds else None), latency


def run_golden(run_fn: RunFn, cases: list[dict] | None = None) -> dict:
    """Grade the planner against the golden set.

    ``accuracy`` is kind accuracy over all cases. ``model_coverage`` is the
    SCORECARD #1 signal: the fraction of cases the model actually produced a
    gradeable (non-``None``) step kind for. A backend that returns empty or
    unparseable ``step_kinds`` for some prompts drops coverage below 100% even
    though a model loaded — which is the honest reading, because those cases
    were never really exercised.
    """
    results = []
    for item in cases or GOLDEN:
        kind, latency = run_fn(item["query"])
        results.append({
            "id": item["id"],
            "ok": kind == item["expected_kind"],
            "expected": item["expected_kind"],
            "kind": kind,
            "latency": latency,
        })
    total = len(results)
    passed = sum(1 for result in results if result["ok"])
    graded = sum(1 for result in results if result["kind"] is not None)
    latencies = sorted(result["latency"] for result in results)
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0
    return {
        "total": total,
        "passed": passed,
        "accuracy": (passed / total) if total else 0.0,
        "model_coverage": (graded / total) if total else 0.0,
        "p95_latency": p95,
        "results": results,
    }


if __name__ == "__main__":
    from ai_assistant.llm.model_loader import get_backend

    backend = get_backend()
    if backend is None:
        # No model: report the skip, never a synthetic 0% or 100%. SCORECARD #1
        # ("tỉ lệ eval có model") reads 0 here — this run exercised no model.
        print("SKIP: no backend loaded — start the server (or load a model) first.")
        print("model_coverage=0.00 (no backend; 0 of the golden set was model-graded)")
        raise SystemExit(0)

    result = run_golden(lambda query: planner_run(query, backend))
    print(
        f"kind accuracy={result['accuracy']:.0%} "
        f"({result['passed']}/{result['total']}), "
        f"model_coverage={result['model_coverage']:.0%}, "
        f"p95_latency={result['p95_latency']:.3f}s"
    )
    print(json.dumps(result["results"], ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] == result["total"] else 1)
