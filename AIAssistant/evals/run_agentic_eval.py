"""Deterministic policy eval for supervisor routing and safety boundaries."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.multi_agent import authorise, delegate, verify_result

CASES = [
    ("search_text", {}, "research", True),
    ("rag_search", {}, "research", True),
    ("application_action", {}, "desktop_workflow", True),
    ("validate_file", {}, "verification", True),
    ("write_file", {}, "code", False),
    ("unknown", {}, "supervisor", False),
]


def main() -> int:
    results = []
    for tool, params, expected_agent, expected_allowed in CASES:
        delegation = delegate("evaluation", "eval", tool, params)
        allowed, _ = authorise(delegation, tool == "write_file")
        # Code tools are approved by the UI before host execution.
        if tool == "write_file":
            allowed, _ = authorise(delegation, False)
        verified = verify_result(delegation, {"success": True})
        passed = str(delegation.specialist) == expected_agent and allowed == expected_allowed and verified["passed"]
        results.append({"tool": tool, "passed": passed, "specialist": str(delegation.specialist)})
    report = {"total": len(results), "passed": sum(r["passed"] for r in results), "cases": results}
    report["accuracy"] = report["passed"] / report["total"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
