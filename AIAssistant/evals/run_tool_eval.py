"""Deterministic regression eval for constrained agent tool envelopes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_assistant.tools.tool_contract import build_tool_models, validate_tool_call

TOOLS = [
    {"name": "application_action", "parameters": {"action": {"type": "string", "required": True},
     "language": {"type": "string", "required": False}, "username": {"type": "string", "required": False},
     "password": {"type": "string", "required": False}}},
    {"name": "read_file", "parameters": {"path": {"type": "string", "required": True}}},
    {"name": "find_files", "parameters": {"pattern": {"type": "string", "required": True},
     "path": {"type": "string", "required": False}, "max_results": {"type": "integer", "required": False}}},
    {"name": "git_diff", "parameters": {"path": {"type": "string", "required": False},
     "staged": {"type": "boolean", "required": False}}},
]


def main() -> int:
    cases = json.loads((Path(__file__).parent / "tool_call_cases.json").read_text(encoding="utf-8"))
    models = build_tool_models(TOOLS)
    passed = 0
    details = []
    for case in cases:
        envelope = json.loads(case["response"])
        if envelope["kind"] == "final":
            ok = case["tool"] is None
        else:
            params, error = validate_tool_call(envelope["tool"], envelope["params"], models)
            ok = error is None and envelope["tool"] == case["tool"] and (
                "action" not in case or params["action"] == case["action"])
        details.append({"name": case["name"], "passed": ok})
        passed += int(ok)
    report = {"total": len(cases), "passed": passed, "accuracy": passed / len(cases), "cases": details}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    (ROOT / "evals" / "latest_tool_call_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
