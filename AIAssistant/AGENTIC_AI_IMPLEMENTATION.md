# Agentic AI roadmap completion

This document records the implementation boundary for `Agentic_AI_Roadmap.md`.

| Phase | Delivered capability | Verification |
| --- | --- | --- |
| 1. Unified Assistant | `/v1/agent/execute` accepts task, history, attachments, language and session context; JSON and SSE use the same route; Qt UI actions complete only after ACK. | `test_agent_actions.py`, `test_agent_ui_continuation.py` |
| 2. Supervisor | Bounded Plan → Reason → Tool → Reflect graph, Pydantic validation, per-tool timeout/policy/approval metadata, audit events and deterministic verification. | `test_langgraph_multi_agent.py`, `run_agentic_eval.py` |
| 3. Specialists | Supervisor-only delegation to Chatbot, Research, Desktop Workflow, Code and Verification specialists. Specialists do not call each other. | `test_multi_agent.py`, `test_agent_reflection.py` |
| 4. Reliability | Checkpoint backends, resume cursor, circuit breaker, role-specific logs, Prometheus/OpenTelemetry hooks, schema-error and approval metrics, and deterministic evals. | `ruff check .`, `compileall`, unit tests and evals |

## Code Agent toolbox

Production coordination is transport-neutral: a Task Coordinator keyed by
session/request ids supports cooperative cancellation, lifecycle state and
resume across JSON, SSE and Qt. Planner output carries a structured goal,
affected areas, acceptance criteria and verification commands. Approval shows
a plan-scoped preview and grants only the unchanged plan scope. Discovery
recovery and selective Reflect are driven by tool contracts and result
evidence, with no DICOM/OBJ-specific dispatch branches.

The Code Agent has the following complete toolbox:

- Discovery: `find_files`, `list_directory`, `search_text`, `read_file` (line or symbol scope), `analyze_code`, `git_diff`.
- Verification: `get_project_status`, `validate_file`.
- Change and execution: `write_file`, `patch_file`, `create_directory`, `run_command`.

Discovery tools are read-only and exposed through MCP Streamable HTTP. Change
and execution tools are approval-gated, allow-listed, bounded and audited.
Every tool definition carries a JSON Schema, timeout, policy and idempotency
flag; the same catalog drives local constrained decoding and MCP discovery.

Coding intent has priority over desktop intent. A request such as “fix DICOM
loading and add tests” is routed to Code Agent even though it mentions a Qt
load action; UI hints are disabled for that route. Coding requests also use
accent-insensitive Vietnamese intent matching, bounded directory discovery and
cannot finish while validated plan steps remain. Analysis requests use focused
source reads, while explicit citation requests use AST-bounded Python ranges.

Change requests also have a domain-independent completion gate: the Agent must
observe relevant source evidence, an approved mutation, a post-change
`git_diff`, and a verification command result before it can emit a completion
answer. This applies equally to new features, bug fixes and refactors without
branching on a feature name or file format.

## Current verification command

From `AIAssistant/`:

```powershell
ruff check .
python -m compileall -q StartChatbotServer.py evals src
python -m unittest discover -s tests -p 'test_*.py'
python evals/run_agentic_eval.py
python evals/run_tool_eval.py
```
