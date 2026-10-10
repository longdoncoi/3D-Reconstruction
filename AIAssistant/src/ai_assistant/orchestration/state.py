"""Agent state definition and type aliases for the orchestration graph."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict


class AgentState(TypedDict):
    messages:        list[dict[str, str]]   # conversation history
    steps:           list[dict[str, Any]]   # steps list (for UI)
    iteration:       int                    # loop iteration count
    temperature:     float                  # LLM generation temperature
    done:            bool                   # has the agent finished
    pending_tool:    dict[str, Any] | None  # tool waiting for user approval
    plan:            list[str] | None       # plan steps
    plan_spec:       dict[str, Any] | None  # structured planner contract
    approval_granted: bool                  # an approval was granted for this run
    granted_fingerprint: str                # exact tool+params the approval covers
    approval_scope:   str                   # stable hash of task/plan scope
    cancelled:        bool                  # cooperative cancellation requested
    tool_call_count: int                    # number of tool calls (triggers summarisation)
    last_reflection: dict[str, Any] | None  # deterministic critic result
    error_count:     int                    # consecutive tool errors
    resume_with_reflection: bool            # resume after an async UI ACK
    skip_reflect:    bool                   # verified low-risk tool, auto-reviewed
    synthesize_after_rag: bool              # route verified RAG evidence back to Reason
    required_ui_actions: list[dict[str, Any]]  # canonical UI actions from host
    plan_verified:    bool                     # current plan passed Plan Reflect
    plan_attempts:    int                      # number of Planner iterations
    plan_feedback:   str                       # last feedback to Planner
    enforce_plan_completion: bool              # coding tasks must complete plan steps
    enforce_coding_workflow: bool              # coding tasks need observable evidence


# ── Callback type aliases ──────────────────────────────────────────────────────

Completion       = Callable[[list[dict[str, str]], float], str]
Parser           = Callable[[str], tuple[str | None, dict[str, Any] | None]]
Executor         = Callable[[str, dict[str, Any]], dict[str, Any]]
NeedsApproval    = Callable[[str], bool]
ApprovalCovers   = Callable[[str, dict[str, Any]], bool]
SelectSpecialist = Callable[[str, dict[str, Any]], dict[str, Any]]
VerifyResult     = Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]]
ReflectResult    = Callable[[str, dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]]


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_STEP_MISMATCH_REJECTIONS = 2
COMPLETION_TOOLS = {"application_action"}
LOW_RISK_TOOLS = {
    "read_file", "list_directory", "find_files", "search_text", "analyze_code",
    "git_diff", "get_project_status", "validate_file", "rag_search",
}
SEMANTIC_REFLECTION_TOOLS = {
    "run_command", "write_file", "patch_file", "replace_file_content",
    "multi_replace_file_content", "create_directory", "application_action",
}
OBSERVATION_SUMMARY_THRESHOLD = 4   # summarise observations after N tool calls
OBSERVATION_KEEP_LAST = 10          # keep last N tool-result messages
CONTEXT_SYSTEM_LIMIT  = 3500        # reason node adds ~1k REASONER_PROMPT on top
CONTEXT_MESSAGE_LIMIT = 4000        # user + evidence must both fit
CONTEXT_TOTAL_LIMIT   = 11000       # hard compaction limit before calling LLM
