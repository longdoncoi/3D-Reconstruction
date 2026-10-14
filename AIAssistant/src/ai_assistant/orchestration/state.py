"""Agent state definition and type aliases for the orchestration graph."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, TypedDict

# ── Step vocabulary (typed-state slice, review #19) ─────────────────────────
#: Closed set of step discriminators. Read sites must compare through
#: :func:`has_step_type` so a typo (``"tool_cal"``) becomes a mypy error
#: instead of a routing condition that silently never matches. The element
#: type of ``AgentState.steps`` stays ``dict[str, Any]``: a TypedDict element
#: is *not* assignable to ``dict[str, Any]`` (mypy invariant), which would
#: force conversions at every boundary (``AgentRunResult``, HTTP payloads,
#: persistence) — that migration is a separate, dedicated slice.
StepType = Literal[
    "approval_granted",
    "cancelled",
    "coding_incomplete",
    "context_compacted",
    "delegation",
    "error",
    "final_answer",
    "pending_approval",
    "plan",
    "plan_incomplete",
    "plan_reflection",
    "reflection",
    "step_answer",
    "thinking",
    "tool_call",
    "tool_result",
    "validation_error",
    "verification",
]


def has_step_type(step: Mapping[str, Any], expected: StepType) -> bool:
    """True when *step* carries the discriminator *expected*."""
    return step.get("type") == expected


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

# ── Context-budget sentinel (ADR 0013) ──────────────────────────────────────
#: Returned by the runner's ``complete`` adapter when the conversation no
#: longer fits the model context window. This is a protocol between the runner
#: and the reason node, so the literal lives here and both import it — the
#: reason node must no longer guess the meaning from its own prefix list.
CONTEXT_EXHAUSTED_MESSAGE = "Context quá dài, dừng Agent."
#: Variants a model may emit on its own when it hits the budget.
CONTEXT_EXHAUSTED_PREFIXES = ("context quá dài", "context qua dai", "context too long")


def is_context_exhausted(answer: str) -> bool:
    """True when *answer* signals the model context budget is exhausted."""
    return answer.casefold().startswith(CONTEXT_EXHAUSTED_PREFIXES)


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
