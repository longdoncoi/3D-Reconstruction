"""Agent state definition and type aliases for the orchestration graph."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

# The step vocabulary (``StepType``/``Step``/``has_step_type``) is a domain
# contract — ``ai_assistant.domain.steps``, ADR 0019 — re-exported here
# because this package is its historical import path; application results and
# persistence must reach it inward, not through orchestration. The runtime
# twin ``validate_steps`` is *not* re-exported: its only caller is the
# persistence boundary, which imports the domain owner directly.
from ai_assistant.domain.steps import Step as Step
from ai_assistant.domain.steps import StepType as StepType
from ai_assistant.domain.steps import has_step_type as has_step_type


class AgentState(TypedDict):
    messages:        list[dict[str, str]]   # conversation history
    steps:           list[Step]             # steps list (for UI)
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
    #: Procedural memory (ADR 0043 §7): the plan written under this task's
    #: exact signature, offered by the service layer so ``plan_node`` can
    #: reuse it instead of calling the planner. ``None`` = no recipe.
    prior_plan:      list[str] | None
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
# The short-term context budgets are domain policy (ADR 0043): one owner in
# ``ai_assistant.domain.memory`` decides them, re-exported here because this
# package is the historical import path of the constants.
from ai_assistant.domain.memory import CONTEXT_MESSAGE_LIMIT as CONTEXT_MESSAGE_LIMIT
from ai_assistant.domain.memory import CONTEXT_SYSTEM_LIMIT as CONTEXT_SYSTEM_LIMIT
from ai_assistant.domain.memory import CONTEXT_TOTAL_LIMIT as CONTEXT_TOTAL_LIMIT
from ai_assistant.domain.memory import OBSERVATION_KEEP_LAST as OBSERVATION_KEEP_LAST
from ai_assistant.domain.memory import OBSERVATION_SUMMARY_THRESHOLD as OBSERVATION_SUMMARY_THRESHOLD
