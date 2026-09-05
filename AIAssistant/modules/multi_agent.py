"""Policy-first supervisor and specialists for the unified assistant."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from .action_manifest import looks_like_ui_action
from .agent_logging import get_agent_logger
from .coding_agent import is_coding_task
from .config import APP_DATA_DIR, logger

supervisor_logger = get_agent_logger("supervisor")
verification_logger = get_agent_logger("verification")


class Specialist(StrEnum):
    SUPERVISOR = "supervisor"
    CHATBOT = "chatbot"
    TOOLAPP = "toolapp"
    RESEARCH = "research"
    WORKFLOW = "desktop_workflow"
    CODE = "code"
    VERIFICATION = "verification"


@dataclass(frozen=True)
class Delegation:
    specialist: Specialist
    tool: str | None
    reason: str
    idempotency_key: str


_SPECIALIST_INSTRUCTIONS = {
    Specialist.CHATBOT: "Answer conversationally using the Chatbot Agent and its RAG context only.",
    Specialist.TOOLAPP: "Dispatch only canonical Qt application_action calls and wait for acknowledgement.",
    Specialist.RESEARCH: "Inspect only the smallest relevant evidence; cite files and do not change state.",
    Specialist.WORKFLOW: "Dispatch only a canonical desktop action and wait for the desktop acknowledgement.",
    Specialist.CODE: "Describe the smallest safe change and require explicit approval before changing project state.",
    Specialist.VERIFICATION: "Independently check observable output; report a failed check instead of assuming success.",
    Specialist.SUPERVISOR: "Clarify intent, enforce policy, and coordinate the next specialist handoff.",
}


def route_task(task: str, channel: str = "agent") -> Specialist:
    """Supervisor routing decision made before a specialist receives context."""
    if channel == "chatbot":
        supervisor_logger.info("Supervisor route | channel=chatbot -> chatbot")
        return Specialist.CHATBOT
    if is_coding_task(task):
        supervisor_logger.info("Supervisor route | coding | task=%s", task[:160])
        return Specialist.CODE
    if looks_like_ui_action(task):
        supervisor_logger.info("Supervisor route | toolapp | task=%s", task[:160])
        return Specialist.TOOLAPP
    supervisor_logger.info("Supervisor route | unresolved -> supervisor | task=%s", task[:160])
    return Specialist.SUPERVISOR


_RESEARCH_TOOLS = {"read_file", "list_directory", "find_files", "search_text", "analyze_code", "git_diff", "rag_search"}
_VERIFICATION_TOOLS = {"validate_file", "get_project_status"}
_CODE_TOOLS = {"write_file", "patch_file", "replace_file_content", "multi_replace_file_content", "create_directory", "run_command"}
CODE_AGENT_TOOLS = frozenset({
    "find_files", "list_directory", "search_text", "read_file", "analyze_code", "git_diff",
    "get_project_status", "validate_file", "write_file", "patch_file",
    "replace_file_content", "multi_replace_file_content", "create_directory", "run_command",
})
_WORKFLOW_TOOLS = {"application_action"}
_AUDIT_LOCK = threading.Lock()
_AUDIT_PATH = os.path.join(APP_DATA_DIR, "AIAssistant", "agent_audit.jsonl")


def delegate(task: str, session_id: str, tool: str | None = None, parameters: dict[str, Any] | None = None,
             prefer_code: bool = False) -> Delegation:
    """Select exactly one worker role. Workers never delegate further."""
    if prefer_code and tool in (_RESEARCH_TOOLS | _VERIFICATION_TOOLS | _CODE_TOOLS):
        specialist, reason = Specialist.CODE, "coding task requires isolated repository context"
    elif tool in _RESEARCH_TOOLS:
        specialist, reason = Specialist.RESEARCH, "tool is read/RAG-only"
    elif tool in _VERIFICATION_TOOLS:
        specialist, reason = Specialist.VERIFICATION, "tool validates an observable result"
    elif tool in _CODE_TOOLS:
        specialist, reason = Specialist.CODE, "tool can change project state and requires approval"
    elif tool in _WORKFLOW_TOOLS:
        specialist, reason = Specialist.WORKFLOW, "tool dispatches a Qt desktop workflow"
    else:
        specialist, reason = Specialist.SUPERVISOR, "direct response or unresolved intent"
    source = json.dumps({"session": session_id, "task": task, "tool": tool, "params": parameters or {}},
                        ensure_ascii=False, sort_keys=True, default=str)
    delegation = Delegation(specialist, tool, reason, hashlib.sha256(source.encode("utf-8")).hexdigest()[:24])
    # Keep a role-specific trace in addition to the supervisor aggregate log.
    # This creates agent_<role>.log lazily for research/workflow/code roles too.
    get_agent_logger(specialist.value).info(
        "Delegated | tool=%s reason=%s idempotency_key=%s",
        tool or "none", reason, delegation.idempotency_key,
    )
    return delegation


def authorise(delegation: Delegation, needs_approval: bool) -> tuple[bool, str | None]:
    """Enforce least privilege before the host calls a tool."""
    if delegation.tool is None:
        return True, None
    if delegation.specialist is Specialist.CODE and delegation.tool in _CODE_TOOLS and not needs_approval:
        return False, "Code tools must be routed through the approval workflow."
    if delegation.specialist is Specialist.SUPERVISOR:
        return False, f"No specialist policy exists for tool: {delegation.tool}"
    return True, None


def verify_result(delegation: Delegation, result: Any) -> dict[str, Any]:
    """Independent result check consumed by the verification specialist."""
    verification_logger.info("Verification start | specialist=%s tool=%s", delegation.specialist, delegation.tool)
    if not isinstance(result, dict):
        outcome = {"passed": False, "reason": "Tool result is not an object."}
        verification_logger.warning("Verification failed | tool=%s reason=%s", delegation.tool, outcome["reason"])
        return outcome
    if result.get("error"):
        outcome = {"passed": False, "reason": str(result["error"])}
        verification_logger.warning("Verification failed | tool=%s reason=%s", delegation.tool, outcome["reason"])
        return outcome
    if result.get("success") is False:
        outcome = {"passed": False, "reason": "Tool reported an unsuccessful operation."}
        verification_logger.warning("Verification failed | tool=%s reason=%s", delegation.tool, outcome["reason"])
        return outcome
    if result.get("return_code", 0) != 0:
        outcome = {"passed": False, "reason": f"Command exited with code {result['return_code']}."}
        verification_logger.warning("Verification failed | tool=%s reason=%s", delegation.tool, outcome["reason"])
        return outcome

    # Read-only discovery is useful only when it produces an observation the
    # reasoner can act on.  These field names are the public result contract of
    # the generic tools, not feature-specific rules.
    evidence_fields = {
        "read_file": ("content",),
        "search_text": ("results",),
        "find_files": ("matches",),
        "list_directory": ("entries",),
        "analyze_code": ("functions", "classes", "includes"),
        "git_diff": ("content",),
    }
    fields = evidence_fields.get(str(delegation.tool))
    if fields and not any(bool(result.get(field)) for field in fields):
        outcome = {"passed": False, "reason": "Tool completed but returned no usable evidence."}
        verification_logger.warning("Verification failed | tool=%s reason=%s", delegation.tool, outcome["reason"])
        return outcome
    if delegation.specialist is Specialist.WORKFLOW:
        outcome = {"passed": bool(result.get("pending_ui_ack") or result.get("success")),
                   "reason": "Qt acknowledgement is required for desktop actions."}
    else:
        outcome = {"passed": True, "reason": "Structured tool result passed policy checks."}
    verification_logger.info("Verification result | tool=%s passed=%s reason=%s",
                             delegation.tool, outcome["passed"], outcome["reason"])
    return outcome


def specialist_instruction(delegation: Delegation) -> str:
    """Return a compact role handoff for the shared local model.

    Specialists are logical roles, not separate model processes: this retains
    role separation without multiplying model context or token cost.
    """
    return _SPECIALIST_INSTRUCTIONS[delegation.specialist]


def reflect_result(delegation: Delegation, result: Any, verification: dict[str, Any]) -> dict[str, Any]:
    """Independent, deterministic critic used after each tool result.

    The next ReAct turn receives this feedback and revises its approach when a
    tool or verification failed.  Keeping the critic deterministic avoids an
    additional LLM call for every tool invocation.
    """
    if not isinstance(result, dict):
        outcome = {"passed": False, "decision": "revise", "reason": "Tool returned an unstructured result."}
    elif result.get("error"):
        outcome = {"passed": False, "decision": "revise", "reason": f"Tool failed: {result['error']}"}
    elif not verification.get("passed"):
        outcome = {"passed": False, "decision": "revise", "reason": verification.get("reason", "Verification failed.")}
    else:
        outcome = {"passed": True, "decision": "continue", "reason": "Independent verification accepted the tool result."}
    verification_logger.info("Reflection result | tool=%s passed=%s decision=%s reason=%s",
                             delegation.tool, outcome["passed"], outcome["decision"], outcome["reason"])
    return outcome


def audit(event: str, delegation: Delegation, **details: Any) -> None:
    """Append audit data without ever breaking a user request on I/O failure."""
    record = {"timestamp": time.time(), "event": event, "delegation": asdict(delegation), **details}
    try:
        os.makedirs(os.path.dirname(_AUDIT_PATH), exist_ok=True)
        with _AUDIT_LOCK, open(_AUDIT_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as error:
        logger.warning("Unable to write agent audit event: %s", error)
