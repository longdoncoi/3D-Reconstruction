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

from .config import APP_DATA_DIR, logger


class Specialist(StrEnum):
    SUPERVISOR = "supervisor"
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
    Specialist.RESEARCH: "Inspect only the smallest relevant evidence; cite files and do not change state.",
    Specialist.WORKFLOW: "Dispatch only a canonical desktop action and wait for the desktop acknowledgement.",
    Specialist.CODE: "Describe the smallest safe change and require explicit approval before changing project state.",
    Specialist.VERIFICATION: "Independently check observable output; report a failed check instead of assuming success.",
    Specialist.SUPERVISOR: "Clarify intent, enforce policy, and coordinate the next specialist handoff.",
}


_RESEARCH_TOOLS = {"read_file", "list_directory", "search_text", "analyze_code", "rag_search"}
_VERIFICATION_TOOLS = {"validate_file", "get_project_status"}
_CODE_TOOLS = {"write_file", "patch_file", "create_directory", "run_command"}
_WORKFLOW_TOOLS = {"application_action"}
_AUDIT_LOCK = threading.Lock()
_AUDIT_PATH = os.path.join(APP_DATA_DIR, "AIAssistant", "agent_audit.jsonl")


def delegate(task: str, session_id: str, tool: str | None = None, parameters: dict[str, Any] | None = None) -> Delegation:
    """Select exactly one worker role. Workers never delegate further."""
    if tool in _RESEARCH_TOOLS:
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
    return Delegation(specialist, tool, reason, hashlib.sha256(source.encode("utf-8")).hexdigest()[:24])


def authorise(delegation: Delegation, needs_approval: bool) -> tuple[bool, str | None]:
    """Enforce least privilege before the host calls a tool."""
    if delegation.tool is None:
        return True, None
    if delegation.specialist is Specialist.CODE and not needs_approval:
        return False, "Code tools must be routed through the approval workflow."
    if delegation.specialist is Specialist.SUPERVISOR:
        return False, f"No specialist policy exists for tool: {delegation.tool}"
    return True, None


def verify_result(delegation: Delegation, result: Any) -> dict[str, Any]:
    """Independent result check consumed by the verification specialist."""
    if not isinstance(result, dict):
        return {"passed": False, "reason": "Tool result is not an object."}
    if result.get("error"):
        return {"passed": False, "reason": str(result["error"])}
    if delegation.specialist is Specialist.WORKFLOW:
        return {"passed": bool(result.get("pending_ui_ack") or result.get("success")),
                "reason": "Qt acknowledgement is required for desktop actions."}
    return {"passed": True, "reason": "Structured tool result passed policy checks."}


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
        return {"passed": False, "decision": "revise", "reason": "Tool returned an unstructured result."}
    if result.get("error"):
        return {"passed": False, "decision": "revise", "reason": f"Tool failed: {result['error']}"}
    if not verification.get("passed"):
        return {"passed": False, "decision": "revise", "reason": verification.get("reason", "Verification failed.")}
    return {"passed": True, "decision": "continue", "reason": "Independent verification accepted the tool result."}


def audit(event: str, delegation: Delegation, **details: Any) -> None:
    """Append audit data without ever breaking a user request on I/O failure."""
    record = {"timestamp": time.time(), "event": event, "delegation": asdict(delegation), **details}
    try:
        os.makedirs(os.path.dirname(_AUDIT_PATH), exist_ok=True)
        with _AUDIT_LOCK, open(_AUDIT_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as error:
        logger.warning("Unable to write agent audit event: %s", error)
