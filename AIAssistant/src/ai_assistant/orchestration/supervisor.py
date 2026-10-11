"""Policy-first supervisor and specialist coordination for the assistant."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from ai_assistant.config.logging import get_agent_logger
from ai_assistant.orchestration.specialists.code import is_coding_task

logger = logging.getLogger("ai_assistant.orchestration.supervisor")

# A2A protocol integration — graceful no-op when a2a_protocol is unavailable.
try:
    from ai_assistant.adapters.a2a_protocol import A2A_ENABLED, a2a_available, get_remote_registry
    _A2A_IMPORT_OK = True
except ImportError:
    _A2A_IMPORT_OK = False
    A2A_ENABLED = False  # type: ignore[assignment]

    def a2a_available() -> bool:  # type: ignore[misc]
        return False

    def get_remote_registry() -> dict:  # type: ignore[misc]
        return {}

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
    remote_endpoint: str | None = field(default=None, compare=False)


_SPECIALIST_INSTRUCTIONS = {
    Specialist.CHATBOT: "Answer conversationally using the Chatbot Agent and its RAG context only.",
    Specialist.TOOLAPP: "Dispatch only canonical Qt application_action calls and wait for acknowledgement.",
    Specialist.RESEARCH: "Inspect only the smallest relevant evidence; cite files and do not change state.",
    Specialist.WORKFLOW: "Dispatch only a canonical desktop action and wait for the desktop acknowledgement.",
    Specialist.CODE: "Describe the smallest safe change and require explicit approval before changing project state.",
    Specialist.VERIFICATION: "Independently check observable output; report a failed check instead of assuming success.",
    Specialist.SUPERVISOR: "Clarify intent, enforce policy, and coordinate the next specialist handoff.",
}

_RESEARCH_TOOLS = {"read_file", "list_directory", "find_files", "search_text", "analyze_code", "git_diff", "rag_search"}
_VERIFICATION_TOOLS = {"validate_file", "get_project_status"}
_CODE_TOOLS = {"write_file", "patch_file", "replace_file_content", "multi_replace_file_content", "create_directory", "run_command"}
CODE_AGENT_TOOLS = frozenset({
    "find_files", "list_directory", "search_text", "read_file", "analyze_code", "git_diff",
    "get_project_status", "validate_file", "write_file", "patch_file",
    "replace_file_content", "multi_replace_file_content", "create_directory", "run_command",
})
_WORKFLOW_TOOLS = {"application_action"}
_TRANSFER_TARGETS = {
    "transfer_to_code_agent": Specialist.CODE,
    "transfer_to_toolapp_agent": Specialist.TOOLAPP,
    "transfer_to_chatbot_agent": Specialist.CHATBOT,
}
_AUDIT_LOCK = threading.Lock()
_RECENT_DELEGATION_LOGS: dict[tuple[str, str], float] = {}


def _get_audit_path() -> str:
    # ADR 0001: APP_DATA_DIR is parsed only in ``settings``; resolve through it.
    from ai_assistant.settings import data_dir
    app_data = data_dir()
    if app_data is not None:
        return str(app_data / "AIAssistant" / "agent_audit.jsonl")
    try:
        from ai_assistant.config.paths import get_paths
        return str(get_paths().cache_dir / "agent_audit.jsonl")
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Cache", "agent_audit.jsonl"))


def route_task(task: str) -> Specialist:
    """Classify the primary intent into a Specialist role."""
    return Specialist.CODE if is_coding_task(task) else Specialist.SUPERVISOR


def delegate(
    task: str,
    session_id: str,
    tool: str | None = None,
    parameters: dict[str, Any] | None = None,
    prefer_code: bool = False,
) -> Delegation:
    """Select exactly one worker role. Workers never delegate further."""
    if tool in _TRANSFER_TARGETS:
        specialist = _TRANSFER_TARGETS[tool]
        reason, basis = f"explicit supervisor handoff to {specialist.value}", "explicit_transfer"
    elif prefer_code and tool in (_RESEARCH_TOOLS | _VERIFICATION_TOOLS | _CODE_TOOLS):
        specialist, reason, basis = Specialist.CODE, "coding task requires isolated repository context", "coding_preference"
    elif tool in _RESEARCH_TOOLS:
        specialist, reason, basis = Specialist.RESEARCH, "tool is read/RAG-only", "research_tool"
    elif tool in _VERIFICATION_TOOLS:
        specialist, reason, basis = Specialist.VERIFICATION, "tool validates an observable result", "verification_tool"
    elif tool in _CODE_TOOLS:
        specialist, reason, basis = Specialist.CODE, "tool can change project state and requires approval", "code_tool"
    elif tool in _WORKFLOW_TOOLS:
        specialist, reason, basis = Specialist.WORKFLOW, "tool dispatches a Qt desktop workflow", "workflow_tool"
    else:
        specialist, reason, basis = Specialist.SUPERVISOR, "direct response or unresolved intent", "supervisor_fallback"

    # A2A: check whether a remote agent can handle this specialist role.
    remote_endpoint: str | None = None
    if a2a_available():
        registry = get_remote_registry()
        for url, agent in registry.items():
            if specialist.value in agent.skills:
                remote_endpoint = url
                reason += f" (A2A remote: {url})"
                supervisor_logger.info(
                    "A2A remote match | specialist=%s url=%s", specialist.value, url,
                )
                break

    source = json.dumps(
        {"session": session_id, "task": task, "tool": tool, "params": parameters or {}},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    delegation = Delegation(
        specialist,
        tool,
        reason,
        hashlib.sha256(source.encode("utf-8")).hexdigest()[:24],
        remote_endpoint=remote_endpoint,
    )

    _log_key = (session_id, delegation.idempotency_key)
    _now = time.monotonic()
    with _AUDIT_LOCK:
        _last = _RECENT_DELEGATION_LOGS.get(_log_key)
        if _last is None or (_now - _last) > 120:
            _RECENT_DELEGATION_LOGS[_log_key] = _now
            if len(_RECENT_DELEGATION_LOGS) > 512:
                _RECENT_DELEGATION_LOGS.clear()
            _should_log = True
        else:
            _should_log = False

    if _should_log:
        supervisor_logger.info(
            "SUPERVISOR_ROUTE | session=%s basis=%s tool=%s prefer_code=%s -> specialist=%s remote=%s "
            "idempotency_key=%s task_chars=%d parameter_keys=%s",
            session_id,
            basis,
            tool or "none",
            prefer_code,
            specialist.value,
            remote_endpoint or "local",
            delegation.idempotency_key,
            len(task),
            sorted((parameters or {}).keys()),
        )
        get_agent_logger(specialist.value).info(
            "Delegated | tool=%s reason=%s idempotency_key=%s remote=%s",
            tool or "none",
            reason,
            delegation.idempotency_key,
            remote_endpoint or "local",
        )
    return delegation


def authorise(delegation: Delegation, needs_approval: bool) -> tuple[bool, str | None]:
    """Enforce least privilege before calling a tool."""
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
        outcome = {
            "passed": bool(result.get("pending_ui_ack") or result.get("success")),
            "reason": "Qt acknowledgement is required for desktop actions.",
        }
    else:
        outcome = {"passed": True, "reason": "Structured tool result passed policy checks."}
    verification_logger.info(
        "Verification result | tool=%s passed=%s reason=%s",
        delegation.tool,
        outcome["passed"],
        outcome["reason"],
    )
    return outcome


def specialist_instruction(delegation: Delegation) -> str:
    """Return a compact role handoff for the shared local model."""
    return _SPECIALIST_INSTRUCTIONS[delegation.specialist]


def reflect_result(delegation: Delegation, result: Any, verification: dict[str, Any]) -> dict[str, Any]:
    """Independent, deterministic critic used after each tool result."""
    if not isinstance(result, dict):
        outcome = {"passed": False, "decision": "revise", "reason": "Tool returned an unstructured result."}
    elif result.get("error"):
        outcome = {"passed": False, "decision": "revise", "reason": f"Tool failed: {result['error']}"}
    elif not verification.get("passed"):
        outcome = {"passed": False, "decision": "revise", "reason": verification.get("reason", "Verification failed.")}
    else:
        outcome = {"passed": True, "decision": "continue", "reason": "Independent verification accepted the tool result."}
    verification_logger.info(
        "Reflection result | tool=%s passed=%s decision=%s reason=%s",
        delegation.tool,
        outcome["passed"],
        outcome["decision"],
        outcome["reason"],
    )
    return outcome


def audit(event: str, delegation: Delegation, **details: Any) -> None:
    """Append audit data without ever breaking a user request on I/O failure."""
    record = {"timestamp": time.time(), "event": event, "delegation": asdict(delegation), **details}
    audit_path = _get_audit_path()
    try:
        os.makedirs(os.path.dirname(audit_path), exist_ok=True)
        with _AUDIT_LOCK, open(audit_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as error:
        logger.warning("Unable to write agent audit event: %s", error)


__all__ = [
    "CODE_AGENT_TOOLS",
    "_CODE_TOOLS",
    "_RESEARCH_TOOLS",
    "_TRANSFER_TARGETS",
    "_VERIFICATION_TOOLS",
    "_WORKFLOW_TOOLS",
    "Delegation",
    "Specialist",
    "audit",
    "authorise",
    "delegate",
    "reflect_result",
    "route_task",
    "specialist_instruction",
    "verify_result",
]
