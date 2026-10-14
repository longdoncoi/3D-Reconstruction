"""Policy-first supervisor and specialist coordination for the assistant."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from ai_assistant.config.logging import get_agent_logger
from ai_assistant.domain.specialists import TRANSFER_TARGETS, Specialist
from ai_assistant.domain.tools import tool_result_has_evidence

logger = logging.getLogger("ai_assistant.orchestration.supervisor")

# A2A remote lookup — dependency inversion, no adapter import.
#
# The supervisor must never import ``ai_assistant.adapters`` (that created an
# import cycle with ``a2a_protocol``, which needs ``Specialist`` to advertise
# skills).  Instead the composition root installs the lookup resolved by the
# A2A adapter via :func:`set_remote_specialist_lookup`; with nothing installed
# delegation stays strictly local.
RemoteSpecialistLookup = Callable[[str], str | None]

_remote_specialist_lookup: RemoteSpecialistLookup | None = None
_remote_lookup_lock = threading.Lock()

supervisor_logger = get_agent_logger("supervisor")
verification_logger = get_agent_logger("verification")


def set_remote_specialist_lookup(lookup: RemoteSpecialistLookup | None) -> None:
    """Install (or clear) the A2A remote-specialist resolver.

    The composition root wires
    ``adapters.a2a_protocol.resolve_remote_specialist`` here at startup so the
    orchestration layer stays free of adapter imports (ADR 0002).
    """
    global _remote_specialist_lookup
    with _remote_lookup_lock:
        _remote_specialist_lookup = lookup


def _resolve_remote_endpoint(specialist: Specialist) -> str | None:
    """Ask the installed lookup for a remote endpoint, failing closed."""
    with _remote_lookup_lock:
        lookup = _remote_specialist_lookup
    if lookup is None:
        return None
    try:
        return lookup(specialist.value)
    except Exception as error:
        logger.warning("Remote specialist lookup failed | specialist=%s error=%s", specialist.value, error)
        return None


@dataclass(frozen=True)
class Delegation:
    specialist: Specialist
    tool: str | None
    reason: str
    idempotency_key: str
    remote_endpoint: str | None = field(default=None, compare=False)


RESEARCH_TOOLS = {"read_file", "list_directory", "find_files", "search_text", "analyze_code", "git_diff", "rag_search"}
VERIFICATION_TOOLS = {"validate_file", "get_project_status"}
CODE_TOOLS = {"write_file", "patch_file", "replace_file_content", "multi_replace_file_content", "create_directory", "run_command"}
WORKFLOW_TOOLS = {"application_action"}
# ``TRANSFER_TARGETS`` is the domain-owned handoff table; it is re-exported
# here because the supervisor is its primary consumer (and existing callers
# import it from this module).

#: Specialists whose tool rows are spelled out explicitly below. A transfer
#: target pointing at one of them rides along in that explicit row (CODE runs
#: its own marker among its tools); any other transfer specialist gets a
#: derived marker-only row, so a new handoff target cannot be forgotten here.
_EXPLICIT_TOOL_ROWS = frozenset({
    Specialist.SUPERVISOR, Specialist.RESEARCH, Specialist.VERIFICATION,
    Specialist.CODE, Specialist.WORKFLOW,
})

#: The tools each specialist may execute. ``SUPERVISOR`` executes nothing (it
#: answers directly); ``CODE`` may also run read/verify tools because a coding
#: task inspects the repository before it mutates it. Each transfer specialist
#: not spelled out below is derived from ``TRANSFER_TARGETS`` and may be
#: handed exactly its own handoff marker. ``authorise`` consults this table
#: fail-closed, so the tool sets are enforced least privilege rather than
#: decorative routing labels (review finding 13).
ALLOWED_TOOLS: dict[Specialist, frozenset[str]] = {
    Specialist.SUPERVISOR: frozenset(),
    Specialist.RESEARCH: frozenset(RESEARCH_TOOLS),
    Specialist.VERIFICATION: frozenset(VERIFICATION_TOOLS),
    Specialist.CODE: frozenset(RESEARCH_TOOLS | VERIFICATION_TOOLS | CODE_TOOLS | set(TRANSFER_TARGETS)),
    Specialist.WORKFLOW: frozenset(WORKFLOW_TOOLS),
    **{
        specialist: frozenset(
            name for name, target in TRANSFER_TARGETS.items() if target is specialist
        )
        for specialist in dict.fromkeys(TRANSFER_TARGETS.values())
        if specialist not in _EXPLICIT_TOOL_ROWS
    },
}


@dataclass(frozen=True)
class ToolRoute:
    """One routing rule: the specialist that owns a class of tools, and why."""

    specialist: Specialist
    tools: frozenset[str]
    reason: str
    basis: str
    requires_prefer_code: bool = False


#: The routing registry ``delegate`` consults after an explicit handoff —
#: first matching rule wins, so the tuple order IS the precedence: coding
#: preference (guard-flagged) > read-only > verification > mutating code >
#: desktop workflow, then the ``SUPERVISOR`` fallback in ``_route``. Adding a
#: tool class or a specialist is one appended rule, not a re-ordered ``if``
#: chain (architecture finding 12).
_TOOL_ROUTE_RULES: tuple[ToolRoute, ...] = (
    ToolRoute(
        Specialist.CODE,
        frozenset(RESEARCH_TOOLS | VERIFICATION_TOOLS | CODE_TOOLS),
        "coding task requires isolated repository context",
        "coding_preference",
        requires_prefer_code=True,
    ),
    ToolRoute(Specialist.RESEARCH, frozenset(RESEARCH_TOOLS), "tool is read/RAG-only", "research_tool"),
    ToolRoute(
        Specialist.VERIFICATION, frozenset(VERIFICATION_TOOLS),
        "tool validates an observable result", "verification_tool",
    ),
    ToolRoute(
        Specialist.CODE, frozenset(CODE_TOOLS),
        "tool can change project state and requires approval", "code_tool",
    ),
    ToolRoute(
        Specialist.WORKFLOW, frozenset(WORKFLOW_TOOLS),
        "tool dispatches a Qt desktop workflow", "workflow_tool",
    ),
)
_AUDIT_LOCK = threading.Lock()
_LOG_CACHE_LOCK = threading.Lock()
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


def _route(tool: str | None, prefer_code: bool) -> tuple[Specialist, str, str]:
    """Resolve the owning specialist from the routing registry (finding 12).

    An explicit handoff and the ordered ``_TOOL_ROUTE_RULES`` table are the
    whole decision; ``delegate`` itself never branches on a tool name again.
    """
    if tool in TRANSFER_TARGETS:
        specialist = TRANSFER_TARGETS[tool]
        return specialist, f"explicit supervisor handoff to {specialist.value}", "explicit_transfer"
    for route in _TOOL_ROUTE_RULES:
        if (not route.requires_prefer_code or prefer_code) and tool in route.tools:
            return route.specialist, route.reason, route.basis
    return Specialist.SUPERVISOR, "direct response or unresolved intent", "supervisor_fallback"


def delegate(
    task: str,
    session_id: str,
    tool: str | None = None,
    parameters: dict[str, Any] | None = None,
    prefer_code: bool = False,
) -> Delegation:
    """Select exactly one worker role. Workers never delegate further."""
    specialist, reason, basis = _route(tool, prefer_code)

    # A2A: check whether a remote agent can handle this specialist role.
    remote_endpoint = _resolve_remote_endpoint(specialist)
    if remote_endpoint:
        reason += f" (A2A remote: {remote_endpoint})"
        supervisor_logger.info(
            "A2A remote match | specialist=%s url=%s", specialist.value, remote_endpoint,
        )

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
    # The throttle cache keeps its own lock: ``_AUDIT_LOCK`` guards a file
    # append further down, and serialising log throttling behind it couples
    # two unrelated concerns (finding 26).
    with _LOG_CACHE_LOCK:
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
    """Enforce least privilege before calling a tool.

    The specialist's allowed-tool set (``ALLOWED_TOOLS``) is the policy, not an
    accident of how ``delegate`` classified the call: an unknown
    ``(specialist, tool)`` pair is refused fail-closed (review finding 13).
    """
    if delegation.tool is None:
        return True, None
    if delegation.specialist is Specialist.SUPERVISOR:
        return False, f"No specialist policy exists for tool: {delegation.tool}"
    if delegation.tool not in ALLOWED_TOOLS.get(delegation.specialist, frozenset()):
        return False, f"Tool {delegation.tool} is not permitted for specialist {delegation.specialist.value}."
    if delegation.specialist is Specialist.CODE and delegation.tool in CODE_TOOLS and not needs_approval:
        return False, "Code tools must be routed through the approval workflow."
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

    # Shared evidence table lives in ``domain.tools`` so this check and the
    # coding-workflow classifier read one source of truth (review #24/7).
    if not tool_result_has_evidence(str(delegation.tool), result):
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
    "ALLOWED_TOOLS",
    "CODE_TOOLS",
    "RESEARCH_TOOLS",
    "TRANSFER_TARGETS",
    "VERIFICATION_TOOLS",
    "WORKFLOW_TOOLS",
    "Delegation",
    "Specialist",
    "ToolRoute",
    "audit",
    "authorise",
    "delegate",
    "reflect_result",
    "set_remote_specialist_lookup",
    "verify_result",
]
