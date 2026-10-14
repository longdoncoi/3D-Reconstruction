"""Audit stream for the single tool-execution chokepoint (ADR 0003).

``ToolExecutionService`` is the only path allowed to run a registered tool, so
its ``event_sink`` sees *every* execution — the agent loop, a HITL-approved
invocation, a desktop action and the A2A task worker alike. Until now the
composition root wired no sink at all, so anything executed outside the agent
loop left no trace: the sink existed, was tested, and was never installed.

The record is one JSON line on the ``agent.tools`` logger:

* ``tool`` / ``correlation_id`` — which call it was, tie-able to a task row;
* ``ok`` / ``error_code`` — the outcome, including the policy rejections
  (``forbidden``, ``approval_required``, ``invalid_input``, ``tool_failure``);
* ``parameters`` — through :func:`ai_assistant.domain.governance.redact_value`,
  the single redaction owner, so a password or token never reaches the log.

Prometheus counters stay where they are (``observability.metrics.record_tool``,
called by the agent runner): counting at the chokepoint too would double-count
agent runs, and this record is about "what happened", not "how often".
"""
from __future__ import annotations

import json
from typing import Any

from ..config.logging import get_agent_logger
from ..domain.governance import redact_value
from ..domain.tools import ToolRequest, ToolResult

logger = get_agent_logger("tools")

__all__ = ["audit_tool_event", "tool_event_payload"]


def tool_event_payload(request: ToolRequest, result: ToolResult) -> dict[str, Any]:
    """The JSON-safe, redacted record for one execution.

    Kept as a pure function so the redaction policy is testable without a
    logger, a container or a disk.
    """
    return {
        "tool": request.tool_name,
        "correlation_id": request.correlation_id,
        "ok": result.success,
        "error_code": result.error_code,
        "parameters": redact_value(dict(request.parameters)),
    }


def audit_tool_event(request: ToolRequest, result: ToolResult) -> None:
    """Default ``ToolEventSink`` installed by the composition root."""
    logger.info(
        "tool_event %s",
        json.dumps(tool_event_payload(request, result), ensure_ascii=False, default=str),
    )
