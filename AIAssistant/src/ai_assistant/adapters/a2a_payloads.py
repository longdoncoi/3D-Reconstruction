"""A2A protocol payload builders and helpers (ADR 0005).

Extracted from ``adapters/a2a.py`` to separate payload construction from routing logic.
Includes: task payload serialization, sensitive data redaction, and state mapping.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..domain.governance import redact_value
from ..domain.tasks import AgentTask

# A2A 0.3 state mapping
_A2A_STATE = {
    "submitted": "submitted", "working": "working", "input_required": "input-required",
    "completed": "completed", "failed": "failed", "canceled": "canceled", "rejected": "rejected",
}

# Approval resume extension URI
APPROVAL_RESUME_EXTENSION = "urn:3d-reconstruction:a2a:approval-resume:v1"


def _text_message(task: AgentTask, text: str) -> dict[str, Any]:
    return {
        "role": "ROLE_AGENT", "messageId": f"status-{task.id}-{int(task.updated_at * 1000)}",
        "contextId": task.context_id, "taskId": task.id, "parts": [{"text": text}],
    }


def _safe_value(value: Any, key: str = "") -> Any:
    """Redaction for A2A payloads — delegates to the shared domain policy."""
    return redact_value(value, key)


def task_payload(task: AgentTask) -> dict[str, Any]:
    """Serialize an AgentTask into A2A 0.3 task payload format."""
    status: dict[str, Any] = {
        "state": _A2A_STATE[task.status],
        "timestamp": datetime.fromtimestamp(task.updated_at, UTC).isoformat().replace("+00:00", "Z"),
    }
    if task.status.value == "input_required" and task.result:
        status["message"] = _text_message(task, str(task.result.get("content", "Additional input is required.")))
    elif task.error:
        status["message"] = _text_message(task, task.error)
    return {
        "id": task.id,
        "contextId": task.context_id,
        "status": status,
        "metadata": _safe_value({key: value for key, value in task.metadata.items() if not key.startswith("agent_run.")}),
        "artifacts": [{"artifactId": f"result-{task.id}", "name": "result", "parts": [{"data": _safe_value(task.result)}]}]
        if task.result and task.status.value == "completed" else [],
    }


def build_agent_card(
    *,
    agent_name: str,
    agent_version: str,
    base_url: str,
    capabilities: list[str],
) -> dict[str, Any]:
    """Build A2A 0.3 Agent Card for discovery endpoint."""
    return {
        "name": agent_name,
        "description": "3D-Reconstruction AI Agent Platform",
        "version": agent_version,
        "protocolVersion": "0.3",
        "url": base_url + "/a2a",
        "supportedInterfaces": [{
            "url": base_url, "protocolBinding": "HTTP+JSON/REST", "protocolVersion": "0.3",
        }],
        "capabilities": {"streaming": True, "pushNotifications": False},
        "defaultInputModes": ["text/plain"], "defaultOutputModes": ["text/plain"],
        "skills": [
            {"id": capability, "name": capability.replace("_", " ").title(),
             "description": f"Execute {capability} tasks through the 3D-Reconstruction agent platform."}
            for capability in sorted(capabilities)
        ],
        "extensions": [APPROVAL_RESUME_EXTENSION],
    }


__all__ = [
    "APPROVAL_RESUME_EXTENSION",
    "build_agent_card",
    "task_payload",
]
