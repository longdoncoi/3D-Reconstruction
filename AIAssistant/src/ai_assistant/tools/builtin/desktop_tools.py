"""Desktop action tools."""
from __future__ import annotations

from typing import Any

from ai_assistant.tools.action_manifest import validate_action_params


def tool_application_action(params: dict[str, Any]) -> dict[str, Any]:
    """Create a UI-action request; success is only reported after Qt ACKs it."""
    canonical_params, error = validate_action_params(params)
    if error is not None or canonical_params is None:
        return {"error": error or "Invalid UI action parameters"}
    return {
        "pending_ui_ack": True,
        "action": canonical_params["action"],
        "request_id": canonical_params.get("request_id", ""),
        "message": "Action is awaiting acknowledgement from the Qt desktop client.",
    }
