"""Compatibility bridge used only by legacy modules during migration."""
from __future__ import annotations

from ..application.tools import ToolExecutionService
from ..domain.security import Principal, current_principal

_tool_service: ToolExecutionService | None = None
_principal = Principal(
    "legacy-local",
    frozenset({"project.read", "project.write", "project.execute", "desktop.action"}),
)


def configure_tool_service(service: ToolExecutionService) -> None:
    global _tool_service
    _tool_service = service


def execute_tool(tool_name: str, parameters: dict) -> dict:
    if _tool_service is None:
        return {"success": False, "error_code": "runtime_unconfigured", "error": "AI Agent Platform is not bootstrapped"}
    return _tool_service.execute(tool_name, parameters, current_principal() or _principal).to_dict()


def execute_approved_tool(tool_name: str, parameters: dict) -> dict:
    """Execute a tool only after the legacy UI approval has been verified."""
    if _tool_service is None:
        return {"success": False, "error_code": "runtime_unconfigured", "error": "AI Agent Platform is not bootstrapped"}
    return _tool_service.execute(tool_name, parameters, current_principal() or _principal, approval_granted=True).to_dict()
