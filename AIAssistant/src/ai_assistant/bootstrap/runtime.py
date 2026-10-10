"""Tool gateway adapter built by the composition root (ADR 0001/0003).

This module used to hold a mutable process-global ``_tool_service`` locator that
agent code reached through module-level functions. That hid the dependency and
inverted the layer arrow (``agents`` -> ``bootstrap``). The gateway is now an
explicit object: :class:`PlatformToolGateway` is constructed from the DI
container and injected into the agent engine, MCP adapter and A2A orchestrator.
No module state remains.
"""
from __future__ import annotations

from ..application.tools import ToolExecutionService
from ..domain.security import Principal, current_principal

# Trusted in-process desktop callers do not bind a principal; they keep the
# historical broad scope set. Remote transports bind a least-privilege principal
# through ``bind_principal`` before the engine runs.
_LOCAL_PRINCIPAL = Principal(
    "legacy-local",
    frozenset({"project.read", "project.write", "project.execute", "desktop.action"}),
)

_UNCONFIGURED_RESULT = {
    "success": False,
    "error_code": "runtime_unconfigured",
    "error": "AI Agent Platform is not bootstrapped",
}


class PlatformToolGateway:
    """Implements :class:`ai_assistant.ports.ToolGateway` over the policy gateway.

    ``service`` is injected by :func:`ai_assistant.bootstrap.build_container`.
    A ``None`` service keeps the pre-bootstrap behaviour and reports a
    structured ``runtime_unconfigured`` result instead of raising.
    """

    def __init__(self, service: ToolExecutionService | None = None) -> None:
        self._service = service

    def _principal(self) -> Principal:
        return current_principal() or _LOCAL_PRINCIPAL

    def execute(self, tool_name: str, parameters: dict) -> dict:
        if self._service is None:
            return dict(_UNCONFIGURED_RESULT)
        return self._service.execute(tool_name, parameters, self._principal()).to_dict()

    def execute_approved(self, tool_name: str, parameters: dict, approval_token: str = "") -> dict:
        """Execute an approval-gated tool by spending its single-use grant.

        Without a grant covering exactly this invocation the service answers
        ``approval_required``; the token is never interpreted as a blanket
        permission.
        """
        if self._service is None:
            return dict(_UNCONFIGURED_RESULT)
        return self._service.execute(
            tool_name, parameters, self._principal(), approval_token=approval_token,
        ).to_dict()

    def issue_approval_grant(self, tool_name: str, parameters: dict, approval_token: str) -> None:
        """Bind a single-use grant to an invocation a user just approved."""
        if self._service is None:
            return
        self._service.issue_approval_grant(tool_name, parameters, approval_token)

    def approval_covers(self, tool_name: str, parameters: dict, approval_token: str) -> bool:
        """Non-consuming probe: is this invocation still covered by a grant?"""
        if self._service is None:
            return False
        return self._service.approval_covers(tool_name, parameters, approval_token)


__all__ = ["PlatformToolGateway"]
