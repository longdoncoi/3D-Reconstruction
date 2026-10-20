from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from ..domain.approvals import approval_fingerprint
from ..domain.security import Principal, classification_at_least
from ..domain.tools import ToolRequest, ToolResult
from ..ports import ToolCatalog, ToolCatalogEntry, ToolEventSink
from .approvals import InMemoryApprovalGrantStore
from .validation import validate_input


class ToolExecutionService:
    """The only application path allowed to execute registered tools.

    Approval-gated tools are unlocked exclusively by a single-use grant
    :meth:`issue_approval_grant` created for the exact ``(tool, params)``
    pair a user approved — never by a boolean the caller can assert.

    The catalogue is the :class:`~ai_assistant.ports.ToolCatalog` *port*
    (ADR 0032), not the concrete plugin registry: the policy chokepoint
    depends on the contract, so the catalogue implementation stays
    replaceable.
    """

    def __init__(self, registry: ToolCatalog, event_sink: ToolEventSink | None = None,
                 grants: InMemoryApprovalGrantStore | None = None) -> None:
        self._registry = registry
        self._event_sink = event_sink
        self._grants = grants if grants is not None else InMemoryApprovalGrantStore()

    def execute(self, tool_name: str, parameters: Mapping[str, Any], principal: Principal,
                correlation_id: str | None = None, *, approval_token: str = "") -> ToolResult:
        request = ToolRequest(tool_name, dict(parameters), correlation_id or str(uuid4()))
        admitted = self._admit(request, principal, approval_token)
        if isinstance(admitted, ToolResult):
            return self._finish(request, admitted)
        try:
            payload = admitted.executor(dict(request.parameters))
        except Exception as error:  # Boundary converts plugin failure to structured result.
            return self._finish(request, ToolResult(False, error_code="tool_failure", message=str(error)))
        if not isinstance(payload, dict):
            return self._finish(request, ToolResult(False, error_code="invalid_tool_result", message="Tool returned a non-object"))
        success = not bool(payload.get("error")) and payload.get("success") is not False
        return self._finish(request, ToolResult(success, payload, None if success else "tool_failure"))

    def preflight(self, tool_name: str, parameters: Mapping[str, Any], principal: Principal,
                  correlation_id: str | None = None, *, approval_token: str = "") -> ToolResult:
        """Clear the policy gate for a dispatch that executes somewhere else.

        The remote A2A leg leaves this process, so it must pass the same
        validation, scope, classification and single-use grant checks the local
        path passes — and spend that grant — without the executor running here.
        A refusal is the caller's instruction not to dispatch.
        """
        request = ToolRequest(tool_name, dict(parameters), correlation_id or str(uuid4()))
        admitted = self._admit(request, principal, approval_token)
        if isinstance(admitted, ToolResult):
            return self._finish(request, admitted)
        return self._finish(request, ToolResult(True, {"tool": tool_name, "preflight": True}))

    def _admit(self, request: ToolRequest, principal: Principal,
               approval_token: str) -> ToolCatalogEntry | ToolResult:
        """Validate and authorise one request: the refusal, or the tool to run."""
        item = self._registry.tool(request.tool_name)
        if item is None:
            return ToolResult(False, error_code="tool_not_found", message=f"Unknown tool: {request.tool_name}")
        spec = item.spec
        validation_error = validate_input(spec.input_schema, request.parameters)
        if validation_error:
            return ToolResult(False, error_code="invalid_input", message=validation_error)
        if not principal.permits(spec.required_scope):
            return ToolResult(False, error_code="forbidden", message="Required tool scope is missing")
        if not classification_at_least(spec.maximum_classification, principal.classification):
            return ToolResult(False, error_code="data_policy_denied", message="Data classification is not permitted")
        if spec.requires_approval and not self._grants.covers(
                approval_token, approval_fingerprint(spec.name, request.parameters)):
            return ToolResult(False, {"approval_required": True, "tool": spec.name},
                              "approval_required", "User approval is required")
        return item

    def issue_approval_grant(self, tool_name: str, parameters: Mapping[str, Any], token: str) -> None:
        """Bind a single-use grant to the invocation the user just approved.

        Only a path that represents a user decision may call this.
        """
        self._grants.issue(token, approval_fingerprint(tool_name, parameters))

    def approval_covers(self, tool_name: str, parameters: Mapping[str, Any], token: str) -> bool:
        """Whether ``token`` still unlocks this invocation (does not spend it)."""
        return self._grants.peek(token, approval_fingerprint(tool_name, parameters))

    def _finish(self, request: ToolRequest, result: ToolResult) -> ToolResult:
        if self._event_sink:
            self._event_sink(request, result)
        return result
