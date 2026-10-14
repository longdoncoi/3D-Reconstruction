"""Compatibility and contract mapping for tool definitions."""
from __future__ import annotations

from typing import Any

from ai_assistant.tools.schema import build_tool_models as _build_tool_models
from ai_assistant.tools.validation import validate_tool_call

from .definition import ToolDefinition

#: Single owner of the *policy contract* for built-in tools. The factory, the
#: registry enrichment and the JSON schema builders all resolve per-tool policy
#: through :func:`tool_contract_for` — nobody may redefine these values.
DEFAULT_CONTRACT: dict[str, Any] = {
    "timeout_seconds": 10,
    "policy": "read_only",
    "requires_approval": False,
}

TOOL_CONTRACT_OVERRIDES: dict[str, dict[str, Any]] = {
    "application_action": {"timeout_seconds": 30, "policy": "desktop_ack"},
    "write_file": {"timeout_seconds": 10, "policy": "code_write", "requires_approval": True},
    "patch_file": {"timeout_seconds": 10, "policy": "code_write", "requires_approval": True},
    "replace_file_content": {"timeout_seconds": 10, "policy": "code_write", "requires_approval": True},
    "multi_replace_file_content": {"timeout_seconds": 10, "policy": "code_write", "requires_approval": True},
    "create_directory": {"timeout_seconds": 10, "policy": "code_write", "requires_approval": True},
    "run_command": {"timeout_seconds": 120, "policy": "code_execute", "requires_approval": True},
    # LSP queries spawn a language server on first use and index the project;
    # 30s covers a cold clangd/pylsp start without hanging the ReAct loop.
    "go_to_definition": {"timeout_seconds": 30, "policy": "read_only"},
    "find_references": {"timeout_seconds": 30, "policy": "read_only"},
}


def tool_contract_for(name: str) -> dict[str, Any]:
    """Resolve the effective policy contract for a tool name."""
    return {**DEFAULT_CONTRACT, **TOOL_CONTRACT_OVERRIDES.get(name, {})}


#: Structured refusal returned by every execution surface when the platform
#: gateway is not bootstrapped yet (ADR 0003/0013). Declared once and imported
#: by the gateway adapter, the MCP adapter, the runner and the approval
#: service, so the ``runtime_unconfigured`` wire shape cannot drift.
RUNTIME_UNCONFIGURED_RESULT: dict[str, Any] = {
    "success": False,
    "error_code": "runtime_unconfigured",
    "error": "AI Agent Platform is not bootstrapped",
}


def _to_tool_specs(tool_definitions: list[dict[str, Any]]) -> list[ToolDefinition]:
    specs = []
    for source in tool_definitions:
        name = source["name"]
        contract = tool_contract_for(name)
        specs.append(
            ToolDefinition(
                name=name,
                description=source.get("description", ""),
                parameters=source.get("parameters", {}),
                timeout_seconds=contract["timeout_seconds"],
                policy=contract["policy"],
                requires_approval=contract["requires_approval"],
            )
        )
    return specs


def build_tool_models(tool_definitions: list[dict[str, Any]]) -> dict[str, type]:
    specs = _to_tool_specs(tool_definitions)
    return _build_tool_models(specs)


__all__ = [
    "DEFAULT_CONTRACT",
    "TOOL_CONTRACT_OVERRIDES",
    "build_tool_models",
    "tool_contract_for",
    "validate_tool_call",
]
