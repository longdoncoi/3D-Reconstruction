"""Compatibility and contract mapping for tool definitions."""
from __future__ import annotations

from typing import Any

from ai_assistant.tools.action_manifest import action_ids
from ai_assistant.tools.schema import build_tool_models as _build_tool_models
from ai_assistant.tools.schema import grammar_schema as _grammar_schema
from ai_assistant.tools.validation import validate_tool_call

from .definition import ToolDefinition

_DEFAULT_CONTRACT = {
    "timeout_seconds": 10,
    "policy": "read_only",
    "requires_approval": False,
    "idempotent": True,
}

_TOOL_CONTRACT_OVERRIDES = {
    "application_action": {"timeout_seconds": 30, "policy": "desktop_ack"},
    "write_file": {"timeout_seconds": 10, "policy": "code_write", "requires_approval": True},
    "patch_file": {"timeout_seconds": 10, "policy": "code_write", "requires_approval": True},
    "replace_file_content": {"timeout_seconds": 10, "policy": "code_write", "requires_approval": True},
    "multi_replace_file_content": {"timeout_seconds": 10, "policy": "code_write", "requires_approval": True},
    "create_directory": {"timeout_seconds": 10, "policy": "code_write", "requires_approval": True},
    "run_command": {"timeout_seconds": 120, "policy": "code_execute", "requires_approval": True},
}


def _to_tool_specs(tool_definitions: list[dict[str, Any]]) -> list[ToolDefinition]:
    specs = []
    for source in tool_definitions:
        name = source["name"]
        overrides = _TOOL_CONTRACT_OVERRIDES.get(name, {})
        specs.append(
            ToolDefinition(
                name=name,
                description=source.get("description", ""),
                parameters=source.get("parameters", {}),
                timeout_seconds=overrides.get("timeout_seconds", _DEFAULT_CONTRACT["timeout_seconds"]),
                policy=overrides.get("policy", _DEFAULT_CONTRACT["policy"]),
                requires_approval=overrides.get("requires_approval", _DEFAULT_CONTRACT["requires_approval"]),
                idempotent=overrides.get("idempotent", _DEFAULT_CONTRACT["idempotent"]),
            )
        )
    return specs


def json_schema(tool: dict[str, Any]) -> dict[str, Any]:
    specs = _to_tool_specs([tool])
    return specs[0].json_schema


def enrich_tool_definitions(tool_definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = _to_tool_specs(tool_definitions)
    enriched = []
    for spec in specs:
        tool_dict = {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
            "timeout_seconds": spec.timeout_seconds,
            "policy": spec.policy,
            "requires_approval": spec.requires_approval,
            "idempotent": spec.idempotent,
            "schema": spec.json_schema,
        }
        if tool_dict["name"] == "application_action" and "action" in tool_dict["parameters"]:
            tool_dict["parameters"]["action"] = {**tool_dict["parameters"]["action"], "enum": sorted(action_ids())}
        enriched.append(tool_dict)
    return enriched


def build_tool_models(tool_definitions: list[dict[str, Any]]) -> dict[str, type]:
    specs = _to_tool_specs(tool_definitions)
    return _build_tool_models(specs)


def openai_tools(tool_definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = _to_tool_specs(tool_definitions)
    return [s.to_openai_format() for s in specs]


def grammar_schema(tool_definitions: list[dict[str, Any]]) -> str:
    specs = _to_tool_specs(tool_definitions)
    return _grammar_schema(specs)


__all__ = [
    "build_tool_models",
    "enrich_tool_definitions",
    "grammar_schema",
    "json_schema",
    "openai_tools",
    "validate_tool_call",
]
