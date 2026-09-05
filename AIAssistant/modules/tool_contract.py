"""Schema-derived validation and policy metadata for agent tools.

The same definitions are consumed by the local reasoner, the MCP boundary and
the approval policy.  Keeping the operational metadata next to the JSON
schema prevents a new tool from accidentally being exposed without a timeout
or an explicit least-privilege policy.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import ConfigDict, Field, ValidationError, create_model

from .action_manifest import validate_action_params

_TYPE_MAP = {"string": str, "integer": int, "number": float, "boolean": bool}


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


def json_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical input JSON Schema for one tool definition."""
    parameters = tool.get("parameters", {})
    return {
        "type": "object",
        "properties": {
            name: {
                key: value for key, value in definition.items()
                if key in {"type", "description", "enum", "minimum", "maximum"}
            }
            for name, definition in parameters.items()
        },
        "required": [name for name, spec in parameters.items() if spec.get("required")],
        "additionalProperties": False,
    }


def enrich_tool_definitions(tool_definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add explicit operational metadata without changing caller-owned data."""
    enriched = []
    for source in tool_definitions:
        tool = dict(source)
        tool["parameters"] = dict(source.get("parameters", {}))
        for key, value in _DEFAULT_CONTRACT.items():
            tool.setdefault(key, value)
        for key, value in _TOOL_CONTRACT_OVERRIDES.get(tool["name"], {}).items():
            tool[key] = value
        tool["schema"] = json_schema(tool)
        enriched.append(tool)
    return enriched


def build_tool_models(tool_definitions: list[dict[str, Any]]) -> dict[str, type]:
    models: dict[str, type] = {}
    for tool in tool_definitions:
        fields: dict[str, tuple[type, Any]] = {}
        for name, definition in tool.get("parameters", {}).items():
            annotation = _TYPE_MAP.get(definition.get("type"), Any)
            default = ... if definition.get("required") else None
            constraints = {}
            if definition.get("minimum") is not None:
                constraints["ge"] = definition["minimum"]
            if definition.get("maximum") is not None:
                constraints["le"] = definition["maximum"]
            fields[name] = (annotation, Field(default, **constraints) if constraints else default)
        models[tool["name"]] = create_model(
            f"{tool['name'].title().replace('_', '')}Params",
            __config__=ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True),
            **fields,
        )
    return models


def openai_tools(tool_definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the legacy AGENT_TOOLS list into llama.cpp/OpenAI tools format."""
    result = []
    for tool in tool_definitions:
        schema = json_schema(tool)
        result.append({"type": "function", "function": {
            "name": tool["name"], "description": tool["description"],
            "parameters": schema,
        }})
    return result


def grammar_schema(tool_definitions: list[dict[str, Any]]) -> str:
    """JSON schema for llama_cpp.LlamaGrammar fallback.

    A response is either a normal final answer or one exact tool envelope.
    """
    variants: list[dict[str, Any]] = [{
        "type": "object", "properties": {"kind": {"const": "final"}, "content": {"type": "string"}},
        "required": ["kind", "content"], "additionalProperties": False,
    }]
    for tool in tool_definitions:
        properties = {
            name: {key: value for key, value in spec.items()
                   if key in {"type", "enum", "minimum", "maximum"}}
            for name, spec in tool.get("parameters", {}).items()
        }
        variants.append({
            "type": "object",
            "properties": {"kind": {"const": "tool"}, "tool": {"const": tool["name"]},
                           "params": {"type": "object", "properties": properties,
                                      "required": [name for name, spec in tool.get("parameters", {}).items() if spec.get("required")],
                                      "additionalProperties": False}},
            "required": ["kind", "tool", "params"], "additionalProperties": False,
        })
    return json.dumps({"oneOf": variants}, ensure_ascii=False)


def validate_tool_call(tool_name: str, params: dict[str, Any], models: dict[str, type]) -> tuple[dict[str, Any] | None, str | None]:
    model = models.get(tool_name)
    if model is None:
        return None, f"Unknown tool: {tool_name}"
    try:
        validated = model.model_validate(params).model_dump(exclude_none=True)
    except ValidationError as error:
        return None, error.json(include_url=False)
    if tool_name == "application_action":
        return validate_action_params(validated)
    return validated, None
