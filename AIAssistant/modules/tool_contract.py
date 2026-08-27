"""Schema-derived validation and constrained-decoding support for agent tools."""
from __future__ import annotations

import json
from typing import Any

from pydantic import ConfigDict, ValidationError, create_model

from .action_manifest import validate_action_params

_TYPE_MAP = {"string": str, "integer": int, "number": float, "boolean": bool}


def build_tool_models(tool_definitions: list[dict[str, Any]]) -> dict[str, type]:
    models: dict[str, type] = {}
    for tool in tool_definitions:
        fields: dict[str, tuple[type, Any]] = {}
        for name, definition in tool.get("parameters", {}).items():
            annotation = _TYPE_MAP.get(definition.get("type"), Any)
            default = ... if definition.get("required") else None
            fields[name] = (annotation, default)
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
        properties = {
            name: {key: value for key, value in definition.items() if key in {"type", "description", "enum"}}
            for name, definition in tool.get("parameters", {}).items()
        }
        result.append({"type": "function", "function": {
            "name": tool["name"], "description": tool["description"],
            "parameters": {"type": "object", "properties": properties,
                           "required": [name for name, spec in tool.get("parameters", {}).items() if spec.get("required")],
                           "additionalProperties": False},
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
        properties = {name: {"type": spec.get("type", "string")} for name, spec in tool.get("parameters", {}).items()}
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
