"""Tool schema parsing and grammar generation."""
from __future__ import annotations

import json
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

from .definition import ToolDefinition

_TYPE_MAP = {"string": str, "integer": int, "number": float, "boolean": bool}


def build_tool_models(tools: list[ToolDefinition]) -> dict[str, type]:
    """Dynamically create Pydantic models for strict parameter validation."""
    models: dict[str, type] = {}
    for tool in tools:
        fields: dict[str, tuple[type, Any]] = {}
        for name, definition in tool.parameters.items():
            annotation = _TYPE_MAP.get(definition.get("type"), Any)
            default = ... if definition.get("required") else None
            constraints = {}
            if definition.get("minimum") is not None:
                constraints["ge"] = definition["minimum"]
            if definition.get("maximum") is not None:
                constraints["le"] = definition["maximum"]
            fields[name] = (annotation, Field(default, **constraints) if constraints else default)
        
        # extra="forbid" ensures the LLM cannot hallucinate extra arguments
        create_kwargs: dict[str, Any] = {
            "__config__": ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True),
            **fields,
        }
        models[tool.name] = cast(
            "type[BaseModel]",
            create_model(
                f"{tool.name.title().replace('_', '')}Params",
                **create_kwargs,
            ),
        )
    return models


def grammar_schema(tools: list[ToolDefinition]) -> str:
    """JSON schema for llama_cpp.LlamaGrammar fallback.
    
    A response is either a normal final answer or one exact tool envelope.
    """
    variants: list[dict[str, Any]] = [
        {
            "type": "object", 
            "properties": {"kind": {"const": "final"}, "content": {"type": "string"}},
            "required": ["kind", "content"], 
            "additionalProperties": False,
        },
        {
            "type": "object", 
            "properties": {"kind": {"const": "step_answer"}, "content": {"type": "string"}},
            "required": ["kind", "content"], 
            "additionalProperties": False,
        },
    ]
    
    for tool in tools:
        properties = {
            name: {key: value for key, value in spec.items()
                   if key in {"type", "enum", "minimum", "maximum"}}
            for name, spec in tool.parameters.items()
        }
        variants.append({
            "type": "object",
            "properties": {
                "kind": {"const": "tool"}, 
                "tool": {"const": tool.name},
                "params": {
                    "type": "object", 
                    "properties": properties,
                    "required": [name for name, spec in tool.parameters.items() if spec.get("required")],
                    "additionalProperties": False
                }
            },
            "required": ["kind", "tool", "params"], 
            "additionalProperties": False,
        })
        
    return json.dumps({"oneOf": variants}, ensure_ascii=False)
