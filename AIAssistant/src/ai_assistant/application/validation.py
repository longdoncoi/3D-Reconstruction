"""Minimal JSON-Schema subset validation used at every tool protocol boundary."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_TYPES: dict[str, type | tuple[type, ...]] = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}


def validate_input(schema: Mapping[str, Any], value: Mapping[str, Any]) -> str | None:
    if not schema:
        return None
    if schema.get("type") not in (None, "object"):
        return "Tool schema root must be an object"
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for name in required:
        if name not in value:
            return f"Missing required parameter: {name}"
    if schema.get("additionalProperties") is False:
        # ``request_id`` is framework-generated correlation metadata for a Qt
        # action. It is not model-controlled input and is intentionally absent
        # from public tool schemas.
        unknown = sorted(set(value) - set(properties) - {"request_id"})
        if unknown:
            return f"Unknown parameter(s): {', '.join(unknown)}"
    for name, item in value.items():
        definition = properties.get(name)
        if not definition:
            continue
        expected = _TYPES.get(definition.get("type"))
        if expected and (not isinstance(item, expected) or (definition.get("type") == "integer" and isinstance(item, bool))):
            return f"Parameter {name} must be {definition['type']}"
        if "enum" in definition and item not in definition["enum"]:
            return f"Parameter {name} is not an allowed value"
        if "minimum" in definition and item < definition["minimum"]:
            return f"Parameter {name} is below its minimum"
        if "maximum" in definition and item > definition["maximum"]:
            return f"Parameter {name} exceeds its maximum"
    return None
