"""Tool input validation."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from .action_manifest import validate_action_params


def validate_tool_call(
    tool_name: str, params: dict[str, Any], models: dict[str, type[BaseModel]]
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate raw tool parameters against strict Pydantic models."""
    model = models.get(tool_name)
    if model is None:
        return None, f"Unknown tool: {tool_name}"
        
    try:
        validated = model.model_validate(params).model_dump(exclude_none=True)
    except ValidationError as error:
        return None, error.json(include_url=False)
        
    # Special handling for desktop UI actions (which have their own JSON schema in the manifest)
    if tool_name == "application_action":
        return validate_action_params(validated)
        
    return validated, None
