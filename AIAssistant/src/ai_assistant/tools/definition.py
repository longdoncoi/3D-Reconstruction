"""Agent-catalog tool definition.

This is the *executable* counterpart to the pure policy descriptor
:class:`ai_assistant.domain.tools.ToolSpec`. It carries the argument schema, the
runtime policy label and the in-process handler used to populate the shared
:class:`~ai_assistant.tools.registry.ToolRegistry`.

The two models are deliberately distinct responsibilities:

* :class:`~ai_assistant.domain.tools.ToolSpec` — framework-free policy record
  (side effect, required scope, classification) consumed by the single
  execution core (:class:`ai_assistant.application.tools.ToolExecutionService`).
* :class:`ToolDefinition` — agent-facing catalog entry (JSON-schema parameters,
  handler, serialization helpers) rendered into prompts/grammars.

ADR 0003: there is exactly one type named ``ToolSpec`` in the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolDefinition:
    """A registrable agent tool with its parameter schema and handler."""
    name: str
    description: str
    parameters: dict[str, Any]
    timeout_seconds: int = 10
    policy: str = "read_only"
    requires_approval: bool = False
    idempotent: bool = True
    handler: Callable | None = None

    @property
    def json_schema(self) -> dict[str, Any]:
        """Return the canonical input JSON Schema for this tool."""
        return {
            "type": "object",
            "properties": {
                name: {
                    key: value for key, value in definition.items()
                    if key in {"type", "description", "enum", "minimum", "maximum"}
                }
                for name, definition in self.parameters.items()
            },
            "required": [name for name, spec in self.parameters.items() if spec.get("required")],
            "additionalProperties": False,
        }

    def to_openai_format(self) -> dict[str, Any]:
        """Format for Llama.cpp / OpenAI function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            }
        }


__all__ = ["ToolDefinition"]
