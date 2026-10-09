"""Legacy re-export of tool_contract."""
from ai_assistant.tools.tool_contract import (
    build_tool_models,
    enrich_tool_definitions,
    grammar_schema,
    json_schema,
    openai_tools,
    validate_tool_call,
)

__all__ = [
    "build_tool_models",
    "enrich_tool_definitions",
    "grammar_schema",
    "json_schema",
    "openai_tools",
    "validate_tool_call",
]
