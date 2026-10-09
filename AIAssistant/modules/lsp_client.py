"""Legacy re-export of lsp_client."""
from ai_assistant.tools.lsp_client import (
    tool_find_references,
    tool_go_to_definition,
)

__all__ = [
    "tool_find_references",
    "tool_go_to_definition",
]
