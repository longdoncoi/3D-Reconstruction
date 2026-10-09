"""Tool factory for constructing the registry from configuration."""
from __future__ import annotations

import json
from typing import Any, Callable

from ai_assistant.config.paths import get_paths

from .builtin import (
    tool_analyze_code,
    tool_application_action,
    tool_create_directory,
    tool_find_files,
    tool_get_project_status,
    tool_git_diff,
    tool_list_directory,
    tool_multi_replace_file_content,
    tool_patch_file,
    tool_rag_search,
    tool_read_file,
    tool_replace_file_content,
    tool_run_command,
    tool_search_text,
    tool_transfer_to_chatbot_agent,
    tool_transfer_to_code_agent,
    tool_transfer_to_toolapp_agent,
    tool_validate_file,
    tool_write_file,
)
from .definition import ToolDefinition
from .registry import ToolRegistry


def builtin_handlers() -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    """Map tool names to their builtin handler functions."""
    return {
        "read_file": tool_read_file,
        "list_directory": tool_list_directory,
        "search_text": tool_search_text,
        "analyze_code": tool_analyze_code,
        "find_files": tool_find_files,
        "git_diff": tool_git_diff,
        "write_file": tool_write_file,
        "run_command": tool_run_command,
        "get_project_status": tool_get_project_status,
        "validate_file": tool_validate_file,
        "patch_file": tool_patch_file,
        "replace_file_content": tool_replace_file_content,
        "multi_replace_file_content": tool_multi_replace_file_content,
        "create_directory": tool_create_directory,
        "application_action": tool_application_action,
        "app_action_viewer": tool_application_action,
        "app_action_reconstruction": tool_application_action,
        "app_action_ai": tool_application_action,
        "app_action_general": tool_application_action,
        "rag_search": tool_rag_search,
        "transfer_to_code_agent": tool_transfer_to_code_agent,
        "transfer_to_toolapp_agent": tool_transfer_to_toolapp_agent,
        "transfer_to_chatbot_agent": tool_transfer_to_chatbot_agent,
    }


def _default_tool_contract_overrides() -> dict[str, dict[str, Any]]:
    return {
        "application_action": {"timeout_seconds": 30, "policy": "desktop_ack"},
        "write_file": {"timeout_seconds": 10, "policy": "code_write", "requires_approval": True},
        "patch_file": {"timeout_seconds": 10, "policy": "code_write", "requires_approval": True},
        "replace_file_content": {"timeout_seconds": 10, "policy": "code_write", "requires_approval": True},
        "multi_replace_file_content": {"timeout_seconds": 10, "policy": "code_write", "requires_approval": True},
        "create_directory": {"timeout_seconds": 10, "policy": "code_write", "requires_approval": True},
        "run_command": {"timeout_seconds": 120, "policy": "code_execute", "requires_approval": True},
    }


def create_tool_registry() -> ToolRegistry:
    """Create and populate the global ToolRegistry from configuration."""
    registry = ToolRegistry()
    paths = get_paths()
    config_file = paths.project_root / "Config" / "agent_tools.json"
    
    if not config_file.exists():
        # Fallback or initialization error
        return registry

    with config_file.open(encoding="utf-8") as f:
        tools_def = json.load(f)

    handlers = builtin_handlers()
    overrides = _default_tool_contract_overrides()

    for item in tools_def:
        name = item["name"]
        override = overrides.get(name, {})
        
        spec = ToolDefinition(
            name=name,
            description=item.get("description", ""),
            parameters=item.get("parameters", {}),
            timeout_seconds=override.get("timeout_seconds", 10),
            policy=override.get("policy", "read_only"),
            requires_approval=override.get("requires_approval", False),
            idempotent=override.get("idempotent", True),
            handler=handlers.get(name),
        )
        registry.register(spec)

    # Note: `application_action` might not be in the file depending on the generation logic,
    # as it was appended dynamically. Let's make sure it's there.
    if registry.get("application_action") is None:
        registry.register(ToolDefinition(
            name="application_action",
            description="Execute exactly ONE canonical desktop action. Choose the id whose meaning matches the CURRENT plan step.",
            parameters={
                "action": {"type": "string", "description": "Canonical desktop action id", "required": True},
                "language": {"type": "string", "description": "Required only for language.change: vi or en", "required": False},
                "username": {"type": "string", "description": "Optional login username", "required": False},
                "password": {"type": "string", "description": "Optional login password", "required": False},
            },
            timeout_seconds=30,
            policy="desktop_ack",
            handler=handlers.get("application_action")
        ))
        
    return registry
