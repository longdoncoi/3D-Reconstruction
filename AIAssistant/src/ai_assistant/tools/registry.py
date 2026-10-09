"""Tool registry for organizing and retrieving tool definitions."""
from __future__ import annotations

import logging

from .action_manifest import action_catalog, action_ids
from .definition import ToolDefinition
from .schema import build_tool_models, grammar_schema

logger = logging.getLogger("ai_assistant.tools.registry")


class ToolRegistry:
    """Registry for agent tools."""
    
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._models: dict[str, type] | None = None
        self._grammar: str | None = None

    def register(self, tool: ToolDefinition) -> None:
        """Register a single tool."""
        self._tools[tool.name] = tool
        self._models = None
        self._grammar = None

    def register_all(self, tools: list[ToolDefinition]) -> None:
        """Register a list of tools."""
        for tool in tools:
            self.register(tool)

    def get(self, tool_name: str) -> ToolDefinition | None:
        """Retrieve a tool by name."""
        return self._tools.get(tool_name)

    def get_all(self) -> list[ToolDefinition]:
        """Get all registered tools."""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    @property
    def models(self) -> dict[str, type]:
        """Get the compiled Pydantic models for all tools."""
        if self._models is None:
            # Special case for application_action: inject dynamic enums from the manifest
            tools_for_models = []
            for spec in self._tools.values():
                if spec.name == "application_action" and "action" in spec.parameters:
                    # Inject current enums
                    new_params = dict(spec.parameters)
                    new_params["action"] = {**new_params["action"], "enum": sorted(action_ids())}
                    # Replace description with dynamic catalog
                    new_desc = f"{spec.description}\n{action_catalog()}"
                    tools_for_models.append(ToolDefinition(
                        name=spec.name,
                        description=new_desc,
                        parameters=new_params,
                        timeout_seconds=spec.timeout_seconds,
                        policy=spec.policy,
                        requires_approval=spec.requires_approval,
                        idempotent=spec.idempotent,
                        handler=spec.handler,
                    ))
                else:
                    tools_for_models.append(spec)
                    
            self._models = build_tool_models(tools_for_models)
        return self._models

    @property
    def grammar(self) -> str:
        """Get the llama.cpp grammar JSON schema for all tools."""
        if self._grammar is None:
            self._grammar = grammar_schema(self.get_all())
        return self._grammar
        
    def get_openai_tools(self) -> list[dict]:
        """Get the tools in OpenAI's API format."""
        # Refresh for dynamic enums (like application_action)
        result = []
        for spec in self.get_all():
            if spec.name == "application_action" and "action" in spec.parameters:
                new_params = dict(spec.parameters)
                new_params["action"] = {**new_params["action"], "enum": sorted(action_ids())}
                new_desc = f"{spec.description}\n{action_catalog()}"
                dynamic_spec = ToolDefinition(
                    name=spec.name, description=new_desc, parameters=new_params,
                    timeout_seconds=spec.timeout_seconds, policy=spec.policy,
                    requires_approval=spec.requires_approval, idempotent=spec.idempotent,
                )
                result.append(dynamic_spec.to_openai_format())
            else:
                result.append(spec.to_openai_format())
        return result
