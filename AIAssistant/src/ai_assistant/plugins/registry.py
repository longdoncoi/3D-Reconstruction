from __future__ import annotations

from dataclasses import dataclass

from ..domain.tools import ToolSpec
from ..ports import ToolExecutor


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    spec: ToolSpec
    executor: ToolExecutor


class PluginRegistry:
    """Explicit capability registry; duplicate tool names fail during bootstrap."""

    def __init__(self, allowed_plugins: frozenset[str]) -> None:
        self._allowed_plugins = allowed_plugins
        self._tools: dict[str, RegisteredTool] = {}

    def register_tool(self, spec: ToolSpec, executor: ToolExecutor) -> None:
        if spec.plugin_id not in self._allowed_plugins:
            msg = f"Plugin is not enabled: {spec.plugin_id}"
            raise PermissionError(msg)
        if spec.name in self._tools:
            msg = f"Duplicate tool registration: {spec.name}"
            raise ValueError(msg)
        self._tools[spec.name] = RegisteredTool(spec, executor)

    def tool(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(item.spec for item in self._tools.values())

    def allows(self, plugin_id: str) -> bool:
        return plugin_id in self._allowed_plugins
