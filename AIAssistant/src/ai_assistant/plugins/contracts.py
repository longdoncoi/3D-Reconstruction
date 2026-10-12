from __future__ import annotations

from typing import Protocol

from ..domain.tools import ToolSpec
from ..ports import ToolExecutor


class Plugin(Protocol):
    plugin_id: str
    version: str

    def register(self, registry: "PluginRegistry") -> None: ...


class ToolPlugin(Plugin, Protocol):
    def tool_specs(self) -> tuple[ToolSpec, ...]: ...
    def executor_for(self, tool_name: str) -> ToolExecutor: ...


from .registry import PluginRegistry
