from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..adapters.persistence import SqliteTaskStore
from ..application.tasks import TaskService
from ..application.tools import ToolExecutionService
from ..domain.security import DataClassification
from ..domain.tools import SideEffect, ToolSpec
from ..plugins.loader import load_entrypoint_plugins
from ..plugins.registry import PluginRegistry
from ..ports import AgentTaskExecutor, ToolExecutor
from ..settings import ArchitectureSettings
from .runtime import configure_tool_service


@dataclass(frozen=True, slots=True)
class PlatformContainer:
    settings: ArchitectureSettings
    plugins: PluginRegistry
    tools: ToolExecutionService
    tasks: TaskService


def _side_effect(legacy: dict[str, Any]) -> SideEffect:
    policy = legacy.get("policy", "read_only")
    if policy == "desktop_ack":
        return SideEffect.DESKTOP
    if policy == "code_write":
        return SideEffect.WRITE
    if policy == "code_execute":
        return SideEffect.EXECUTE
    return SideEffect.READ


def build_container(base_dir: Path, legacy_tools: "list[Any]",
                    legacy_executors: dict[str, ToolExecutor], task_executor: AgentTaskExecutor) -> PlatformContainer:
    """Build the DI container.

    ``legacy_tools`` may be either:
    - A list of ToolSpec-like objects (with .name, .description, .policy, …
      attributes) — the preferred path when coming from TOOL_REGISTRY.get_all().
    - A list of dicts with at minimum a ``name`` key — the old format.
    """
    settings = ArchitectureSettings.load(base_dir)
    plugins = PluginRegistry(settings.allowed_plugins)
    for legacy in legacy_tools:
        # Support both ToolSpec objects and legacy dicts
        if hasattr(legacy, "name"):
            # ToolSpec object from the new registry
            name = str(legacy.name)
            executor = legacy_executors.get(name) or getattr(legacy, "handler", None)
            if executor is None:
                continue
            policy = str(getattr(legacy, "policy", "read_only"))
            scope = {
                "desktop_ack": "desktop.action",
                "code_write": "project.write",
                "code_execute": "project.execute",
            }.get(policy, "project.read")
            parameters = getattr(legacy, "json_schema", None) or getattr(legacy, "parameters", {})
            spec = ToolSpec(
                name=name,
                description=str(getattr(legacy, "description", "")),
                input_schema=dict(parameters) if isinstance(parameters, dict) else {},
                timeout_seconds=int(getattr(legacy, "timeout_seconds", 10)),
                side_effect=_side_effect({"policy": policy}),
                requires_approval=bool(getattr(legacy, "requires_approval", False)),
                required_scope=scope,
                idempotent=bool(getattr(legacy, "idempotent", True)),
                maximum_classification=DataClassification.RESTRICTED,
                plugin_id="builtin.legacy-tools",
            )
        else:
            # Legacy dict format — must have 'name' key
            if "name" not in legacy:
                continue  # skip malformed entries (e.g. raw JSON schemas)
            name = str(legacy["name"])
            executor = legacy_executors.get(name)
            if executor is None:
                continue
            policy = str(legacy.get("policy", "read_only"))
            scope = {
                "desktop_ack": "desktop.action",
                "code_write": "project.write",
                "code_execute": "project.execute",
            }.get(policy, "project.read")
            schema = legacy.get("schema") or legacy.get("json_schema") or legacy.get("parameters", {})
            spec = ToolSpec(
                name=name, description=str(legacy.get("description", "")),
                input_schema=dict(schema) if isinstance(schema, dict) else {},
                timeout_seconds=int(legacy.get("timeout_seconds", 10)),
                side_effect=_side_effect(legacy),
                requires_approval=bool(legacy.get("requires_approval", False)),
                required_scope=scope,
                idempotent=bool(legacy.get("idempotent", True)),
                maximum_classification=DataClassification.RESTRICTED,
                plugin_id="builtin.legacy-tools",
            )
        plugins.register_tool(spec, executor)
    load_entrypoint_plugins(plugins)
    tools = ToolExecutionService(plugins)
    configure_tool_service(tools)
    store = SqliteTaskStore(settings.data_dir / "tasks.sqlite")
    task_service = TaskService(store, task_executor, settings.agent_capabilities)
    return PlatformContainer(settings, plugins, tools, task_service)
