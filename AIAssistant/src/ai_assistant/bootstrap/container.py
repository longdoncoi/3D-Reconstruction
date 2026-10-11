from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..adapters.persistence import SqliteTaskStore
from ..application.coordination import TaskCoordinator
from ..application.tasks import TaskService
from ..application.tools import ToolExecutionService
from ..domain.security import DataClassification
from ..domain.tools import SideEffect, ToolSpec
from ..plugins.loader import load_entrypoint_plugins
from ..plugins.registry import PluginRegistry
from ..ports import AgentTaskExecutor, ToolExecutor, ToolGateway
from ..settings import ArchitectureSettings
from .runtime import PlatformToolGateway


@dataclass(frozen=True, slots=True)
class PlatformContainer:
    settings: ArchitectureSettings
    plugins: PluginRegistry
    tools: ToolExecutionService
    tasks: TaskService
    gateway: ToolGateway
    coordinator: TaskCoordinator


def _side_effect(policy: str) -> SideEffect:
    if policy == "desktop_ack":
        return SideEffect.DESKTOP
    if policy == "code_write":
        return SideEffect.WRITE
    if policy == "code_execute":
        return SideEffect.EXECUTE
    return SideEffect.READ


def _required_scope(policy: str) -> str:
    """Map a runtime policy label to the ToolSpec scope the gateway enforces."""
    return {
        "desktop_ack": "desktop.action",
        "code_write": "project.write",
        "code_execute": "project.execute",
    }.get(policy, "project.read")


def tool_spec_from(entry: Any, executor: ToolExecutor | None) -> ToolSpec | None:
    """Bridge one catalog entry (:class:`~ai_assistant.tools.definition.ToolDefinition`)
    to its domain policy record (:class:`~ai_assistant.domain.tools.ToolSpec`).

    This is the single conversion point between the two sides of the tool
    catalog (ADR 0003/0008): the framework-free domain record and the
    executable agent-facing definition. Entries without a usable resolver or
    name are skipped rather than silently registered.
    """
    if not hasattr(entry, "name") or not entry.name:
        return None
    name = str(entry.name)
    if executor is None:
        return None
    policy = str(getattr(entry, "policy", "read_only"))
    parameters = getattr(entry, "json_schema", None) or getattr(entry, "parameters", None)
    return ToolSpec(
        name=name,
        description=str(getattr(entry, "description", "")),
        input_schema=dict(parameters) if isinstance(parameters, dict) else {},
        timeout_seconds=int(getattr(entry, "timeout_seconds", 10)),
        side_effect=_side_effect(policy),
        requires_approval=bool(getattr(entry, "requires_approval", False)),
        required_scope=_required_scope(policy),
        idempotent=bool(getattr(entry, "idempotent", True)),
        maximum_classification=DataClassification.RESTRICTED,
        plugin_id="builtin.legacy-tools",
    )


def build_container(base_dir: Path, legacy_tools: "list[Any]",
                    legacy_executors: dict[str, ToolExecutor], task_executor: AgentTaskExecutor,
                    coordinator: TaskCoordinator | None = None) -> PlatformContainer:
    """Build the DI container.

    ``legacy_tools`` is a list of catalog entries (``ToolDefinition`` objects
    from :func:`ai_assistant.tools.factory.create_tool_registry`). Each entry is
    converted once, through :func:`tool_spec_from`, into the domain
    ``ToolSpec`` the single execution core consumes. The legacy dict format is
    no longer accepted.
    """
    settings = ArchitectureSettings.load(base_dir)
    plugins = PluginRegistry(settings.allowed_plugins)
    for entry in legacy_tools:
        name = str(getattr(entry, "name", ""))
        executor = legacy_executors.get(name) or getattr(entry, "handler", None)
        if executor is None:
            continue
        spec = tool_spec_from(entry, executor)
        if spec is not None:
            plugins.register_tool(spec, executor)
    load_entrypoint_plugins(plugins)
    tools = ToolExecutionService(plugins)
    gateway = PlatformToolGateway(tools)
    store = SqliteTaskStore(settings.data_dir / "tasks.sqlite")
    task_service = TaskService(store, task_executor, settings.agent_capabilities)
    shared_coordinator = coordinator if coordinator is not None else TaskCoordinator()
    return PlatformContainer(settings, plugins, tools, task_service, gateway, shared_coordinator)
