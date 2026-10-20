from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..adapters.persistence import SqliteTaskStore
from ..adapters.persistence.storage_crypto import FieldCipher
from ..application.coordination import TaskCoordinator
from ..application.tasks import TaskService
from ..application.tools import ToolExecutionService
from ..domain.security import DataClassification
from ..domain.tools import SideEffect, ToolSpec
from ..observability.metrics import record_recovery
from ..observability.tool_events import audit_tool_event
from ..plugins.loader import load_entrypoint_plugins
from ..plugins.registry import PluginRegistry
from ..ports import AgentTaskExecutor, ToolEventSink, ToolExecutor, ToolGateway
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
    storage_cipher: FieldCipher | None = None


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
        maximum_classification=DataClassification.RESTRICTED,
        plugin_id="builtin.legacy-tools",
    )


def build_container(base_dir: Path, legacy_tools: "list[Any]",
                    legacy_executors: Mapping[str, ToolExecutor], task_executor: AgentTaskExecutor,
                    coordinator: TaskCoordinator | None = None,
                    tool_event_sink: ToolEventSink | None = None,
                    storage_cipher: FieldCipher | None = None) -> PlatformContainer:
    """Build the DI container.

    ``legacy_tools`` is a list of catalog entries (``ToolDefinition`` objects
    from :func:`ai_assistant.tools.factory.create_tool_registry`). Each entry is
    converted once, through :func:`tool_spec_from`, into the domain
    ``ToolSpec`` the single execution core consumes. The legacy dict format is
    no longer accepted.

    ``tool_event_sink`` observes every execution at the chokepoint (agent loop,
    HITL approval, desktop action, A2A task worker). The default is
    :func:`~ai_assistant.observability.tool_events.audit_tool_event`, which
    writes the redacted one-line record; tests inject their own.
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
    # The sink is what makes the chokepoint observable: without it an approved
    # action or an A2A task run leaves no trace outside the agent loop.
    sink = audit_tool_event if tool_event_sink is None else tool_event_sink
    tools = ToolExecutionService(plugins, event_sink=sink)
    gateway = PlatformToolGateway(tools)
    # One coordinator for the whole platform: the A2A task service shares it
    # with AgentService so ``POST /tasks/{id}:cancel`` stops the LangGraph loop
    # instead of only writing the task row (P0 — A2A cancel). It also carries
    # the recovery-time observer (cancel request → runner stopped, SCORECARD #4).
    shared_coordinator = (
        coordinator
        if coordinator is not None
        else TaskCoordinator(recovery_observer=record_recovery)
    )
    store = SqliteTaskStore(settings.data_dir / "tasks.sqlite", cipher=storage_cipher)
    task_service = TaskService(store, task_executor, settings.agent_capabilities, shared_coordinator)
    return PlatformContainer(
        settings, plugins, tools, task_service, gateway, shared_coordinator,
        storage_cipher=storage_cipher,
    )
