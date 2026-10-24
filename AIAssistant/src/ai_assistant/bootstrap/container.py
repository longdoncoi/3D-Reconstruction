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
from ..domain.specialists import Specialist
from ..domain.tools import BUILTIN_PLUGIN_ID, SideEffect, ToolSpec
from ..observability.metrics import record_recovery
from ..observability.tool_events import audit_tool_event
from ..plugins.loader import load_entrypoint_plugins
from ..plugins.registry import PluginRegistry
from ..ports import AgentTaskExecutor, ToolEventSink, ToolExecutor, ToolGateway
from ..settings import ArchitectureSettings
from ..tools.aliases import canonical_tool_name
from ..tools.definition import ToolDefinition, parameters_from_json_schema
from ..tools.registry import ToolRegistry
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


_POLICY_TO_SIDE_EFFECT: dict[str, SideEffect] = {
    "read_only": SideEffect.READ,
    "desktop_ack": SideEffect.DESKTOP,
    "code_write": SideEffect.WRITE,
    "code_execute": SideEffect.EXECUTE,
}

_POLICY_TO_SCOPE: dict[str, str] = {
    "read_only": "project.read",
    "desktop_ack": "desktop.action",
    "code_write": "project.write",
    "code_execute": "project.execute",
}

_SIDE_EFFECT_TO_POLICY: dict[SideEffect, str] = {
    side_effect: policy for policy, side_effect in _POLICY_TO_SIDE_EFFECT.items()
}


def _side_effect(policy: str) -> SideEffect:
    """Map a runtime policy label to its side effect, fail-closed.

    An unknown label is a policy gap, not a read-only tool: the old fallback
    silently weakened a write/drive tool whose label was misspelled.
    """
    try:
        return _POLICY_TO_SIDE_EFFECT[policy]
    except KeyError:
        msg = f"Unknown tool policy: {policy!r}"
        raise ValueError(msg) from None


def _required_scope(policy: str) -> str:
    """Map a runtime policy label to the ToolSpec scope the gateway enforces."""
    try:
        return _POLICY_TO_SCOPE[policy]
    except KeyError:
        msg = f"Unknown tool policy: {policy!r}"
        raise ValueError(msg) from None


def _classification(label: str) -> DataClassification:
    """Map a runtime classification label to the domain enum.

    An unknown or missing label is the **most** protective classification
    (``REGULATED``), never the least: a label nobody recognises must not lower
    a tool's data ceiling.  ``RESTRICTED`` was still below the top of the
    lattice, so the old fallback let a regulated-data principal through.
    """
    return {
        "public": DataClassification.PUBLIC,
        "internal": DataClassification.INTERNAL,
        "restricted": DataClassification.RESTRICTED,
        "regulated": DataClassification.REGULATED,
    }.get(label.casefold(), DataClassification.REGULATED)


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
    if executor is None:
        return None
    # Plugin entries carry the authored domain policy verbatim. Re-deriving it
    # from display labels would lose the plugin's declared scope/approval and
    # create a second, weaker policy source.
    existing_spec = getattr(entry, "spec", None)
    if isinstance(existing_spec, ToolSpec):
        return existing_spec
    raw_name = str(entry.name)
    name = canonical_tool_name(raw_name)
    if name != raw_name:
        # An alias is an advertised spelling, not a policy row. Refusing it
        # here means a caller that bypasses ``build_container``'s skip cannot
        # turn ``app_action_viewer`` into a read-only spec.
        msg = f"Tool alias {raw_name!r} has no execution policy; use {name!r}"
        raise ValueError(msg)
    policy_value = getattr(entry, "policy", None)
    if policy_value is None:
        # A missing label is not "read-only": a mutating tool whose policy was
        # never authored would otherwise inherit the weakest scope and could
        # be driven by a read-only principal. Unknown labels already fail in
        # ``_side_effect``; missing ones must fail here for the same reason.
        msg = f"Tool {name!r} has no execution policy"
        raise ValueError(msg)
    policy = str(policy_value)
    parameters = getattr(entry, "json_schema", None) or getattr(entry, "parameters", None)
    # The tool's data ceiling comes from the tool's own contract. It used to
    # be a constant (``RESTRICTED`` for every tool), which made the gateway's
    # classification check a comparison against a constant — the gate could
    # not tell a read-only tool from one that rewrites the project.
    classification = getattr(entry, "classification", None)
    return ToolSpec(
        name=name,
        description=str(getattr(entry, "description", "")),
        input_schema=dict(parameters) if isinstance(parameters, dict) else {},
        timeout_seconds=int(getattr(entry, "timeout_seconds", 10)),
        side_effect=_side_effect(policy),
        requires_approval=bool(getattr(entry, "requires_approval", False)),
        required_scope=_required_scope(policy),
        maximum_classification=(
            _classification(str(classification)) if classification else DataClassification.REGULATED
        ),
        plugin_id=str(getattr(entry, "plugin_id", BUILTIN_PLUGIN_ID)),
    )


def tool_definition_from_plugin(entry: Any) -> ToolDefinition | None:
    """Adapt one registered plugin tool to the agent-facing catalogue.

    The plugin's :class:`ToolSpec` stays the policy source of truth; the
    definition only supplies the schema/handler shape the model and graph
    need.  Plugin tools are owned by :class:`Specialist.PLUGIN`, never by a
    builtin specialist, and therefore require an explicit dynamic policy row.
    """
    spec = getattr(entry, "spec", None)
    executor = getattr(entry, "executor", None)
    if not isinstance(spec, ToolSpec) or executor is None:
        return None
    if canonical_tool_name(spec.name) != spec.name:
        # Aliases are advertised names, not separate execution surfaces.
        return None
    schema = dict(spec.input_schema) if isinstance(spec.input_schema, Mapping) else {}
    return ToolDefinition(
        name=spec.name,
        description=spec.description,
        parameters=parameters_from_json_schema(schema),
        timeout_seconds=spec.timeout_seconds,
        policy=_SIDE_EFFECT_TO_POLICY.get(spec.side_effect, "read_only"),
        requires_approval=spec.requires_approval,
        classification=spec.maximum_classification.value,
        handler=executor,
        plugin_id=spec.plugin_id,
        specialists=frozenset({Specialist.PLUGIN}),
        input_schema=schema or None,
        spec=spec,
    )


def merge_plugin_tools(registry: ToolRegistry, plugins: PluginRegistry) -> frozenset[str]:
    """Add enabled third-party tools to the single agent catalogue.

    Builtins are already present (and reserved); this only adds contributions
    from enabled plugin distributions. The returned names are the dynamic
    :class:`Specialist.PLUGIN` policy row the composition root binds into the
    supervisor, so a plugin tool is visible to the agent *and* authorised by
    an explicit row rather than falling back to a builtin specialist.
    """
    merged: set[str] = set()
    for entry in plugins.tools():
        if entry.spec.plugin_id == BUILTIN_PLUGIN_ID:
            continue
        if registry.get(entry.spec.name) is not None:
            # A plugin may not shadow a builtin or another catalogue entry.
            continue
        definition = tool_definition_from_plugin(entry)
        if definition is None:
            continue
        registry.register(definition)
        merged.add(definition.name)
    return frozenset(merged)


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
        if canonical_tool_name(name) != name:
            # The model still sees the alias; the execution catalogue holds
            # only the canonical tool, so one invocation has one policy row.
            continue
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
    task_service = TaskService(store, task_executor, settings.agent_capabilities,
                               shared_coordinator,
                               max_concurrent_tasks=settings.max_concurrent_tasks)
    return PlatformContainer(
        settings, plugins, tools, task_service, gateway, shared_coordinator,
        storage_cipher=storage_cipher,
    )
