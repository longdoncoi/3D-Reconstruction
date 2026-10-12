"""Entry-point loader for independently versioned third-party plugins."""
from __future__ import annotations

from importlib.metadata import entry_points

from .registry import PluginRegistry


def load_entrypoint_plugins(registry: PluginRegistry, group: str = "ai_assistant.plugins") -> tuple[str, ...]:
    """Load only enabled plugins; invalid plugins fail bootstrap explicitly."""
    loaded: list[str] = []
    for entry_point in entry_points().select(group=group):
        plugin = entry_point.load()()
        plugin_id = str(getattr(plugin, "plugin_id", ""))
        if not registry.allows(plugin_id):
            continue
        register = getattr(plugin, "register", None)
        if not callable(register):
            msg = f"Plugin {plugin_id} has no register method"
            raise TypeError(msg)
        register(registry)
        loaded.append(plugin_id)
    return tuple(loaded)
