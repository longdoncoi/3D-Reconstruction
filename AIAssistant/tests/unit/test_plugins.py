"""Unit tests for the plugin registry and entry-point loader.

Covers ``PluginRegistry`` capability gating and the ``importlib.metadata``
entry-point loading path in ``ai_assistant.plugins.loader`` using fakes, so no
third-party plugin package is required.
"""
from __future__ import annotations

import unittest
from unittest import mock

from ai_assistant.domain.tools import SideEffect, ToolSpec
from ai_assistant.plugins import loader as loader_module
from ai_assistant.plugins.registry import PluginRegistry


def _spec(name: str, plugin_id: str = "builtin") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} tool",
        input_schema={"type": "object"},
        timeout_seconds=5,
        side_effect=SideEffect.READ,
        plugin_id=plugin_id,
    )


class PluginRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = PluginRegistry(frozenset({"builtin", "extra"}))

    def test_allows_known_plugin(self):
        self.assertTrue(self.registry.allows("builtin"))
        self.assertFalse(self.registry.allows("rogue"))

    def test_register_and_lookup_tool(self):
        executor = object()
        self.registry.register_tool(_spec("echo"), executor)
        registered = self.registry.tool("echo")
        self.assertIsNotNone(registered)
        self.assertIs(registered.executor, executor)
        self.assertEqual([spec.name for spec in self.registry.specs()], ["echo"])

    def test_unknown_plugin_is_rejected(self):
        with self.assertRaises(PermissionError):
            self.registry.register_tool(_spec("x", plugin_id="rogue"), object())

    def test_duplicate_tool_name_is_rejected(self):
        self.registry.register_tool(_spec("dup"), object())
        with self.assertRaises(ValueError):
            self.registry.register_tool(_spec("dup"), object())

    def test_missing_tool_returns_none(self):
        self.assertIsNone(self.registry.tool("nope"))


class _FakePlugin:
    def __init__(self, plugin_id, register=None):
        self.plugin_id = plugin_id
        self.register = register


class _FakeEntryPoint:
    def __init__(self, factory):
        self._factory = factory

    def load(self):
        return self._factory


class _FakeEntryPoints:
    def __init__(self, points):
        self._points = points
        self.groups = []

    def select(self, group):
        self.groups.append(group)
        return list(self._points)


class EntryPointLoaderTests(unittest.TestCase):
    def _patch(self, points):
        fake = _FakeEntryPoints(points)
        return mock.patch.object(loader_module, "entry_points", return_value=fake), fake

    def test_loads_enabled_plugins_only(self):
        called = []

        def register(reg):
            called.append(reg)

        enabled = _FakePlugin("builtin", register)
        disabled = _FakePlugin("nope", register)
        patch, fake = self._patch([
            _FakeEntryPoint(lambda: enabled),
            _FakeEntryPoint(lambda: disabled),
        ])
        registry = PluginRegistry(frozenset({"builtin"}))
        with patch:
            loaded = loader_module.load_entrypoint_plugins(registry)
        self.assertEqual(loaded, ("builtin",))
        self.assertEqual(called, [registry])
        self.assertEqual(fake.groups, ["ai_assistant.plugins"])

    def test_custom_group_is_forwarded(self):
        patch, fake = self._patch([])
        with patch:
            loader_module.load_entrypoint_plugins(PluginRegistry(frozenset()), group="custom")
        self.assertEqual(fake.groups, ["custom"])

    def test_plugin_without_register_raises(self):
        plugin = _FakePlugin("builtin")
        plugin.register = None
        patch, _ = self._patch([_FakeEntryPoint(lambda: plugin)])
        with patch, self.assertRaises(TypeError):
            loader_module.load_entrypoint_plugins(PluginRegistry(frozenset({"builtin"})))

    def test_plugin_missing_id_is_treated_as_disabled(self):
        plugin = _FakePlugin("", lambda reg: None)
        patch, _ = self._patch([_FakeEntryPoint(lambda: plugin)])
        with patch:
            loaded = loader_module.load_entrypoint_plugins(PluginRegistry(frozenset()))
        self.assertEqual(loaded, ())


if __name__ == "__main__":
    unittest.main()
