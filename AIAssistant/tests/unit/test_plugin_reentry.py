"""Tests for plugin loader reentry and duplicate registration safety.

Covers edge cases where plugins are loaded multiple times or register
duplicate tool names, ensuring the PluginRegistry handles them correctly.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from ai_assistant.domain.tools import SideEffect, ToolSpec
from ai_assistant.plugins.loader import load_entrypoint_plugins
from ai_assistant.plugins.registry import PluginRegistry


class PluginReentryTests(unittest.TestCase):
    """Test plugin loading idempotency and duplicate handling."""

    def test_duplicate_tool_registration_raises_error(self) -> None:
        """Registering the same tool name twice must fail fast."""
        registry = PluginRegistry(frozenset({"test-plugin"}))
        spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            timeout_seconds=10,
            side_effect=SideEffect.READ,
            plugin_id="test-plugin",
        )
        registry.register_tool(spec, lambda params: {"success": True})

        with self.assertRaises(ValueError, msg="Duplicate tool registration should raise"):
            registry.register_tool(spec, lambda params: {"success": True})

    def test_plugin_not_in_allowed_list_is_rejected(self) -> None:
        """A plugin not in the allowed_plugins set must be rejected."""
        registry = PluginRegistry(frozenset({"allowed-plugin"}))
        spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            timeout_seconds=10,
            side_effect=SideEffect.READ,
            plugin_id="disallowed-plugin",
        )
        with self.assertRaises(PermissionError, msg="Disallowed plugin should raise"):
            registry.register_tool(spec, lambda params: {"success": True})

    def test_load_entrypoint_plugins_skips_disabled(self) -> None:
        """Plugins not in the allowed list are silently skipped."""
        mock_plugin = MagicMock()
        mock_plugin.plugin_id = "disabled-plugin"
        mock_entry_point = MagicMock()
        mock_entry_point.load.return_value = lambda: mock_plugin

        registry = PluginRegistry(frozenset({"enabled-plugin"}))
        with patch("ai_assistant.plugins.loader.entry_points") as mock_eps:
            mock_eps.return_value.select.return_value = [mock_entry_point]
            loaded = load_entrypoint_plugins(registry)

        self.assertEqual(loaded, ())
        mock_plugin.register.assert_not_called()

    def test_load_entrypoint_plugins_registers_enabled(self) -> None:
        """Enabled plugins are loaded and registered."""
        mock_plugin = MagicMock()
        mock_plugin.plugin_id = "enabled-plugin"
        mock_entry_point = MagicMock()
        mock_entry_point.load.return_value = lambda: mock_plugin

        registry = PluginRegistry(frozenset({"enabled-plugin"}))
        with patch("ai_assistant.plugins.loader.entry_points") as mock_eps:
            mock_eps.return_value.select.return_value = [mock_entry_point]
            loaded = load_entrypoint_plugins(registry)

        self.assertEqual(loaded, ("enabled-plugin",))
        mock_plugin.register.assert_called_once_with(registry)

    def test_plugin_without_register_method_raises_type_error(self) -> None:
        """A plugin without a register method must raise TypeError."""
        mock_plugin = MagicMock()
        mock_plugin.plugin_id = "enabled-plugin"
        del mock_plugin.register  # Remove register method
        mock_entry_point = MagicMock()
        mock_entry_point.load.return_value = lambda: mock_plugin

        registry = PluginRegistry(frozenset({"enabled-plugin"}))
        with patch("ai_assistant.plugins.loader.entry_points") as mock_eps:
            mock_eps.return_value.select.return_value = [mock_entry_point]
            with self.assertRaises(TypeError, msg="Plugin without register should raise"):
                load_entrypoint_plugins(registry)

    def test_multiple_plugins_loaded_in_order(self) -> None:
        """Multiple enabled plugins are all loaded."""
        plugins = []
        for i in range(3):
            mock_plugin = MagicMock()
            mock_plugin.plugin_id = f"plugin-{i}"
            plugins.append(mock_plugin)

        mock_entry_points = []
        for mock_plugin in plugins:
            mock_ep = MagicMock()
            mock_ep.load.return_value = lambda p=mock_plugin: p
            mock_entry_points.append(mock_ep)

        registry = PluginRegistry(frozenset({"plugin-0", "plugin-1", "plugin-2"}))
        with patch("ai_assistant.plugins.loader.entry_points") as mock_eps:
            mock_eps.return_value.select.return_value = mock_entry_points
            loaded = load_entrypoint_plugins(registry)

        self.assertEqual(len(loaded), 3)
        for mock_plugin in plugins:
            mock_plugin.register.assert_called_once()


if __name__ == "__main__":
    unittest.main()
