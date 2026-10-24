"""Guard-path coverage for the gateway adapter and the DI container bridge.

An unconfigured ``PlatformToolGateway`` (pre-bootstrap) must answer every
operation with a structured fallback instead of raising: the approved-path
execute, the grant issue (a silent no-op) and the grant probe (``False``).
``tool_spec_from`` skips entries without a name or without an executor, and
``build_container`` tolerates an entry that resolves to no spec — the loop
simply moves on to the next catalog entry.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai_assistant.adapters.persistence.storage_crypto import AesGcmFieldCipher
from ai_assistant.application.coordination import TaskCoordinator
from ai_assistant.bootstrap.container import (
    PlatformContainer,
    _required_scope,
    build_container,
    tool_spec_from,
)
from ai_assistant.bootstrap.runtime import PlatformToolGateway
from ai_assistant.domain.security import DataClassification
from ai_assistant.domain.tools import SideEffect, ToolSpec
from ai_assistant.tools.definition import ToolDefinition
from ai_assistant.tools.tool_contract import RUNTIME_UNCONFIGURED_RESULT


class UnconfiguredGatewayTests(unittest.TestCase):
    def test_approved_execute_reports_unconfigured(self):
        gateway = PlatformToolGateway()
        self.assertEqual(gateway.execute_approved("read_file", {}), dict(RUNTIME_UNCONFIGURED_RESULT))

    def test_plain_execute_reports_unconfigured(self):
        gateway = PlatformToolGateway()
        self.assertEqual(gateway.execute("read_file", {}), dict(RUNTIME_UNCONFIGURED_RESULT))

    def test_grant_issue_is_a_silent_noop(self):
        self.assertIsNone(PlatformToolGateway().issue_approval_grant("t", {}, "tok"))

    def test_grant_probe_is_false(self):
        self.assertFalse(PlatformToolGateway().approval_covers("t", {}, "tok"))


class ToolSpecBridgeTests(unittest.TestCase):
    def test_an_entry_without_a_name_is_skipped(self):
        self.assertIsNone(tool_spec_from(SimpleNamespace(), MagicMock()))

    def test_an_entry_without_an_executor_is_skipped(self):
        self.assertIsNone(tool_spec_from(SimpleNamespace(name="read_file"), None))

    def test_a_complete_entry_bridges_to_a_spec(self):
        entry = SimpleNamespace(name="read_file", description="read",
                                json_schema={"type": "object"}, policy="read_only")
        spec = tool_spec_from(entry, lambda params: {})
        self.assertIsNotNone(spec)
        self.assertEqual(spec.name, "read_file")

    def test_the_ceiling_comes_from_the_tool_contract(self):
        """The tool's data ceiling is the tool's own, not a constant.

        Every tool used to carry the same ``restricted`` ceiling, so the
        gateway's classification gate compared every principal against one
        value — a read-only tool and one that rewrites the project were
        indistinguishable to it.
        """
        entry = SimpleNamespace(name="read_file", description="read",
                                json_schema={"type": "object"}, policy="read_only",
                                classification="internal")
        spec = tool_spec_from(entry, lambda params: {})
        self.assertEqual(spec.maximum_classification, DataClassification.INTERNAL)

    def test_a_mutating_tool_carries_the_restricted_ceiling(self):
        entry = SimpleNamespace(name="write_file", description="write",
                                json_schema={"type": "object"}, policy="code_write",
                                classification="restricted")
        spec = tool_spec_from(entry, lambda params: {})
        self.assertEqual(spec.maximum_classification, DataClassification.RESTRICTED)

    def test_an_unknown_classification_label_is_the_most_protective(self):
        """A label nobody recognises must not lower a tool's ceiling."""
        entry = SimpleNamespace(name="mystery", description="", json_schema={},
                                policy="read_only", classification="nonsense")
        spec = tool_spec_from(entry, lambda params: {})
        self.assertEqual(spec.maximum_classification, DataClassification.REGULATED)

    def test_an_entry_without_a_classification_is_the_most_protective(self):
        entry = SimpleNamespace(name="mystery", description="", json_schema={},
                                policy="read_only")
        spec = tool_spec_from(entry, lambda params: {})
        self.assertEqual(spec.maximum_classification, DataClassification.REGULATED)

    def test_an_unknown_policy_is_refused_at_bootstrap(self):
        """A misspelled policy must not degrade into a read-only tool."""
        entry = SimpleNamespace(name="mystery", description="", json_schema={},
                                policy="write_typo")
        with self.assertRaises(ValueError):
            tool_spec_from(entry, lambda params: {})
        with self.assertRaises(ValueError):
            _required_scope("write_typo")

    def test_an_entry_without_a_policy_is_refused_at_bootstrap(self):
        """A missing label is a policy gap, not an implicit read-only tool.

        ``ToolDefinition`` declares a default, so this arm only fires for a
        hand-built dict/namespace whose author never chose a policy. Assuming
        read-only there would let a mutating tool inherit the weakest scope.
        """
        entry = SimpleNamespace(name="mystery", description="", json_schema={})
        with self.assertRaisesRegex(ValueError, "no execution policy"):
            tool_spec_from(entry, lambda params: {})

    def test_an_alias_is_refused_as_an_execution_spec(self):
        """The friendly spellings must not become separate weak policies."""
        entry = SimpleNamespace(
            name="app_action_viewer", description="", json_schema={},
            policy="read_only", classification="internal",
        )
        with self.assertRaises(ValueError):
            tool_spec_from(entry, lambda params: {})

    def test_a_plugin_spec_bridges_verbatim(self):
        spec = ToolSpec(
            name="plugin_tool", description="from a plugin", input_schema={},
            timeout_seconds=3, side_effect=SideEffect.WRITE,
            requires_approval=True, required_scope="project.write",
            maximum_classification=DataClassification.RESTRICTED,
            plugin_id="acme",
        )
        entry = SimpleNamespace(name="plugin_tool", spec=spec)
        self.assertIs(tool_spec_from(entry, lambda params: {}), spec)


class BuildContainerTests(unittest.TestCase):
    def test_an_entry_without_a_spec_does_not_break_the_build(self):
        """A nameless entry resolves to no spec; the loop moves on."""
        with tempfile.TemporaryDirectory() as directory:
            settings = SimpleNamespace(
                allowed_plugins=frozenset({"builtin.legacy-tools"}),
                data_dir=Path(directory),
                agent_capabilities=frozenset({"chat"}),
                max_concurrent_tasks=2,
            )
            nameless = SimpleNamespace(handler=lambda params: {})
            with patch(
                "ai_assistant.bootstrap.container.ArchitectureSettings.load",
                return_value=settings,
            ):
                container = build_container(
                    Path(directory), [nameless], {},
                    task_executor=lambda task: {},
                    coordinator=TaskCoordinator(),
                )
            try:
                self.assertIsNotNone(container.gateway)
            finally:
                # Release the SQLite handle before the tmp dir is removed.
                container.tasks._store.close()



# ── Container build paths (moved from test_coverage_boost) ────────

class BootstrapContainerTests(unittest.TestCase):
    """Test build_container coverage."""

    def test_build_container_returns_platform_container(self) -> None:
        """build_container returns a PlatformContainer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            container = build_container(
                Path(tmpdir),
                [],
                {},
                MagicMock(),
            )
            self.assertIsInstance(container, PlatformContainer)
            self.assertIsNotNone(container.settings)
            self.assertIsNotNone(container.plugins)
            self.assertIsNotNone(container.tools)
            self.assertIsNotNone(container.tasks)
            self.assertIsNotNone(container.gateway)
            self.assertIsNone(container.storage_cipher)

    def test_build_container_injects_the_storage_cipher(self) -> None:
        cipher = AesGcmFieldCipher(b"z" * 32)
        with tempfile.TemporaryDirectory() as tmpdir:
            container = build_container(
                Path(tmpdir),
                [],
                {},
                MagicMock(),
                storage_cipher=cipher,
            )
            try:
                self.assertIs(container.storage_cipher, cipher)
                self.assertIs(container.tasks._store._cipher, cipher)
            finally:
                container.tasks.close()

    def test_build_container_with_legacy_tools(self) -> None:
        """build_container handles legacy tool dicts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_tools = [
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "policy": "read_only",
                    "schema": {"type": "object"},
                }
            ]
            container = build_container(
                Path(tmpdir),
                legacy_tools,
                {"test_tool": lambda params: {"success": True}},
                MagicMock(),
            )
            self.assertIsInstance(container, PlatformContainer)

    def test_build_container_registers_the_canonical_tool_not_the_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = [
                ToolDefinition(name="application_action", description="canonical", parameters={}),
                ToolDefinition(name="app_action_viewer", description="alias", parameters={}),
            ]
            def executor(params):
                return {"success": True}

            container = build_container(
                Path(tmpdir), tools,
                {tool.name: executor for tool in tools}, MagicMock(),
            )
            try:
                names = {spec.name for spec in container.plugins.specs()}
                self.assertIn("application_action", names)
                self.assertNotIn("app_action_viewer", names)
            finally:
                container.tasks.close()


if __name__ == "__main__":
    unittest.main()
