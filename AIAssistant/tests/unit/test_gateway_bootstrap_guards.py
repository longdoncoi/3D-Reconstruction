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
from ai_assistant.bootstrap.container import PlatformContainer, build_container, tool_spec_from
from ai_assistant.bootstrap.runtime import PlatformToolGateway
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


class BuildContainerTests(unittest.TestCase):
    def test_an_entry_without_a_spec_does_not_break_the_build(self):
        """A nameless entry resolves to no spec; the loop moves on."""
        with tempfile.TemporaryDirectory() as directory:
            settings = SimpleNamespace(
                allowed_plugins=frozenset({"builtin.legacy-tools"}),
                data_dir=Path(directory),
                agent_capabilities=frozenset({"chat"}),
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


if __name__ == "__main__":
    unittest.main()
