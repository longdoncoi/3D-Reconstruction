"""Unit tests for the composition-root helpers in ``bootstrap/server.py``.

The module is imported once at test-collection time (its module-level wiring is
pinned by :mod:`tests.integration.test_composition_root_bootstrap`). These tests
drive the *branches* that import-time execution cannot reach: the optional
A2A/MCP registration matrix, the lifespan lifecycle, the chatbot-agent holder,
win32 platform shims and the A2A task executor.
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

from ai_assistant.adapters.http.security import GuardedMountedApp, TransportPolicy
from ai_assistant.bootstrap import server


class _AsyncCM:
    """Trivial ``async with`` double recording enter/exit."""

    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *exc) -> bool:
        self.exited = True
        return False


class PackageRootBootstrapTests(unittest.TestCase):
    def test_ensure_package_root_adds_missing_root(self) -> None:
        fake_root = "fake-root-not-on-path"
        try:
            sys.path.remove(fake_root)
        except ValueError:
            pass
        try:
            server._ensure_package_root_on_path(fake_root)
            self.assertIn(fake_root, sys.path)
        finally:
            sys.path.remove(fake_root)

    def test_ensure_package_root_skips_existing_root(self) -> None:
        fake_root = "fake-root-on-path"
        sys.path.append(fake_root)
        try:
            server._ensure_package_root_on_path(fake_root)
            self.assertEqual(sys.path.count(fake_root), 1)
        finally:
            sys.path.remove(fake_root)


class ProactorShimTests(unittest.TestCase):
    def test_silence_connection_reset_calls_original(self) -> None:
        received: list = []

        class _FakeTransport:
            @staticmethod
            def _call_connection_lost(self_, exc):
                received.append(exc)

        server._silence_connection_reset(_FakeTransport)
        _FakeTransport._call_connection_lost("self", ValueError("other"))
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], ValueError)
        self.assertEqual(received[0].args, ("other",))

    def test_silence_connection_reset_swallows_reset(self) -> None:
        received: list = []

        class _FakeTransport:
            @staticmethod
            def _call_connection_lost(self_, exc):
                received.append(exc)
                raise ConnectionResetError("reset")

        server._silence_connection_reset(_FakeTransport)
        _FakeTransport._call_connection_lost("self", ConnectionResetError("x"))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].args, ("x",))
        # The reset was swallowed, so the wrapping returned normally.

    def test_patch_proactor_transport_is_noop_on_posix(self) -> None:
        with (
            patch.object(server.sys, "platform", "linux"),
            patch.object(server, "_silence_connection_reset") as silence,
        ):
            server._patch_proactor_transport()  # returns before importing asyncio internals
            # POSIX returns first: the proactor transport class is never touched.
            silence.assert_not_called()

    def test_patch_proactor_transport_installs_shim_on_windows(self) -> None:
        import asyncio.proactor_events as proactor_events

        original = lambda self_, exc: None  # noqa: E731
        fake_transport_cls = SimpleNamespace(_call_connection_lost=original)
        with patch.object(server.sys, "platform", "win32"), \
                patch.object(proactor_events, "_ProactorBasePipeTransport", fake_transport_cls):
            server._patch_proactor_transport()
        self.assertIsNot(fake_transport_cls._call_connection_lost, original)


class ChatbotAgentHolderTests(unittest.TestCase):
    def setUp(self) -> None:
        server.runtime_modules.chatbot_agent._agent = None

    def tearDown(self) -> None:
        server.runtime_modules.chatbot_agent._agent = None

    def test_get_chatbot_agent_creates_once_and_rebuild_replaces(self) -> None:
        holder = server.runtime_modules.chatbot_agent
        first, second = MagicMock(), MagicMock()
        with patch("ai_assistant.orchestration.specialists.ChatbotAgent", side_effect=[first, second]) as mock_cls:
            got_a = holder.get()
            got_b = holder.get()
            holder.rebuild()
            got_c = holder.get()
            mock_cls.assert_called_with(
                server.runtime_modules.llm,
                server.runtime_modules.rag,
                server.runtime_modules.chatbot_agent._prompt_builder,
            )
        self.assertIs(got_a, first)
        self.assertIs(got_b, first)
        self.assertIs(got_c, second)
        self.assertIsNot(got_a, got_c)


class A2ATaskExecutorTests(unittest.TestCase):
    def test_execute_a2a_task_requires_loaded_llm(self) -> None:
        with patch.object(server.runtime_modules.llm, "llm", None):
            with self.assertRaisesRegex(RuntimeError, "LLM is not initialized"):
                server.runtime_modules.execute_a2a_task(MagicMock())

    def test_execute_a2a_task_requires_the_run_service(self) -> None:
        with patch.object(server.runtime_modules.llm, "llm", object()), \
                patch.object(server.runtime_modules, "agent_run_service", None):
            with self.assertRaisesRegex(RuntimeError, "Agent run service is not initialized"):
                server.runtime_modules.execute_a2a_task(MagicMock())

    def test_execute_a2a_task_delegates_to_agent_run_service(self) -> None:
        task = MagicMock()
        mock_service = MagicMock()
        with patch.object(server.runtime_modules.llm, "llm", object()), \
                patch.object(server.runtime_modules, "agent_run_service", mock_service):
            server.runtime_modules.execute_a2a_task(task)
        mock_service.run.assert_called_once_with(task)


class AgentRouteRefreshTests(unittest.TestCase):
    def test_refresh_agent_routes_resets_state_and_schema(self) -> None:
        with patch.object(server, "agent_service") as mock_service:
            server.app.openapi_schema = "stale"
            server._refresh_agent_routes()
        mock_service.reset_state.assert_called_once()
        self.assertIsNone(server.app.openapi_schema)


class LifespanTests(unittest.TestCase):
    def setUp(self) -> None:
        self._logger_patch = patch.object(server, "logger")
        self.mock_logger = self._logger_patch.start()
        self.addCleanup(self._logger_patch.stop)

    def _run_lifespan(self, platform: object, mcp_server: object,
                      app: object | None = None) -> str:
        async def drive() -> str:
            async with server.lifespan(app if app is not None else MagicMock()):
                return "yielded"

        # lifespan reads the *module-level* platform/mcp_server globals, so the
        # doubles must be installed before the async run starts.
        with patch.object(server, "platform", platform), \
                patch.object(server, "mcp_server", mcp_server):
            return asyncio.run(drive())

    def test_lifespan_without_a_registry_skips_discovery(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace(a2a_registry=None))
        with patch("builtins.print"), \
                patch("ai_assistant.adapters.persistence.checkpointing.cleanup_old_checkpoints"), \
                patch.object(server, "task_retention_days", return_value=0), \
                patch.object(server, "_discover_a2a_peers") as discover:
            result = self._run_lifespan(
                SimpleNamespace(settings=SimpleNamespace(enable_mcp=False), tasks=MagicMock()),
                MagicMock(available=False),
                app=app,
            )
        self.assertEqual(result, "yielded")
        discover.assert_not_called()

    def test_lifespan_yields_when_mcp_disabled_and_logs_cleanup_error(self) -> None:
        with patch("builtins.print"), \
                patch(
                    "ai_assistant.adapters.persistence.checkpointing.cleanup_old_checkpoints",
                    side_effect=RuntimeError("boom"),
                ):
            result = self._run_lifespan(
                SimpleNamespace(
                    settings=SimpleNamespace(enable_mcp=False),
                    tasks=MagicMock(),
                ),
                MagicMock(available=True),
            )
        self.assertEqual(result, "yielded")
        self.mock_logger.warning.assert_called_once()
        self.assertIn("Lỗi dọn dẹp checkpoint", self.mock_logger.warning.call_args.args[0])
        self.mock_logger.info.assert_any_call("MCP endpoint disabled by configuration")

    def test_a2a_discovery_noops_without_configured_endpoints(self) -> None:
        registry = MagicMock()
        with patch("ai_assistant.settings.a2a_trusted_endpoints", return_value=frozenset()):
            asyncio.run(server._discover_a2a_peers(registry))
        registry.discover.assert_not_called()

    def test_a2a_discovery_fetches_the_allowlist_in_sorted_order(self) -> None:
        registry = MagicMock()
        registry.discover.return_value = {"https://a.test": object()}
        with patch(
            "ai_assistant.settings.a2a_trusted_endpoints",
            return_value=frozenset({"https://b.test", "https://a.test"}),
        ):
            asyncio.run(server._discover_a2a_peers(registry))
        registry.discover.assert_called_once_with(["https://a.test", "https://b.test"])
        self.mock_logger.info.assert_any_call(
            "A2A discovery: %d/%d endpoint(s) reachable", 1, 2,
        )

    def test_lifespan_logs_a_discovery_failure_without_blocking_startup(self) -> None:
        async def boom(_registry: object) -> None:
            msg = "discovery boom"
            raise RuntimeError(msg)

        with patch("builtins.print"), \
                patch("ai_assistant.adapters.persistence.checkpointing.cleanup_old_checkpoints"), \
                patch.object(server, "_discover_a2a_peers", side_effect=boom):
            result = self._run_lifespan(
                SimpleNamespace(settings=SimpleNamespace(enable_mcp=False), tasks=MagicMock()),
                MagicMock(available=False),
            )
        self.assertEqual(result, "yielded")
        self.mock_logger.warning.assert_any_call("A2A discovery failed: discovery boom")

    def test_lifespan_uses_mcp_lifespan_when_available(self) -> None:
        settings = SimpleNamespace(enable_mcp=True)
        tasks = MagicMock()
        mcp = MagicMock(available=True)
        cm = _AsyncCM()
        mcp.lifespan.return_value = cm
        with patch("builtins.print"), \
                patch("ai_assistant.adapters.persistence.checkpointing.cleanup_old_checkpoints"):
            result = self._run_lifespan(SimpleNamespace(settings=settings, tasks=tasks), mcp)
        self.assertEqual(result, "yielded")
        self.assertTrue(cm.entered)
        self.assertTrue(cm.exited)
        tasks.close.assert_called_once()

    def test_lifespan_logs_a_pruning_failure_without_blocking_startup(self) -> None:
        tasks = MagicMock()
        tasks.prune.side_effect = RuntimeError("prune boom")
        with patch("builtins.print"), \
                patch("ai_assistant.adapters.persistence.checkpointing.cleanup_old_checkpoints"), \
                patch.object(server, "task_retention_days", return_value=30):
            result = self._run_lifespan(
                SimpleNamespace(settings=SimpleNamespace(enable_mcp=False), tasks=tasks),
                MagicMock(available=False),
            )
        self.assertEqual(result, "yielded")
        self.mock_logger.warning.assert_any_call("Task retention pruning failed: prune boom")

    def test_lifespan_skips_pruning_when_the_window_is_disabled(self) -> None:
        tasks = MagicMock()
        with patch("builtins.print"), \
                patch("ai_assistant.adapters.persistence.checkpointing.cleanup_old_checkpoints"), \
                patch.object(server, "task_retention_days", return_value=0):
            self._run_lifespan(
                SimpleNamespace(settings=SimpleNamespace(enable_mcp=False), tasks=tasks),
                MagicMock(available=False),
            )
        tasks.prune.assert_not_called()

    def test_lifespan_does_not_log_when_no_task_was_pruned(self) -> None:
        tasks = MagicMock()
        tasks.prune.return_value = 0
        with patch("builtins.print"), \
                patch("ai_assistant.adapters.persistence.checkpointing.cleanup_old_checkpoints"), \
                patch.object(server, "task_retention_days", return_value=30):
            self._run_lifespan(
                SimpleNamespace(settings=SimpleNamespace(enable_mcp=False), tasks=tasks),
                MagicMock(available=False),
            )
        self.assertFalse(
            [call for call in self.mock_logger.info.call_args_list if "Pruned" in str(call)],
        )

    def test_lifespan_warns_when_mcp_unavailable(self) -> None:
        tasks = MagicMock()
        with patch("builtins.print"), \
                patch("ai_assistant.adapters.persistence.checkpointing.cleanup_old_checkpoints"):
            result = self._run_lifespan(
                SimpleNamespace(
                    settings=SimpleNamespace(enable_mcp=True),
                    tasks=tasks,
                ),
                MagicMock(available=False),
            )
        self.assertEqual(result, "yielded")
        self.mock_logger.warning.assert_called_once()
        self.assertIn("MCP SDK is not installed", self.mock_logger.warning.call_args.args[0])
        tasks.close.assert_called_once()


class OptionalProtocolTests(unittest.TestCase):
    def _run(self, *, enable_a2a: bool, enable_mcp: bool, mcp_available: bool,
             bind_host: str = "127.0.0.1", policy: TransportPolicy | None = None):
        app = MagicMock()
        mcp = MagicMock()
        mcp.available = mcp_available
        platform = SimpleNamespace(
            settings=SimpleNamespace(enable_a2a=enable_a2a, enable_mcp=enable_mcp),
            tasks=object(),
            plugins=object(),
        )
        with patch("ai_assistant.adapters.a2a.build_a2a_router", return_value="router") as mock_build, \
                patch.object(server, "agent_bind_host", return_value=bind_host), \
                patch.object(server, "logger") as mock_logger:
            server._register_optional_protocols(
                app, platform, mcp, policy if policy is not None else TransportPolicy(),
            )
        return SimpleNamespace(app=app, mcp=mcp, platform=platform,
                               mock_build=mock_build, logger=mock_logger)

    def test_registers_a2a_and_mcp_when_enabled_and_available(self) -> None:
        run = self._run(enable_a2a=True, enable_mcp=True, mcp_available=True)
        run.mock_build.assert_called_once_with(
            run.platform.tasks,
            "3D-Reconstruction AI Assistant",
            "3.0.0",
            policy=ANY,
        )
        self.assertIsInstance(run.mock_build.call_args.kwargs["policy"], TransportPolicy)
        run.app.include_router.assert_called_once_with("router")
        run.mcp.register_plugin_tools.assert_called_once_with(run.platform.plugins)
        # The sub-app is mounted behind the transport guard: ``mount`` cannot
        # take route dependencies, so the wrapper carries the MCP policy.
        run.app.mount.assert_called_once()
        path, mounted = run.app.mount.call_args.args
        self.assertEqual(path, "/mcp")
        self.assertIsInstance(mounted, GuardedMountedApp)
        run.logger.warning.assert_not_called()

    def test_skips_mcp_when_sdk_unavailable(self) -> None:
        run = self._run(enable_a2a=True, enable_mcp=True, mcp_available=False)
        run.mock_build.assert_called_once()
        run.app.include_router.assert_called_once()
        run.mcp.register_plugin_tools.assert_not_called()
        run.app.mount.assert_not_called()

    def test_disables_a2a_and_mcp_when_deployment_profile_says_so(self) -> None:
        run = self._run(enable_a2a=False, enable_mcp=False, mcp_available=True)
        run.mock_build.assert_not_called()
        run.app.include_router.assert_not_called()
        run.mcp.register_plugin_tools.assert_not_called()
        run.app.mount.assert_not_called()

    def test_exposed_bind_without_a_token_keeps_mcp_unmounted(self) -> None:
        # Docker publishes AGENT_HOST=0.0.0.0; without AI_MCP_TOKEN there is
        # nothing for the guard to authenticate, so the surface stays closed
        # and the startup log says why instead of serving anonymous 403s.
        run = self._run(enable_a2a=True, enable_mcp=True, mcp_available=True,
                        bind_host="0.0.0.0")
        run.app.mount.assert_not_called()
        run.mcp.register_plugin_tools.assert_not_called()
        run.logger.warning.assert_called_once()
        self.assertIn("AI_MCP_TOKEN", run.logger.warning.call_args.args[0])

    def test_exposed_bind_with_a_token_still_mounts_mcp(self) -> None:
        run = self._run(enable_a2a=True, enable_mcp=True, mcp_available=True,
                        bind_host="0.0.0.0", policy=TransportPolicy(mcp_token="mcp-secret"))
        run.app.mount.assert_called_once()
        run.logger.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
