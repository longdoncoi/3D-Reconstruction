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
    def tearDown(self) -> None:
        server._chatbot_agent_holder.clear()

    def test_get_chatbot_agent_creates_once_and_rebuild_replaces(self) -> None:
        first, second = MagicMock(), MagicMock()
        with patch.object(server, "ChatbotAgent", side_effect=[first, second]) as mock_cls:
            got_a = server._get_chatbot_agent()
            got_b = server._get_chatbot_agent()
            server._rebuild_chatbot_agent()
            got_c = server._get_chatbot_agent()
            mock_cls.assert_called_with(server.llm_module, server.rag_module)
        self.assertIs(got_a, first)
        self.assertIs(got_b, first)
        self.assertIs(got_c, second)
        self.assertIsNot(got_a, got_c)


class A2ATaskExecutorTests(unittest.TestCase):
    def test_execute_a2a_task_requires_loaded_llm(self) -> None:
        with patch.object(server.llm_module, "llm", None):
            with self.assertRaisesRegex(RuntimeError, "LLM is not initialized"):
                server._execute_a2a_task(MagicMock())

    def test_execute_a2a_task_delegates_to_agent_run_service(self) -> None:
        task = MagicMock()
        with patch.object(server.llm_module, "llm", object()), \
                patch.object(server, "_agent_run_service") as mock_service:
            server._execute_a2a_task(task)
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

    def _run_lifespan(self, platform: object, mcp_server: object) -> str:
        async def drive() -> str:
            async with server.lifespan(MagicMock()):
                return "yielded"

        # lifespan reads the *module-level* platform/mcp_server globals, so the
        # doubles must be installed before the async run starts.
        with patch.object(server, "platform", platform), \
                patch.object(server, "mcp_server", mcp_server):
            return asyncio.run(drive())

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
        with patch.object(server, "set_remote_specialist_lookup") as mock_set, \
                patch.object(server, "build_a2a_router", return_value="router") as mock_build, \
                patch.object(server, "agent_bind_host", return_value=bind_host), \
                patch.object(server, "logger") as mock_logger:
            server._register_optional_protocols(
                app, platform, mcp, policy if policy is not None else TransportPolicy(),
            )
        return SimpleNamespace(app=app, mcp=mcp, platform=platform, mock_set=mock_set,
                               mock_build=mock_build, logger=mock_logger)

    def test_registers_a2a_and_mcp_when_enabled_and_available(self) -> None:
        run = self._run(enable_a2a=True, enable_mcp=True, mcp_available=True)
        run.mock_set.assert_called_once_with(server.resolve_remote_specialist)
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
        run.mock_set.assert_called_once_with(server.resolve_remote_specialist)
        run.mock_build.assert_called_once()
        run.app.include_router.assert_called_once()
        run.mcp.register_plugin_tools.assert_not_called()
        run.app.mount.assert_not_called()

    def test_disables_a2a_and_mcp_when_deployment_profile_says_so(self) -> None:
        run = self._run(enable_a2a=False, enable_mcp=False, mcp_available=True)
        run.mock_set.assert_called_once_with(None)
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
