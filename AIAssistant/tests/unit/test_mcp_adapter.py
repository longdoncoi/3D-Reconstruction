"""Unit tests for the MCP tool server adapter.

``ai_assistant.adapters.mcp`` degrades gracefully when the ``mcp`` package is
missing (it is absent from the offline CI dependency set), so every assertion
must hold in both environments. Tool registrations on the real FastMCP server
are exercised only when ``MCP_AVAILABLE``. The gateway is bound per server
instance at construction time — the adapter exposes no module-level locator.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from unittest import mock

from ai_assistant.adapters import mcp as mcp_module


class _FakeGateway:
    def __init__(self, result=None) -> None:
        self.result = result or {"success": True, "result": "done"}
        self.calls: list[tuple[str, dict]] = []

    def execute(self, tool_name: str, parameters: dict) -> dict:
        self.calls.append((tool_name, parameters))
        return self.result


class _FakePlugins:
    def __init__(self, specs) -> None:
        self._specs = specs

    def specs(self):
        return self._specs


class _Spec:
    def __init__(self, name, description="desc") -> None:
        self.name = name
        self.description = description


class _FakeServer:
    def __init__(self, existing: bool = False, has_tool_attr: bool = True) -> None:
        self.existing = existing
        self.has_tool_attr = has_tool_attr
        self.registered: list = []

    def get_tool(self, name):
        return "registered" if self.existing and name == "demo" else None

    def tool(self):
        def decorator(fn):
            self.registered.append(fn)
            return fn

        return decorator


class McpToolServerTest(unittest.TestCase):
    def test_no_module_level_service_locator(self) -> None:
        # Removing the old ``_tool_gateway``/``configure_tool_gateway`` globals
        # is part of the architecture boundary; the gateway must be injected
        # through the constructor only.
        self.assertFalse(any(name in mcp_module.__all__ for name in ("configure_tool_gateway", "_tool_gateway")))

    def test_dispatch_unconfigured_returns_error(self) -> None:
        server = mcp_module.McpToolServer()
        payload = json.loads(server._dispatch("read_file", {"path": "x"}))
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "runtime_unconfigured")

    def test_dispatch_forwards_to_gateway_and_filters_none(self) -> None:
        gateway = _FakeGateway({"success": True, "result": "works"})
        server = mcp_module.McpToolServer(gateway)
        payload = json.loads(
            server._dispatch(
                "read_file",
                {"path": "src/main.py", "start_line": None, "end_line": 5},
            )
        )
        self.assertTrue(payload["success"])
        self.assertEqual(gateway.calls, [("read_file", {"path": "src/main.py", "end_line": 5})])

    def test_dispatch_stays_bound_to_its_own_gateway(self) -> None:
        first = mcp_module.McpToolServer(_FakeGateway({"success": True, "result": "a"}))
        second = mcp_module.McpToolServer(None)
        self.assertTrue(json.loads(first._dispatch("read_file", {"path": "x"}))["success"])
        self.assertEqual(json.loads(second._dispatch("read_file", {"path": "x"}))["error_code"],
                         "runtime_unconfigured")

    def test_asgi_app_shape(self) -> None:
        server = mcp_module.McpToolServer()
        if mcp_module.MCP_AVAILABLE:
            self.assertTrue(server.available)
            self.assertIsNotNone(server.asgi_app())
        else:
            self.assertFalse(server.available)
            self.assertIsNone(server.asgi_app())

    def test_lifespan_yields_when_mcp_missing(self) -> None:
        async def _run() -> bool:
            with mock.patch.object(mcp_module, "MCP_AVAILABLE", False):
                server = mcp_module.McpToolServer()
                async with server.lifespan():
                    return True
            return False

        self.assertTrue(asyncio.run(_run()))

    def test_register_plugin_tools_is_noop_without_inputs(self) -> None:
        server = mcp_module.McpToolServer()
        self.assertEqual(server.register_plugin_tools(), 0)
        plugins = _FakePlugins([])
        self.assertEqual(server.register_plugin_tools(plugins=plugins), 0)

    def test_bind_plugin_tools_binds_dynamic_tools(self) -> None:
        server = _FakeServer()
        plugins = _FakePlugins([_Spec("demo"), _Spec("verify")])
        count = mcp_module.bind_plugin_tools(server, lambda name, params: "{}", plugins)
        self.assertEqual(count, 2)
        self.assertEqual(len(server.registered), 2)
        self.assertEqual(server.registered[0].__name__, "demo")

    def test_bind_plugin_tools_skips_existing_and_reserved_tools(self) -> None:
        server = _FakeServer(existing=True)
        plugins = _FakePlugins([_Spec("demo"), _Spec("read_file")])
        count = mcp_module.bind_plugin_tools(server, lambda name, params: "{}", plugins,
                                             reserved={"read_file"})
        self.assertEqual(count, 0)
        self.assertEqual(server.registered, [])

    def test_list_registered_tools_covers_builtins(self) -> None:
        server = mcp_module.McpToolServer()
        names = server.list_registered_tools()
        self.assertIn("read_file", names)
        self.assertIn("application_action", names)

    @unittest.skipUnless(mcp_module.MCP_AVAILABLE, "mcp package not installed")
    def test_registered_tools_dispatch_through_gateway(self) -> None:
        gateway = _FakeGateway({"success": True, "result": "file"})
        server = mcp_module.McpToolServer(gateway)
        result = json.loads(server.read_file("src/main.py"))
        self.assertTrue(result["success"])
        self.assertEqual(gateway.calls[0][0], "read_file")
        self.assertEqual(gateway.calls[0][1]["path"], "src/main.py")

    @unittest.skipUnless(mcp_module.MCP_AVAILABLE, "mcp package not installed")
    def test_lifespan_starts_real_session_manager(self) -> None:
        server = mcp_module.McpToolServer()

        async def _run() -> bool:
            async with server.lifespan():
                return True

        self.assertTrue(asyncio.run(_run()))


if __name__ == "__main__":
    unittest.main()
