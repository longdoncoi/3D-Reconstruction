"""Unit tests for the MCP tool server adapter.

``ai_assistant.adapters.mcp`` degrades gracefully when the ``mcp`` package is
missing (it is absent from the offline CI dependency set), so every assertion
must hold in both environments. Tool registrations on the real FastMCP server
are exercised only when ``MCP_AVAILABLE``.
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


class McpAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        # The module-level gateway is process-global; always reset it.
        mcp_module.configure_tool_gateway(None)

    def tearDown(self) -> None:
        mcp_module.configure_tool_gateway(None)

    def test_dispatch_unconfigured_returns_error(self) -> None:
        payload = json.loads(mcp_module._dispatch("read_file", {"path": "x"}))
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "runtime_unconfigured")

    def test_dispatch_forwards_to_gateway_and_filters_none(self) -> None:
        gateway = _FakeGateway({"success": True, "result": "works"})
        mcp_module.configure_tool_gateway(gateway)
        payload = json.loads(
            mcp_module._dispatch(
                "read_file",
                {"path": "src/main.py", "start_line": None, "end_line": 5},
            )
        )
        self.assertTrue(payload["success"])
        self.assertEqual(gateway.calls, [("read_file", {"path": "src/main.py", "end_line": 5})])

    def test_configure_tool_gateway_accepts_none_reset(self) -> None:
        mcp_module.configure_tool_gateway(_FakeGateway())
        mcp_module.configure_tool_gateway(None)
        payload = json.loads(mcp_module._dispatch("read_file", {"path": "x"}))
        self.assertEqual(payload["error_code"], "runtime_unconfigured")

    def test_asgi_app_shape(self) -> None:
        if mcp_module.MCP_AVAILABLE:
            self.assertIsNotNone(mcp_module.asgi_app())
        else:
            self.assertIsNone(mcp_module.asgi_app())

    def test_lifespan_yields_when_mcp_missing(self) -> None:
        async def _run() -> bool:
            with mock.patch.object(mcp_module, "mcp", None):
                async with mcp_module.lifespan():
                    return True
            return False

        self.assertTrue(asyncio.run(_run()))

    def test_register_plugin_tools_is_noop_without_inputs(self) -> None:
        self.assertEqual(mcp_module.register_plugin_tools(), 0)
        plugins = _FakePlugins([])
        self.assertEqual(mcp_module.register_plugin_tools(plugins=plugins), 0)

    def test_register_plugin_tools_binds_dynamic_tools(self) -> None:
        server = _FakeServer()
        plugins = _FakePlugins([_Spec("demo"), _Spec("verify")])
        count = mcp_module.register_plugin_tools(server=server, plugins=plugins)
        self.assertEqual(count, 2)
        self.assertEqual(len(server.registered), 2)
        self.assertEqual(server.registered[0].__name__, "demo")

    def test_register_plugin_tools_skips_existing_tool(self) -> None:
        server = _FakeServer(existing=True)
        plugins = _FakePlugins([_Spec("demo")])
        count = mcp_module.register_plugin_tools(server=server, plugins=plugins)
        self.assertEqual(count, 0)
        self.assertEqual(server.registered, [])

    @unittest.skipUnless(mcp_module.MCP_AVAILABLE, "mcp package not installed")
    def test_registered_tools_dispatch_through_gateway(self) -> None:
        gateway = _FakeGateway({"success": True, "result": "file"})
        mcp_module.configure_tool_gateway(gateway)
        result = json.loads(mcp_module.read_file("src/main.py"))
        self.assertTrue(result["success"])
        self.assertEqual(gateway.calls[0][0], "read_file")
        self.assertEqual(gateway.calls[0][1]["path"], "src/main.py")

    @unittest.skipUnless(mcp_module.MCP_AVAILABLE, "mcp package not installed")
    def test_lifespan_starts_real_session_manager(self) -> None:
        async def _run() -> bool:
            async with mcp_module.lifespan():
                return True

        self.assertTrue(asyncio.run(_run()))


if __name__ == "__main__":
    unittest.main()
