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


class BuiltinWrapperDispatchTests(unittest.TestCase):
    """Every builtin wrapper must forward its exact payload to the gateway.

    Only ``read_file`` was reached by the original suite, leaving the other
    nine wrappers — and the parameter shape they hand to the gateway —
    unverified. These are plain methods, so they need no ``mcp`` package.
    """

    def setUp(self) -> None:
        self.gateway = _FakeGateway({"success": True, "result": "ok"})
        self.server = mcp_module.McpToolServer(self.gateway)

    def _call(self, name: str, *args, **kwargs) -> dict:
        payload = json.loads(getattr(self.server, name)(*args, **kwargs))
        self.assertTrue(payload["success"])
        self.assertEqual(self.gateway.calls[-1][0], name)
        return self.gateway.calls[-1][1]

    def test_list_directory_forwards_its_depth_controls(self) -> None:
        params = self._call("list_directory", "src", True, 7)
        self.assertEqual(params, {"path": "src", "recursive": True, "max_depth": 7})

    def test_find_files_forwards_its_glob_and_cap(self) -> None:
        params = self._call("find_files", "*.py", "src", 42)
        self.assertEqual(params, {"pattern": "*.py", "path": "src", "max_results": 42})

    def test_search_text_forwards_every_filter(self) -> None:
        params = self._call("search_text", "query", "src", "*.py", True, 5)
        self.assertEqual(params, {
            "query": "query", "path": "src", "file_pattern": "*.py",
            "case_sensitive": True, "max_results": 5,
        })

    def test_analyze_code_forwards_the_path(self) -> None:
        self.assertEqual(self._call("analyze_code", "src/main.py"), {"path": "src/main.py"})

    def test_get_project_status_dispatches_an_empty_request(self) -> None:
        self.assertEqual(self._call("get_project_status"), {})

    def test_git_diff_forwards_its_refinement_flags(self) -> None:
        params = self._call("git_diff", "src", True)
        self.assertEqual(params, {"path": "src", "staged": True})

    def test_validate_file_forwards_the_path(self) -> None:
        self.assertEqual(self._call("validate_file", "config.json"), {"path": "config.json"})

    def test_rag_search_forwards_its_result_limit(self) -> None:
        params = self._call("rag_search", "how to build", 3)
        self.assertEqual(params, {"query": "how to build", "top_k": 3})

    def test_application_action_forwards_the_desktop_credentials(self) -> None:
        params = self._call("application_action", "login", "vi", "user", "secret")
        self.assertEqual(params, {
            "action": "login", "language": "vi",
            "username": "user", "password": "secret",
        })


class BindPluginToolsEdgeTests(unittest.TestCase):
    """The two guard arms of the dynamic-tool binder the happy path skips."""

    def test_a_server_without_the_tool_decorator_registers_nothing(self) -> None:
        class _Plain:
            def get_tool(self, _name):
                return None

        count = mcp_module.bind_plugin_tools(
            _Plain(), lambda _name, _params: "{}", _FakePlugins([_Spec("demo")]),
        )
        self.assertEqual(count, 0)

    def test_a_registration_failure_is_swallowed_and_stops_the_loop(self) -> None:
        class _Broken:
            def get_tool(self, _name):
                return None

            def tool(self):
                msg = "server shutting down"
                raise RuntimeError(msg)

        count = mcp_module.bind_plugin_tools(
            _Broken(), lambda _name, _params: "{}", _FakePlugins([_Spec("demo")]),
        )
        self.assertEqual(count, 0)

    def test_a_registered_plugin_tool_dispatches_when_invoked(self) -> None:
        """The generated closure is what actually reaches the gateway."""
        server = _FakeServer()
        seen: list[tuple[str, dict]] = []

        def _dispatch(name, params):
            seen.append((name, params))
            return "{}"

        mcp_module.bind_plugin_tools(server, _dispatch, _FakePlugins([_Spec("demo")]))
        server.registered[0](x=1, y="two")
        self.assertEqual(seen, [("demo", {"x": 1, "y": "two"})])


class McpAbsentFallbackTests(unittest.TestCase):
    """Pin the branch taken when *neither* mcp import location exists.

    ``mcp.server.fastmcp`` is already missing in this environment (the outer
    ``except`` runs), but ``mcp.server.mcpserver`` still resolves, so the
    innermost fallback is only reached on an offline install with no ``mcp``
    at all. Reloading with every ``mcp`` entry masked as ``None`` reproduces
    exactly that, and the ``finally`` restores the real module so no other
    test inherits a degraded adapter.
    """

    def test_both_import_failures_leave_the_adapter_unavailable(self) -> None:
        import importlib
        import sys

        saved = {
            name: module for name, module in sys.modules.items()
            if name == "mcp" or name.startswith("mcp.")
        }
        available_before = mcp_module.MCP_AVAILABLE
        try:
            for name in saved:
                sys.modules[name] = None  # None in sys.modules makes `import` raise
            reloaded = importlib.reload(mcp_module)
            self.assertFalse(reloaded.MCP_AVAILABLE)
            self.assertIsNone(reloaded.FastMCP)
        finally:
            for name in saved:
                sys.modules.pop(name, None)
            sys.modules.update(saved)
            importlib.reload(mcp_module)
        # The adapter must come back exactly as it was for every later test.
        self.assertEqual(mcp_module.MCP_AVAILABLE, available_before)


if __name__ == "__main__":
    unittest.main()
