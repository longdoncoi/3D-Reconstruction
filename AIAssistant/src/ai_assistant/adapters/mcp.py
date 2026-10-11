"""MCP tool server for the 3D-Reconstruction desktop agent.

The composition root injects the shared tool gateway once, at construction
time (:class:`McpToolServer`), and every tool call funnels through the
instance-bound :meth:`McpToolServer._dispatch`. There is no module-level
service locator and no later ``configure_*`` rebind — the dispatch callback is
captured when the server is built.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Callable

from ..ports import ToolGateway

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:
        FastMCP = None  # type: ignore[misc,assignment]

MCP_AVAILABLE = FastMCP is not None

# Builtin tools, in registration order. The method names mirror the exposed
# MCP tool names so ``mcp.tool()(getattr(self, name))`` maps 1:1.
_BUILTIN_TOOLS = (
    "read_file",
    "list_directory",
    "find_files",
    "search_text",
    "analyze_code",
    "get_project_status",
    "git_diff",
    "validate_file",
    "rag_search",
    "application_action",
)


class McpToolServer:
    """FastMCP server bound to one injected tool gateway.

    Degrades to a no-op envelope when MCP is not installed (``available`` is
    False and ``asgi_app()`` returns ``None``), so the composition root and the
    offline test suite can import this module unconditionally.
    """

    def __init__(self, gateway: ToolGateway | None = None) -> None:
        self._gateway = gateway
        self._server: Any = None
        self._asgi_app: Any = None
        if MCP_AVAILABLE:
            self._server = FastMCP(
                "3D-Reconstruction Tools",
                instructions=(
                    "Safe tools for inspecting the 3D-Reconstruction project and dispatching "
                    "a desktop action. Desktop actions are only complete after the Qt client acknowledges them."
                ),
            )
            for name in _BUILTIN_TOOLS:
                self._server.tool()(getattr(self, name))
            self._asgi_app = self._server.streamable_http_app(
                streamable_http_path="/", stateless_http=True, json_response=True,
            )

    @property
    def available(self) -> bool:
        """True when a real MCP server could be constructed."""
        return self._server is not None

    def asgi_app(self):
        """Return the Streamable HTTP application, or None when MCP is not installed."""
        return self._asgi_app

    def _dispatch(self, tool_name: str, parameters: dict[str, Any]) -> str:
        """Adapt an MCP call to the platform's single policy-enforced tool path."""
        parameters = {name: value for name, value in parameters.items() if value is not None}
        if self._gateway is None:
            return json.dumps(
                {"success": False, "error_code": "runtime_unconfigured",
                 "error": "AI Agent Platform is not bootstrapped"},
                ensure_ascii=False,
            )
        return json.dumps(self._gateway.execute(tool_name, parameters), ensure_ascii=False)

    @asynccontextmanager
    async def lifespan(self):
        """Start/stop the MCP session manager when mounted inside FastAPI."""
        if self._server is None:
            yield
            return
        async with self._server.session_manager.run():
            yield

    def register_plugin_tools(self, plugins: Any = None) -> int:
        """Dynamically bind tools from an injected PluginRegistry into MCP.

        ``plugins`` is supplied by the composition root (``platform.plugins``);
        without it this is a no-op rather than reaching for a global container.
        Builtin names are reserved so plugin specs that mirror the builtin
        catalog never get re-registered (the ``MCPServer`` fallback does not
        expose ``get_tool``).
        """
        return bind_plugin_tools(self._server, self._dispatch, plugins,
                                 reserved=set(_BUILTIN_TOOLS))

    def list_registered_tools(self) -> list[str]:
        """Names of builtin tools this server exposes (for health/debugging)."""
        return list(_BUILTIN_TOOLS)

    # ── Builtin tools ─────────────────────────────────────────────────────────
    # Each is defined as a method so the same callable is both registered on
    # the server and directly testable.
    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        symbol: str | None = None,
    ) -> str:
        """Read a focused range or named source symbol within the project."""
        return self._dispatch("read_file", {
            "path": path, "start_line": start_line, "end_line": end_line, "symbol": symbol,
        })

    def list_directory(self, path: str, recursive: bool | None = None, max_depth: int | None = None) -> str:
        """List files and folders within the project directory."""
        return self._dispatch("list_directory", {"path": path, "recursive": recursive, "max_depth": max_depth})

    def find_files(self, pattern: str, path: str | None = None, max_results: int | None = None) -> str:
        """Find project files by glob without reading their contents."""
        return self._dispatch("find_files", {"pattern": pattern, "path": path, "max_results": max_results})

    def search_text(
        self,
        query: str,
        path: str | None = None,
        file_pattern: str | None = None,
        case_sensitive: bool | None = None,
        max_results: int | None = None,
    ) -> str:
        """Search project text and return matching file paths and line numbers."""
        return self._dispatch("search_text", {
            "query": query, "path": path, "file_pattern": file_pattern,
            "case_sensitive": case_sensitive, "max_results": max_results,
        })

    def analyze_code(self, path: str) -> str:
        """Summarise the imports, classes, and functions in a source file."""
        return self._dispatch("analyze_code", {"path": path})

    def get_project_status(self) -> str:
        """Get the current Git branch, modified files, and source summary."""
        return self._dispatch("get_project_status", {})

    def git_diff(self, path: str | None = None, staged: bool | None = None) -> str:
        """Read the current Git diff for Code Agent review."""
        return self._dispatch("git_diff", {"path": path, "staged": staged})

    def validate_file(self, path: str) -> str:
        """Syntax-check a Python file or parse a JSON file without changing it."""
        return self._dispatch("validate_file", {"path": path})

    def rag_search(self, query: str, top_k: int | None = None) -> str:
        """Search indexed project documentation and source code."""
        return self._dispatch("rag_search", {"query": query, "top_k": top_k})

    def application_action(
        self,
        action: str,
        language: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> str:
        """Request a canonical desktop action; the Qt desktop client must acknowledge it."""
        return self._dispatch("application_action", {
            "action": action, "language": language,
            "username": username, "password": password,
        })


def bind_plugin_tools(
    server: Any,
    dispatch: Callable[[str, dict[str, Any]], str],
    plugins: Any = None,
    *,
    reserved: set[str] | None = None,
) -> int:
    """Bind dynamic tools from a PluginRegistry through the given dispatch.

    ``dispatch`` is the instance-bound gateway dispatcher of the owning server;
    the composition root never reaches into this module for a global gateway.
    ``reserved`` holds names that must never be shadowed (the builtin catalog).
    """
    if server is None or plugins is None:
        return 0
    reserved = reserved or set()
    registered = 0
    try:
        for spec in plugins.specs():
            tool_name = spec.name
            if tool_name in reserved:
                continue
            if hasattr(server, "get_tool") and server.get_tool(tool_name):
                continue

            def _make_tool(name: str, desc: str):
                def _tool(**kwargs: Any) -> str:
                    return dispatch(name, kwargs)

                _tool.__name__ = name
                _tool.__doc__ = desc
                return _tool

            if hasattr(server, "tool"):
                server.tool()(_make_tool(tool_name, spec.description or f"Dynamic tool {tool_name}"))
                registered += 1
    except Exception:
        pass
    return registered


__all__ = [
    "MCP_AVAILABLE",
    "McpToolServer",
    "bind_plugin_tools",
]
