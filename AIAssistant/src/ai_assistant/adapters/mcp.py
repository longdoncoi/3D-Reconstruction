"""MCP tool server for the 3D-Reconstruction desktop agent."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from ..ports import ToolGateway

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:
        FastMCP = None  # type: ignore[misc,assignment]

MCP_AVAILABLE = FastMCP is not None

_tool_gateway: ToolGateway | None = None


def configure_tool_gateway(gateway: ToolGateway) -> None:
    """Inject the shared tool gateway into the MCP adapter (composition root)."""
    global _tool_gateway
    _tool_gateway = gateway


def _dispatch(tool_name: str, parameters: dict[str, Any]) -> str:
    """Adapt an MCP call to the platform's single policy-enforced tool path."""
    parameters = {name: value for name, value in parameters.items() if value is not None}
    if _tool_gateway is None:
        return json.dumps(
            {"success": False, "error_code": "runtime_unconfigured",
             "error": "AI Agent Platform is not bootstrapped"},
            ensure_ascii=False,
        )
    return json.dumps(_tool_gateway.execute(tool_name, parameters), ensure_ascii=False)


mcp = None
_asgi_app = None

if MCP_AVAILABLE:
    mcp = FastMCP(
        "3D-Reconstruction Tools",
        instructions=(
            "Safe tools for inspecting the 3D-Reconstruction project and dispatching "
            "a desktop action. Desktop actions are only complete after the Qt client acknowledges them."
        ),
    )

    @mcp.tool()
    def read_file(
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        symbol: str | None = None,
    ) -> str:
        """Read a focused range or named source symbol within the project."""
        return _dispatch("read_file", {
            "path": path, "start_line": start_line, "end_line": end_line, "symbol": symbol,
        })

    @mcp.tool()
    def list_directory(path: str, recursive: bool | None = None, max_depth: int | None = None) -> str:
        """List files and folders within the project directory."""
        return _dispatch("list_directory", {"path": path, "recursive": recursive, "max_depth": max_depth})

    @mcp.tool()
    def find_files(pattern: str, path: str | None = None, max_results: int | None = None) -> str:
        """Find project files by glob without reading their contents."""
        return _dispatch("find_files", {"pattern": pattern, "path": path, "max_results": max_results})

    @mcp.tool()
    def search_text(
        query: str,
        path: str | None = None,
        file_pattern: str | None = None,
        case_sensitive: bool | None = None,
        max_results: int | None = None,
    ) -> str:
        """Search project text and return matching file paths and line numbers."""
        return _dispatch("search_text", {
            "query": query, "path": path, "file_pattern": file_pattern,
            "case_sensitive": case_sensitive, "max_results": max_results,
        })

    @mcp.tool()
    def analyze_code(path: str) -> str:
        """Summarise the imports, classes, and functions in a source file."""
        return _dispatch("analyze_code", {"path": path})

    @mcp.tool()
    def get_project_status() -> str:
        """Get the current Git branch, modified files, and source summary."""
        return _dispatch("get_project_status", {})

    @mcp.tool()
    def git_diff(path: str | None = None, staged: bool | None = None) -> str:
        """Read the current Git diff for Code Agent review."""
        return _dispatch("git_diff", {"path": path, "staged": staged})

    @mcp.tool()
    def validate_file(path: str) -> str:
        """Syntax-check a Python file or parse a JSON file without changing it."""
        return _dispatch("validate_file", {"path": path})

    @mcp.tool()
    def rag_search(query: str, top_k: int | None = None) -> str:
        """Search indexed project documentation and source code."""
        return _dispatch("rag_search", {"query": query, "top_k": top_k})

    @mcp.tool()
    def application_action(
        action: str,
        language: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> str:
        """Request a canonical desktop action; the Qt desktop client must acknowledge it."""
        return _dispatch("application_action", {
            "action": action, "language": language,
            "username": username, "password": password,
        })

    _asgi_app = mcp.streamable_http_app(
        streamable_http_path="/", stateless_http=True, json_response=True,
    )


def asgi_app():
    """Return the Streamable HTTP application, or None when MCP is not installed."""
    return _asgi_app


@asynccontextmanager
async def lifespan():
    """Start/stop the MCP session manager when mounted inside FastAPI."""
    if mcp is None:
        yield
        return
    async with mcp.session_manager.run():
        yield


def register_plugin_tools(server: Any = None, plugins: Any = None) -> int:
    """Dynamically bind tools from an injected PluginRegistry into MCP.

    ``plugins`` is supplied by the composition root (``platform.plugins``);
    without it this is a no-op rather than reaching for a global container.
    """
    target_server = server or mcp
    if target_server is None or plugins is None:
        return 0
    registered = 0
    try:
        for spec in plugins.specs():
            tool_name = spec.name
            if hasattr(target_server, "get_tool") and target_server.get_tool(tool_name):
                continue

            def _make_tool(name: str, desc: str):
                def _tool(**kwargs: Any) -> str:
                    return _dispatch(name, kwargs)

                _tool.__name__ = name
                _tool.__doc__ = desc
                return _tool

            if hasattr(target_server, "tool"):
                target_server.tool()(_make_tool(tool_name, spec.description or f"Dynamic tool {tool_name}"))
                registered += 1
    except Exception:
        pass
    return registered


__all__ = [
    "MCP_AVAILABLE",
    "asgi_app",
    "configure_tool_gateway",
    "lifespan",
    "mcp",
    "register_plugin_tools",
]

