"""MCP tool server for the 3D-Reconstruction desktop agent.

The agent owns authorisation and the Qt client owns desktop actions.  This
server only exposes the non-destructive tools that can run without an
interactive approval.  It is deliberately mounted in the existing FastAPI
process so a local MCP client and the bundled agent use exactly the same tool
contract.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:  # MCP SDK 2.x renamed FastMCP to MCPServer.
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:  # Keep the chat server importable until its dependency is installed.
        FastMCP = None  # type: ignore[misc,assignment]


MCP_AVAILABLE = FastMCP is not None


def _dispatch(tool_name: str, parameters: dict[str, Any]) -> str:
    """Call the existing, policy-enforced implementation and encode MCP text."""
    # Import lazily: agent_module imports the MCP client adapter.
    from . import agent_module

    parameters = {name: value for name, value in parameters.items() if value is not None}
    executor: Callable[[dict[str, Any]], dict[str, Any]] | None = agent_module.MCP_LOCAL_EXECUTORS.get(tool_name)
    if executor is None:
        return json.dumps({"error": f"MCP tool is not available: {tool_name}"}, ensure_ascii=False)
    try:
        return json.dumps(executor(parameters), ensure_ascii=False)
    except Exception as error:  # noqa: BLE001 - MCP must return tool failures to the model.
        return json.dumps({"error": f"MCP tool failed: {error}"}, ensure_ascii=False)


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
    def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        """Read a text file within the project directory."""
        return _dispatch("read_file", {"path": path, "start_line": start_line, "end_line": end_line})

    @mcp.tool()
    def list_directory(path: str, recursive: bool | None = None, max_depth: int | None = None) -> str:
        """List files and folders within the project directory."""
        return _dispatch("list_directory", {"path": path, "recursive": recursive, "max_depth": max_depth})

    @mcp.tool()
    def search_text(query: str, path: str | None = None, file_pattern: str | None = None,
                    case_sensitive: bool | None = None, max_results: int | None = None) -> str:
        """Search project text and return matching file paths and line numbers."""
        return _dispatch("search_text", {"query": query, "path": path, "file_pattern": file_pattern,
                                          "case_sensitive": case_sensitive, "max_results": max_results})

    @mcp.tool()
    def analyze_code(path: str) -> str:
        """Summarise the imports, classes, and functions in a source file."""
        return _dispatch("analyze_code", {"path": path})

    @mcp.tool()
    def get_project_status() -> str:
        """Get the current Git branch, modified files, and source summary."""
        return _dispatch("get_project_status", {})

    @mcp.tool()
    def validate_file(path: str) -> str:
        """Syntax-check a Python file or parse a JSON file without changing it."""
        return _dispatch("validate_file", {"path": path})

    @mcp.tool()
    def rag_search(query: str, top_k: int | None = None) -> str:
        """Search indexed project documentation and source code."""
        return _dispatch("rag_search", {"query": query, "top_k": top_k})

    @mcp.tool()
    def application_action(action: str, language: str | None = None, username: str | None = None,
                           password: str | None = None) -> str:
        """Request a canonical desktop action; the Qt desktop client must acknowledge it."""
        return _dispatch("application_action", {"action": action, "language": language,
                                                  "username": username, "password": password})

    # The app is mounted at /mcp, so make its endpoint the mount root.
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
