"""Synchronous bridge from the LangGraph worker to the local MCP server."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any


def _endpoint() -> str:
    return os.getenv("AGENT_MCP_URL", "http://127.0.0.1:8080/mcp").rstrip("/")


async def _call(tool_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(_endpoint()) as (read_stream, write_stream, *_):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, parameters)

    if getattr(result, "isError", False):
        return {"error": "MCP tool returned an error", "details": str(result.content)}
    content = getattr(result, "content", [])
    if not content:
        return {"error": "MCP tool returned no content"}
    text = getattr(content[0], "text", None)
    if not isinstance(text, str):
        return {"error": "MCP tool returned unsupported content"}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"error": "MCP tool returned invalid JSON", "content": text}
    return payload if isinstance(payload, dict) else {"result": payload}


def call_tool(tool_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Call a server tool through MCP from the synchronous agent graph.

    LangGraph executes this in FastAPI's worker thread, therefore a dedicated
    event loop is safe and prevents sharing state with Uvicorn's event loop.
    """
    try:
        return asyncio.run(_call(tool_name, parameters))
    except ImportError:
        return {"error": "MCP SDK is not installed. Install AIAssistant/requirements.txt."}
    except Exception as error:  # noqa: BLE001 - callers need a model-visible tool error.
        return {"error": f"MCP call failed for {tool_name}: {error}"}
