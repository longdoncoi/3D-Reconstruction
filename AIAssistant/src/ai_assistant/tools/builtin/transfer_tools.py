"""Agent transfer tools.

Each transfer tool is a handoff *marker*: it executes nothing and reports the
specialist the supervisor will route to. ``domain.specialists.TRANSFER_TARGETS``
is the naming authority — one entry yields the tool name
(``transfer_to_<specialist>_agent``), the status it returns
(``transferred_to_<specialist>``), the supervisor routing and the
least-privilege tool row — so a new handoff target is added in one place
instead of five parallel ones (architecture finding 12). The LLM-facing
catalog (``Config/agent_tools.json``) still describes each marker; a catalog
entry without a registry entry simply never reaches a handler — it cannot
misroute.
"""
from __future__ import annotations

from typing import Any, Callable

from ai_assistant.domain.specialists import TRANSFER_TARGETS

TransferHandler = Callable[[dict[str, Any]], dict[str, Any]]


def _transfer_tool(specialist_value: str) -> TransferHandler:
    """Build the marker whose status names *specialist_value*."""

    def _transfer(params: dict[str, Any]) -> dict[str, Any]:
        return {"status": f"transferred_to_{specialist_value}", "intent": params.get("intent")}

    return _transfer


#: One handler per handoff target, named after the domain registry.
TRANSFER_TOOL_HANDLERS: dict[str, TransferHandler] = {
    name: _transfer_tool(specialist.value) for name, specialist in TRANSFER_TARGETS.items()
}
