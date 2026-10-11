"""Legacy agent compatibility bridges (ADR 0001).

Production wires a single :class:`AgentService` in the composition root with an
injected tool gateway and LLM runtime. This adapter keeps the process-level
singleton and the constrained-completion helpers that the legacy
``ai_assistant.legacy.agent_module`` bridge and its regression scripts import,
so that ``ai_assistant.agents`` and ``ai_assistant.application`` stay free of
the ``ai_assistant.legacy`` package.
"""
from __future__ import annotations

import os  # noqa: F401  (re-exported for legacy callers via the legacy adapters)
from collections.abc import Callable

from ai_assistant.agents.completion import constrained_completion, parse_tool_call
from ai_assistant.agents.models import (
    AgentApproveRequest,
    AgentCancelRequest,
    AgentExecuteRequest,
    AgentUiActionResultRequest,
)
from ai_assistant.agents.service import AgentService
from ai_assistant.legacy import llm_module as llm_runtime
from ai_assistant.llm.inference import backend_mode, openai_compatible_completion
from ai_assistant.observability import record_schema_error, record_token_usage
from ai_assistant.tools.factory import create_tool_registry, platform_executors
from ai_assistant.tools.tool_contract import validate_tool_call

# The legacy bridge owns a private catalog instance; production still builds
# the single shared registry in the composition root.
TOOL_REGISTRY = create_tool_registry()
AGENT_TOOLS = list(TOOL_REGISTRY.get_all())

_TOOL_PARAM_MODELS = TOOL_REGISTRY.models
_LLAMA_CPP_TOOLS = TOOL_REGISTRY.get_openai_tools()
_TOOL_GRAMMAR_SCHEMA = TOOL_REGISTRY.grammar

_default_service = AgentService(llm_runtime=llm_runtime, tool_registry=TOOL_REGISTRY)
_pending_actions = _default_service.pending_actions
_pending_lock = _default_service.pending_lock


def _save_pending_actions() -> None:
    _default_service._save_pending()


def _load_pending_actions() -> None:
    _default_service.pending_actions.load()


def _constrained_agent_completion(messages: list[dict], max_tokens: int, temperature: float) -> str:
    """Compatibility entry point for constrained completion."""
    return constrained_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        llm_runtime=llm_runtime,
        backend_mode_fn=backend_mode,
        openai_compatible_fn=openai_compatible_completion,
        openai_tools=_LLAMA_CPP_TOOLS,
        grammar_schema=_TOOL_GRAMMAR_SCHEMA,
        record_token_usage_fn=record_token_usage,
    )


def _parse_tool_call(response_text: str) -> tuple[str | None, dict | None]:
    return parse_tool_call(response_text, _TOOL_PARAM_MODELS, validate_tool_call, record_schema_error)


def agent_execute(request: AgentExecuteRequest, *, event_sink: Callable[[dict], None] | None = None,
                  client_host: str = "unknown") -> dict:
    return _default_service.execute(request, event_sink=event_sink, client_host=client_host)


def agent_cancel(request: AgentCancelRequest) -> dict:
    return _default_service.cancel(request)


def agent_ui_action_result(request: AgentUiActionResultRequest) -> dict:
    return _default_service.ui_action_result(request)


def agent_approve(request: AgentApproveRequest) -> dict:
    return _default_service.approve(request)


def reset_agent_state() -> None:
    _default_service.reset_state()


__all__ = [
    "AGENT_TOOLS",
    "TOOL_REGISTRY",
    "AgentApproveRequest",
    "AgentCancelRequest",
    "AgentExecuteRequest",
    "AgentService",
    "AgentUiActionResultRequest",
    "_constrained_agent_completion",
    "_load_pending_actions",
    "_parse_tool_call",
    "_pending_actions",
    "_pending_lock",
    "_save_pending_actions",
    "agent_approve",
    "agent_cancel",
    "agent_execute",
    "agent_ui_action_result",
    "backend_mode",
    "llm_runtime",
    "platform_executors",
    "reset_agent_state",
]
