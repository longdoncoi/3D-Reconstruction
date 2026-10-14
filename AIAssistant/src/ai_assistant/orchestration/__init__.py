"""Orchestration package for LangGraph-based agent loop.

This package re-organises the former LangGraphAgent module into focused
modules; its former public exports live directly in ``orchestration/graph.py``
(the tests-only ``compat`` shim was removed by the ADR 0015 dead-code sweep).

The package itself is framework-free.  ``LocalAgentGraph`` lives in
``orchestration/graph.py`` and is imported from there, never re-exported here: an eager re-export made ``import
ai_assistant.orchestration`` — and with it the whole ``agents`` layer, which
imports ``orchestration.specialists`` — hard-fail in an environment without
langgraph, turning the runner's documented ``None``-engine degradation into
``ServiceUnavailableError`` into unreachable code (architecture review
finding 5).
"""
from .state import AgentState

__all__ = ["AgentState"]
