"""Prometheus metrics for agent observability.

Enabled only if ``AGENT_OBSERVABILITY=1`` is set. The flag is parsed through
``settings`` on first use, so importing this module never reads the process
environment (ADR 0001).
"""
from __future__ import annotations

from typing import Any

from ai_assistant.settings import observability_enabled

_metrics: dict[str, Any] | None = None


def get_metrics() -> dict[str, Any]:
    """Lazily build the Prometheus registry, or an empty dict when disabled."""
    global _metrics
    if _metrics is None:
        _metrics = {}
        if observability_enabled():
            try:
                from prometheus_client import Counter, Histogram

                _metrics = {
                    "requests": Counter("agent_requests_total", "Agent requests", ["endpoint", "status"]),
                    "latency": Histogram("agent_request_duration_seconds", "Agent request latency", ["endpoint"]),
                    "tool": Counter("agent_tool_calls_total", "Tool calls", ["tool", "outcome"]),
                    "tool_latency": Histogram("agent_tool_duration_seconds", "Tool duration", ["tool"]),
                    "tokens": Counter("agent_tokens_total", "Tokens used", ["type"]),
                    "approval": Counter("agent_approval_decisions_total", "Approval decisions", ["outcome"]),
                    "schema": Counter("agent_schema_errors_total", "Rejected tool schemas", ["tool"]),
                    "guard": Counter("agent_step_action_guard_total", "Plan-step/action guard decisions", ["outcome"]),
                }
            except ImportError:
                _metrics = {}
    return _metrics


def record_tool(tool: str, success: bool, duration_seconds: float | None = None) -> None:
    metrics = get_metrics()
    if metrics:
        metrics["tool"].labels(tool, "success" if success else "error").inc()
        if duration_seconds is not None:
            metrics["tool_latency"].labels(tool).observe(max(0.0, duration_seconds))


def record_approval(outcome: str) -> None:
    """Record an approval decision (approved, rejected, expired or missing)."""
    metrics = get_metrics()
    if metrics:
        metrics["approval"].labels(outcome).inc()


def record_schema_error(tool: str) -> None:
    metrics = get_metrics()
    if metrics:
        metrics["schema"].labels(tool).inc()


def prometheus_payload() -> bytes | None:
    metrics = get_metrics()
    if not metrics:
        return None
    from prometheus_client import generate_latest
    return generate_latest()


def record_token_usage(in_tokens: int, out_tokens: int) -> None:
    metrics = get_metrics()
    if metrics:
        metrics["tokens"].labels("prompt").inc(in_tokens)
        metrics["tokens"].labels("completion").inc(out_tokens)


def record_step_guard(outcome: str) -> None:
    metrics = get_metrics()
    if metrics:
        metrics["guard"].labels(outcome).inc()
