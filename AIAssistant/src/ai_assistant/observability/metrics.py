"""Prometheus metrics for agent observability.

Enabled only if ``AGENT_OBSERVABILITY=1`` is set. The flag is parsed through
``settings`` on first use, so importing this module never reads the process
environment (ADR 0001).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any

from ai_assistant.settings import observability_enabled

_metrics: dict[str, Any] | None = None

#: Per-node samples retained for in-process quantiles (SCORECARD #2). Bounded
#: so a long-lived process cannot grow this without limit.
_LATENCY_WINDOW = 512
_latency_samples: dict[str, deque[float]] = {}


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
                    "recovery": Histogram(
                        "agent_recovery_seconds",
                        "Time from the cancel request to the runner actually stopping",
                    ),
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


def record_recovery(seconds: float) -> None:
    """Report cancel-to-stop latency — SCORECARD metric #4."""
    metrics = get_metrics()
    if metrics:
        metrics["recovery"].observe(max(0.0, seconds))


def record_latency(name: str, seconds: float) -> None:
    """Track a bounded window of raw latency samples per node.

    Prometheus histograms answer quantiles only at query time (PromQL
    ``histogram_quantile``), but SCORECARD metric #2 wants a *measured* p95
    without a round-trip, so the most recent ``_LATENCY_WINDOW`` samples per
    node are kept in-process. Gated behind observability like every other
    recorder — a process with observability off measures nothing.
    """
    metrics = get_metrics()
    if metrics:
        _latency_samples.setdefault(name, deque(maxlen=_LATENCY_WINDOW)).append(
            max(0.0, seconds)
        )


def _p95_of(samples: deque[float]) -> float:
    """p95 of a non-empty sample window (nearest-rank)."""
    ordered = sorted(samples)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def p95_latency_seconds(name: str) -> float | None:
    """p95 of the retained window for one node, or ``None`` when it has none."""
    samples = _latency_samples.get(name)
    if not samples:
        return None
    return _p95_of(samples)


def p95_latency_by_node(prefix: str = "agent.") -> dict[str, float]:
    """Measured p95 per timed node, keyed without the span prefix.

    ``span()`` names graph nodes ``agent.<node>``; this returns ``{node: p95}``
    for the nodes that actually carry a timed span (``plan``, ``reason``,
    ``tool``, ``reflect``). ``approval`` is a decision counter, not a timed
    span, so it never appears here — measuring it would report human response
    time rather than system latency. Every entry in ``_latency_samples`` is
    non-empty by construction (``record_latency`` appends what it creates), so
    no emptiness guard is needed here.
    """
    return {
        name.removeprefix(prefix): _p95_of(samples)
        for name, samples in sorted(_latency_samples.items())
        if name.startswith(prefix)
    }


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
