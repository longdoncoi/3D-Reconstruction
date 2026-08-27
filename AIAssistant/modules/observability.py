"""Optional OpenTelemetry and Prometheus instrumentation for the AI server."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator

_enabled = os.getenv("AGENT_OBSERVABILITY", "0") == "1"
_metrics = None
_tracer = None

if _enabled:
    try:
        from prometheus_client import Counter, Histogram
        _metrics = {
            "requests": Counter("agent_requests_total", "Agent requests", ["endpoint", "status"]),
            "latency": Histogram("agent_request_duration_seconds", "Agent request latency", ["endpoint"]),
            "tool": Counter("agent_tool_calls_total", "Tool calls", ["tool", "outcome"]),
            "tokens": Counter("agent_tokens_total", "Tokens used", ["type"]),
        }
        from opentelemetry import trace
        _tracer = trace.get_tracer("3d-reconstruction.agent")
    except ImportError:
        _enabled = False


@contextmanager
def span(name: str, **attributes: str) -> Iterator[None]:
    started = time.monotonic()
    trace_span = _tracer.start_as_current_span(name) if _tracer else None
    if trace_span:
        trace_span.__enter__()
        for key, value in attributes.items():
            trace_span.set_attribute(key, value)
    try:
        yield
        outcome = "success"
    except Exception:
        outcome = "error"
        raise
    finally:
        if _metrics:
            _metrics["requests"].labels(name, outcome).inc()
            _metrics["latency"].labels(name).observe(time.monotonic() - started)
        if trace_span:
            trace_span.__exit__(None, None, None)


def record_tool(tool: str, success: bool) -> None:
    if _metrics:
        _metrics["tool"].labels(tool, "success" if success else "error").inc()


def prometheus_payload() -> bytes | None:
    if not _metrics:
        return None
    from prometheus_client import generate_latest
    return generate_latest()


def record_token_usage(in_tokens: int, out_tokens: int) -> None:
    if _metrics:
        _metrics["tokens"].labels("prompt").inc(in_tokens)
        _metrics["tokens"].labels("completion").inc(out_tokens)
