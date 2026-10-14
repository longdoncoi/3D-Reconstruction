"""OpenTelemetry tracing and unified context managers.

The OTel tracer and the Prometheus registry are both built lazily on first use,
so importing this module never reads the process environment (ADR 0001). A
disabled or unavailable SDK yields a no-op span identical to the current API.
When ``AGENT_OBSERVABILITY=1`` *and* ``AGENT_OTLP_ENDPOINT`` points at a
collector (and the exporter package is installed) the tracer is given a
``BatchSpanProcessor`` so spans actually leave the process; every other
combination records nothing and never raises into the call path.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from ai_assistant.settings import observability_enabled, otlp_exporter_endpoint

from .langsmith_integration import (
    _active_langsmith_run,
    finish_langsmith_run,
    start_langsmith_run,
)
from .metrics import get_metrics, record_latency

logger = logging.getLogger("ai_assistant.observability.tracing")

_tracer_lock = threading.Lock()
# ``False`` is the "disabled" sentinel; otherwise holds the OTel Tracer (or None).
_tracer: Any = None


def _build_tracer() -> Any:
    """Resolve the tracer, wiring the OTLP exporter when one is configured."""
    from opentelemetry import trace

    endpoint = otlp_exporter_endpoint()
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            logger.warning(
                "AGENT_OTLP_ENDPOINT is set but opentelemetry-exporter-otlp is "
                "not installed — spans stay in-process"
            )
        else:
            provider = TracerProvider()
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
            trace.set_tracer_provider(provider)
            logger.info("OTLP span export enabled -> %s", endpoint)
    return trace.get_tracer("3d-reconstruction.agent")


def _get_tracer():
    """Return the cached OTel tracer, or ``None`` when observability is off."""
    global _tracer
    if _tracer is None:
        with _tracer_lock:
            if _tracer is None:
                _tracer = False
                if observability_enabled():
                    try:
                        _tracer = _build_tracer()
                    except ImportError:
                        _tracer = False
    return _tracer


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """Emit an OpenTelemetry span and a nested LangSmith run when enabled."""
    started = time.monotonic()
    tracer = _get_tracer()
    trace_span = tracer.start_as_current_span(name) if tracer else None
    if trace_span:
        trace_span.__enter__()
        for key, value in attributes.items():
            if value is not None:
                trace_span.set_attribute(key, value)

    run_id = start_langsmith_run(name, "chain", {"attributes": attributes}, attributes)
    token = _active_langsmith_run.set(run_id) if run_id else None
    error: Exception | None = None
    try:
        yield
        outcome = "success"
    except Exception as exc:
        error = exc
        outcome = "error"
        raise
    finally:
        elapsed = time.monotonic() - started
        record_latency(name, elapsed)
        metrics = get_metrics()
        if metrics:
            metrics["requests"].labels(name, outcome).inc()
            metrics["latency"].labels(name).observe(elapsed)
        if trace_span:
            trace_span.__exit__(type(error) if error else None, error,
                                error.__traceback__ if error else None)
        if run_id:
            finish_langsmith_run(run_id, {
                "outcome": outcome, "duration_ms": round(elapsed * 1000),
            }, error)
        if token is not None:
            _active_langsmith_run.reset(token)


@contextmanager
def langsmith_trace(name: str, run_type: str = "chain",
                    inputs: dict[str, Any] | None = None,
                    metadata: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Trace a logical agent run and yield a dict for its output payload.

    A disabled or unavailable SDK yields the same no-op context, so callers
    do not need environment checks and the existing OTel/file logging path is
    unaffected.
    """
    context: dict[str, Any] = {"run_id": None, "outputs": {}}
    run_id = start_langsmith_run(name, run_type, inputs or {}, metadata)
    if run_id is None:
        yield context
        return

    context["run_id"] = run_id
    token = _active_langsmith_run.set(run_id)
    error: Exception | None = None
    try:
        yield context
    except Exception as exc:
        error = exc
        raise
    finally:
        finish_langsmith_run(run_id, context.get("outputs", {}), error)
        _active_langsmith_run.reset(token)


def current_traceparent() -> str | None:
    """The W3C ``traceparent`` for the active span, or ``None`` outside one.

    Lets an outbound call (the A2A client) carry the trace across the process
    boundary so a remote agent's spans can join the same trace. With no OTel
    SDK importable, observability off, or no active span there is nothing
    valid to propagate — ``None`` is returned rather than a fabricated id, so
    the caller omits the header instead of starting a bogus trace.
    """
    try:
        from opentelemetry import trace
    except ImportError:
        return None
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    trace_flags = getattr(context, "trace_flags", 0)
    return f"00-{context.trace_id:032x}-{context.span_id:016x}-{trace_flags:02x}"
