"""Optional OpenTelemetry, Prometheus, and LangSmith instrumentation."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator
from uuid import uuid4

_enabled = os.getenv("AGENT_OBSERVABILITY", "0") == "1"
_metrics = None
_tracer = None

if _enabled:
    try:
        from opentelemetry import trace
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
        _tracer = trace.get_tracer("3d-reconstruction.agent")
    except ImportError:
        _enabled = False

_langsmith_enabled = False
_langsmith_client = None
_active_langsmith_run: ContextVar[str | None] = ContextVar(
    "active_langsmith_run", default=None,
)
_ls_project = os.getenv("LANGSMITH_PROJECT", "3d-reconstruction")

if (os.getenv("LANGSMITH_TRACING", "").lower() in {"true", "1", "yes"}
        and os.getenv("LANGSMITH_API_KEY", "")):
    try:
        import langsmith

        _langsmith_client = langsmith.Client(
            api_key=os.environ["LANGSMITH_API_KEY"],
            api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        )
        _langsmith_enabled = True
    except (ImportError, Exception):
        # Observability must never prevent the server from starting.
        pass


def langsmith_available() -> bool:
    """Return whether LangSmith tracing has an enabled, usable client."""
    return _langsmith_enabled


def get_langsmith_client() -> Any:
    """Return the LangSmith client, or ``None`` when tracing is disabled."""
    return _langsmith_client


def _start_langsmith_run(name: str, run_type: str, inputs: dict[str, Any],
                         metadata: dict[str, Any] | None = None) -> str | None:
    """Start a nested run without allowing telemetry failures to escape."""
    if not _langsmith_enabled or _langsmith_client is None:
        return None
    run_id = str(uuid4())
    payload: dict[str, Any] = {
        "name": name,
        "run_type": run_type,
        "id": run_id,
        "project_name": _ls_project,
        "inputs": inputs,
    }
    parent_run_id = _active_langsmith_run.get()
    if parent_run_id:
        payload["parent_run_id"] = parent_run_id
    if metadata:
        payload["extra"] = {"metadata": metadata}
    try:
        _langsmith_client.create_run(**payload)
        return run_id
    except TypeError:
        # Compatibility with older clients that accepted ``run_id`` instead.
        payload["run_id"] = payload.pop("id")
        try:
            _langsmith_client.create_run(**payload)
            return run_id
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None


def _finish_langsmith_run(run_id: str, outputs: dict[str, Any],
                          error: Exception | None = None) -> None:
    """End a run using the current or a compatible older SDK signature."""
    if _langsmith_client is None:
        return
    payload: dict[str, Any] = {"run_id": run_id, "outputs": outputs}
    if error is not None:
        payload["error"] = str(error)
    try:
        _langsmith_client.update_run(**payload)
    except TypeError:
        payload["id"] = payload.pop("run_id")
        try:
            _langsmith_client.update_run(**payload)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """Emit an OpenTelemetry span and a nested LangSmith run when enabled."""
    started = time.monotonic()
    trace_span = _tracer.start_as_current_span(name) if _tracer else None
    if trace_span:
        trace_span.__enter__()
        for key, value in attributes.items():
            if value is not None:
                trace_span.set_attribute(key, value)

    run_id = _start_langsmith_run(name, "chain", {"attributes": attributes}, attributes)
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
        if _metrics:
            _metrics["requests"].labels(name, outcome).inc()
            _metrics["latency"].labels(name).observe(elapsed)
        if trace_span:
            trace_span.__exit__(type(error) if error else None, error,
                                error.__traceback__ if error else None)
        if run_id:
            _finish_langsmith_run(run_id, {
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
    run_id = _start_langsmith_run(name, run_type, inputs or {}, metadata)
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
        _finish_langsmith_run(run_id, context.get("outputs", {}), error)
        _active_langsmith_run.reset(token)


def record_langsmith_feedback(key: str, score: float | None = None,
                              value: Any | None = None, comment: str | None = None,
                              run_id: str | None = None) -> bool:
    """Attach evaluation feedback to the active (or supplied) LangSmith run."""
    if not _langsmith_enabled or _langsmith_client is None:
        return False
    target_run_id = run_id or _active_langsmith_run.get()
    if not target_run_id:
        return False
    payload: dict[str, Any] = {"run_id": target_run_id, "key": key}
    if score is not None:
        payload["score"] = score
    if value is not None:
        payload["value"] = value
    if comment:
        payload["comment"] = comment
    try:
        _langsmith_client.create_feedback(**payload)
        return True
    except Exception:  # noqa: BLE001
        return False


def record_tool(tool: str, success: bool, duration_seconds: float | None = None) -> None:
    if _metrics:
        _metrics["tool"].labels(tool, "success" if success else "error").inc()
        if duration_seconds is not None:
            _metrics["tool_latency"].labels(tool).observe(max(0.0, duration_seconds))


def record_approval(outcome: str) -> None:
    """Record an approval decision (approved, rejected, expired or missing)."""
    if _metrics:
        _metrics["approval"].labels(outcome).inc()


def record_schema_error(tool: str) -> None:
    if _metrics:
        _metrics["schema"].labels(tool).inc()


def prometheus_payload() -> bytes | None:
    if not _metrics:
        return None
    from prometheus_client import generate_latest
    return generate_latest()


def record_token_usage(in_tokens: int, out_tokens: int) -> None:
    if _metrics:
        _metrics["tokens"].labels("prompt").inc(in_tokens)
        _metrics["tokens"].labels("completion").inc(out_tokens)

def record_step_guard(outcome: str) -> None:
    if _metrics:
        _metrics["guard"].labels(outcome).inc()        
