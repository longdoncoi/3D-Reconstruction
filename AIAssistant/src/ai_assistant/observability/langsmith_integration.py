"""LangSmith integration for tracing LLM execution.

Enabled only if ``LANGSMITH_TRACING`` is set and ``LANGSMITH_API_KEY`` is
provided. Credentials are parsed on demand through ``settings``, so importing
this module reads no environment variables (ADR 0001).
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any
from uuid import uuid4

from ai_assistant.settings import langsmith_settings

_active_langsmith_run: ContextVar[str | None] = ContextVar(
    "active_langsmith_run", default=None,
)
_initialized = False
_client_settings: dict[str, str] | None = None
_langsmith_client: Any | None = None


def _get_client() -> Any:
    """Return the cached LangSmith client, or ``None`` when tracing is disabled."""
    global _initialized, _client_settings, _langsmith_client
    if not _initialized:
        _initialized = True
        _client_settings = langsmith_settings()
        if _client_settings:
            try:
                import langsmith

                _langsmith_client = langsmith.Client(
                    api_key=_client_settings["api_key"],
                    api_url=_client_settings["endpoint"],
                )
            except Exception:  # noqa: BLE001  — observability must never block startup
                _langsmith_client = None
    return _langsmith_client


def langsmith_available() -> bool:
    """Return whether LangSmith tracing has an enabled, usable client."""
    return _get_client() is not None


def get_langsmith_client() -> Any:
    """Return the LangSmith client, or ``None`` when tracing is disabled."""
    return _get_client()


def start_langsmith_run(name: str, run_type: str, inputs: dict[str, Any],
                        metadata: dict[str, Any] | None = None) -> str | None:
    """Start a nested run without allowing telemetry failures to escape."""
    client = _get_client()
    if client is None:
        return None
    run_id = str(uuid4())
    payload: dict[str, Any] = {
        "name": name,
        "run_type": run_type,
        "id": run_id,
        "project_name": (_client_settings or {}).get("project", "3d-reconstruction"),
        "inputs": inputs,
    }
    parent_run_id = _active_langsmith_run.get()
    if parent_run_id:
        payload["parent_run_id"] = parent_run_id
    if metadata:
        payload["extra"] = {"metadata": metadata}
    try:
        client.create_run(**payload)
        return run_id
    except TypeError:
        # Compatibility with older clients that accepted ``run_id`` instead.
        payload["run_id"] = payload.pop("id")
        try:
            client.create_run(**payload)
            return run_id
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None


def finish_langsmith_run(run_id: str, outputs: dict[str, Any],
                         error: Exception | None = None) -> None:
    """End a run using the current or a compatible older SDK signature."""
    client = _get_client()
    if client is None:
        return
    payload: dict[str, Any] = {"run_id": run_id, "outputs": outputs}
    if error is not None:
        payload["error"] = str(error)
    try:
        client.update_run(**payload)
    except TypeError:
        payload["id"] = payload.pop("run_id")
        try:
            client.update_run(**payload)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


def record_langsmith_feedback(key: str, score: float | None = None,
                              value: Any | None = None, comment: str | None = None,
                              run_id: str | None = None) -> bool:
    """Attach evaluation feedback to the active (or supplied) LangSmith run."""
    client = _get_client()
    if client is None:
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
        client.create_feedback(**payload)
        return True
    except Exception:  # noqa: BLE001
        return False
