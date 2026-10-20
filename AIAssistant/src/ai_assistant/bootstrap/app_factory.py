"""FastAPI composition boundary for the platform."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..adapters.http.rate_limit import PeerRateLimiter, RateLimitMiddleware
from ..domain.errors import (
    AuthorizationError,
    ModelNotLoadedError,
    NotFoundError,
    ServiceUnavailableError,
    UnprocessableRequestError,
)
from ..observability.tracing import attach_inbound_traceparent
from ..settings import ArchitectureSettings, max_request_bytes, rate_limit_per_minute

# FastAPI accepts both sync and async exception handlers.
ExceptionHandler = Callable[[Request, Exception], JSONResponse | Awaitable[JSONResponse]]


def _traceparent_from_scope(scope: dict[str, object]) -> str | None:
    """Read the W3C ``traceparent`` header out of an ASGI HTTP scope."""
    headers = scope.get("headers", ())
    for name, value in headers:  # type: ignore[attr-defined]
        if name.lower() == b"traceparent":
            return cast(str, value.decode("latin-1"))
    return None


class TraceContextMiddleware:
    """Inbound W3C trace propagation for every HTTP request (ADR 0001).

    The outbound half lives in ``observability.tracing.current_traceparent``
    (the A2A client attaches the header on the way out); this is the inbound
    half. It is a pure ASGI middleware rather than ``BaseHTTPMiddleware`` so
    the OTel context attach happens in the request's own task, which sync
    route handlers (run in the threadpool) and the A2A worker thread (which
    copies the context in ``TaskService``) both inherit.

    A missing or malformed header, or an SDK that is not importable, is a
    no-op — propagation may fail, the request may not.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: dict[str, object], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        with attach_inbound_traceparent(_traceparent_from_scope(scope)):
            await self._app(scope, receive, send)


class BodySizeLimitMiddleware:
    """Refuse request bodies over the deployment cap before they are parsed.

    The declared ``Content-Length`` is refused up front (413); a chunked or
    lying body is re-checked as it streams and surfaces to the application as
    a disconnect, which Starlette turns into ``ClientDisconnect`` — the
    connection dies instead of the payload being buffered whole. The cap is
    ``settings.max_request_bytes`` (default 2 MiB): every JSON surface takes
    small documents, so an unbounded body is only ever an attack.
    """

    def __init__(self, app: Any, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def _refuse(self, send: Any) -> None:
        await send({
            "type": "http.response.start", "status": 413,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": b'{"detail":"Request body too large"}'})

    async def __call__(self, scope: dict[str, object], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        headers = scope.get("headers", ())
        for name, value in headers:  # type: ignore[attr-defined]
            if name.lower() == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = -1  # unparsable length: let the stream guard decide
                if declared > self._max_bytes:
                    await self._refuse(send)
                    return
        received = 0

        async def guarded_receive() -> dict[str, Any]:
            nonlocal received
            message: dict[str, Any] = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_bytes:
                    # A disconnect is the one in-band way to stop the app
                    # mid-parse without inventing an exception protocol.
                    return {"type": "http.disconnect"}
            return message

        await self._app(scope, guarded_receive, send)


def _register_domain_error_handlers(app: FastAPI) -> None:
    """Translate transport-agnostic domain errors into HTTP responses."""

    def handler_for(status_code: int) -> ExceptionHandler:
        async def handler(_request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(status_code=status_code, content={"detail": str(exc)})

        return handler

    for error_type, status_code in (
        (NotFoundError, 404),
        (AuthorizationError, 403),
        (UnprocessableRequestError, 422),
        (ModelNotLoadedError, 503),
        (ServiceUnavailableError, 503),
    ):
        app.add_exception_handler(error_type, handler_for(status_code))


def create_app(settings: ArchitectureSettings, lifespan: Callable[..., Any]) -> FastAPI:
    # ADR 0032: ``/docs``, ``/redoc`` and ``/openapi.json`` are *app-level*
    # FastAPI routes, not router routes, so the transport guard that protects
    # every other surface never reaches them. They therefore ship disabled and
    # a deployment opts in explicitly.
    docs = settings.expose_api_docs
    app = FastAPI(
        title="3D-Reconstruction AI Server",
        version="3.0.0",
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )
    _register_domain_error_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["POST", "GET", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Mcp-Session-Id", "traceparent"],
        expose_headers=["Mcp-Session-Id"],
    )
    # Every JSON surface is bounded before parsing: declared oversize bodies
    # die with 413, streamed oversize bodies surface as a disconnect.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_request_bytes())
    # Admission governor (ADR 0024): a fixed-window per-peer budget answers
    # 429 + Retry-After before any handler runs. Installed only when
    # configured; the container probes are exempt inside the middleware.
    limiter = PeerRateLimiter(rate_limit_per_minute())
    if limiter.enabled:
        app.add_middleware(RateLimitMiddleware, limiter=limiter)
    # Host header allow-list (ADR 0032). Installed only when a deployment
    # declares one: an empty list means "no policy declared", not "deny
    # everything" — the desktop deployment answers on loopback under whatever
    # name the Qt client used.
    if settings.allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    # Outermost: the inbound traceparent must wrap every other middleware and
    # route so their spans join the caller's trace (ADR 0001).
    app.add_middleware(TraceContextMiddleware)
    return app
