"""FastAPI composition boundary for the platform."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..domain.errors import (
    AuthorizationError,
    ModelNotLoadedError,
    NotFoundError,
    ServiceUnavailableError,
    UnprocessableRequestError,
)
from ..settings import ArchitectureSettings


def _register_domain_error_handlers(app: FastAPI) -> None:
    """Translate transport-agnostic domain errors into HTTP responses."""

    def handler_for(status_code: int) -> Callable[[Request, Exception], JSONResponse]:
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
    app = FastAPI(title="3D-Reconstruction AI Server", version="3.0.0", lifespan=lifespan)
    _register_domain_error_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["POST", "GET", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Mcp-Session-Id"],
        expose_headers=["Mcp-Session-Id"],
    )
    return app
