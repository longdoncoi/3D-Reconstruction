"""Typed exceptions for the AI Assistant platform.

All domain-level exceptions derive from ``PlatformError`` so that adapters
can catch them in a single handler and translate to the appropriate HTTP or
protocol response.
"""
from __future__ import annotations


class PlatformError(Exception):
    """Base for all recoverable platform errors."""


class ConfigurationError(PlatformError):
    """Raised when a required configuration value is missing or invalid."""


class ModelNotLoadedError(PlatformError):
    """The LLM backend has not been initialised yet."""


class ModelLoadError(PlatformError):
    """The LLM backend failed to load the requested model."""


class ContextWindowExceededError(PlatformError):
    """The conversation or agent context exceeds the model window."""


class ToolNotFoundError(PlatformError):
    """A requested tool is not registered in the tool registry."""


class ToolValidationError(PlatformError):
    """Tool parameters failed schema validation."""

    def __init__(self, tool_name: str, detail: str) -> None:
        self.tool_name = tool_name
        self.detail = detail
        super().__init__(f"Validation failed for tool '{tool_name}': {detail}")


class ToolExecutionError(PlatformError):
    """A tool executor raised an unhandled error."""

    def __init__(self, tool_name: str, cause: Exception) -> None:
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(f"Tool '{tool_name}' failed: {cause}")


class SandboxViolationError(PlatformError):
    """An agent action was blocked by the sandbox policy."""


class ApprovalRequiredError(PlatformError):
    """A state-changing tool requires explicit user approval."""

    def __init__(self, tool_name: str, action_id: str) -> None:
        self.tool_name = tool_name
        self.action_id = action_id
        super().__init__(f"Tool '{tool_name}' requires approval (action_id={action_id})")


class RAGNotReadyError(PlatformError):
    """The RAG index has not been built or is currently rebuilding."""


class TaskCancelledError(PlatformError):
    """The agent task was cancelled cooperatively."""


class NotFoundError(PlatformError):
    """A requested resource does not exist or is no longer available."""


class AuthorizationError(PlatformError):
    """The caller is not permitted to perform the requested operation."""


class UnprocessableRequestError(PlatformError):
    """The request is syntactically valid but semantically invalid."""


class ServiceUnavailableError(PlatformError):
    """A required backend or capability is not currently available."""
