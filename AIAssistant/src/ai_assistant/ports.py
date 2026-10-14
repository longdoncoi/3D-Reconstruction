from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

from .domain.security import Principal
from .domain.tasks import AgentTask, TaskStatus
from .domain.tools import ToolRequest, ToolResult


class LockLike(Protocol):
    """Structural type satisfied by ``threading.Lock`` and ``threading.RLock``.

    The agent layer only enters the lock; typing it as the concrete
    ``threading.Lock`` would wrongly reject the ``RLock`` the legacy runtime
    owns, and ``contextlib.AbstractContextManager`` requires nominal
    inheritance that neither lock type declares.
    """

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool: ...
    def release(self) -> None: ...
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...


class LLMBackend(Protocol):
    """Port for a loaded inference backend.

    The core owns this interface so neither ``ports`` nor ``agents`` has to
    import the ``ai_assistant.llm`` infrastructure package (import-linter
    contract 1). ``ai_assistant.llm.contracts`` re-exports it for the adapters
    that used to define it.
    """

    @property
    def is_vision_supported(self) -> bool:
        """Return True if this backend can process images."""
        ...

    @property
    def model_description(self) -> str:
        """Return a human-readable description of the loaded model."""
        ...

    def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
        stream: bool = False,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a response synchronously (raw backend payload)."""
        ...


# A plain function type, not a ``Protocol`` with ``__call__``: every executor
# in the tree is a free function, and mypy does not unify a
# ``Callable[[dict[str, Any]], dict[str, Any]]`` with a callable protocol whose
# parameter is ``dict[str, Any]`` (the ``Any`` makes the structural match fail),
# so the alias is both simpler and actually checkable.
ToolExecutor = Callable[[dict[str, Any]], dict[str, Any]]


class ToolGateway(Protocol):
    """Least-privilege gateway through which agent engines execute tools.

    The adapter behind this port wraps :class:`~ai_assistant.application.tools.ToolExecutionService`
    and resolves the active principal at call time, so the agent engine never
    imports the composition root and never reaches the registry directly. The
    production implementation is ``bootstrap.runtime.PlatformToolGateway``.

    Approval-gated tools are gated by a *capability*, not a flag: a grant is
    issued for the exact invocation a user approved, spent on execution and
    probed with :meth:`approval_covers` when the engine decides whether it has
    to ask again.
    """

    def execute(self, tool_name: str, parameters: dict) -> dict: ...

    def execute_approved(self, tool_name: str, parameters: dict, approval_token: str = "") -> dict: ...

    def issue_approval_grant(self, tool_name: str, parameters: dict, approval_token: str) -> None: ...

    def approval_covers(self, tool_name: str, parameters: dict, approval_token: str) -> bool: ...


class AgentOrchestrator(Protocol):
    """Decision-loop engine behind an A2A task (ADR 0003).

    The application layer owns scope resolution and the durable continuation
    envelope; the orchestrator owns "given this task, what should the agent do
    next". The production implementation is the LangGraph engine, reached
    through the adapter in ``adapters/orchestration``.
    """

    def run(self, task: AgentTask, principal: Principal) -> dict: ...


class TaskStore(Protocol):
    def create(self, task: AgentTask) -> None: ...
    def get(self, task_id: str) -> AgentTask | None: ...
    def save(self, task: AgentTask) -> None: ...

    def save_if_status(self, task: AgentTask, expected: TaskStatus) -> bool:
        """Compare-and-set write: persist ``task`` only when the stored status
        still equals ``expected``; return whether the write happened.

        This is the TOCTOU guard for the A2A worker: without it a cancel that
        lands while the executor runs is overwritten by an unconditional
        ``UPDATE ... SET status='completed'``.
        """
        ...

    def append_event(self, task_id: str, kind: str, data: dict) -> None: ...

    def events(self, task_id: str, *, since: int = 0) -> Iterable[dict]:
        """Yield events with an ordering key strictly greater than ``since``.

        ``since`` is a store-native cursor (SQLite rowid); stores without a
        native cursor may ignore it, in which case the caller falls back to
        slicing by count.
        """
        ...

    def list(self, *, context_id: str | None = None, limit: int = 100) -> Iterable[AgentTask]: ...


class LLMRuntime(Protocol):
    """Infrastructure LLM handle injected into the agent decision loop.

    The agent layer only needs the loaded backend and the lock that guards it;
    it never imports the model loader, so the inference technology stays
    replaceable and the composition root owns the concrete runtime
    (ADR 0001). ``legacy.llm_module`` satisfies this protocol.

    The lock is typed structurally (:class:`LockLike`) so both
    ``threading.Lock`` and ``threading.RLock`` satisfy it.
    """

    llm: LLMBackend | None
    llm_lock: LockLike


class AgentTaskExecutor(Protocol):
    def __call__(self, task: AgentTask) -> dict: ...


ToolEventSink = Callable[[ToolRequest, ToolResult], None]
