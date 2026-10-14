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


class ToolExecutionGateway(Protocol):
    """Least-privilege *execution* surface a decision loop may reach (ADR 0010).

    The runner executes, pre-flights and probes approval grants but can never
    mint one; grant minting is confined to the approval services behind
    :class:`ApprovalGrantIssuer`. Splitting the two ports makes that separation
    of duties type-enforced rather than a convention: a bug that tried to mint a
    grant from inside the loop would not type-check.

    The adapter behind this port wraps
    :class:`~ai_assistant.application.tools.ToolExecutionService` and resolves
    the active principal at call time, so the agent engine never imports the
    composition root and never reaches the registry directly. The production
    implementation is ``bootstrap.runtime.PlatformToolGateway``.

    Approval-gated tools are gated by a *capability*, not a flag: a grant is
    issued for the exact invocation a user approved, spent on execution and
    probed with :meth:`approval_covers` when the engine decides whether it has
    to ask again. :meth:`preflight` is that same gate for a dispatch that
    executes elsewhere — it validates, authorises and spends the grant without
    running the tool, so a remote leg cannot skip policy by leaving the
    process.
    """

    def execute(self, tool_name: str, parameters: dict) -> dict: ...

    def execute_approved(self, tool_name: str, parameters: dict, approval_token: str = "") -> dict: ...

    def preflight(self, tool_name: str, parameters: dict, approval_token: str = "") -> dict: ...

    def approval_covers(self, tool_name: str, parameters: dict, approval_token: str) -> bool: ...


class ApprovalGrantIssuer(Protocol):
    """Grant-minting surface, reachable only from a user-approval path.

    Only a service that represents a user decision may mint a single-use grant;
    the decision loop itself holds no reference to this port.
    """

    def issue_approval_grant(self, tool_name: str, parameters: dict, approval_token: str) -> None: ...


class ToolGateway(ToolExecutionGateway, ApprovalGrantIssuer, Protocol):
    """The full gateway: execution plus grant minting, as the composition root builds it.

    The approval services hold this (they both mint a grant and immediately
    execute against it); the runner is handed only the
    :class:`ToolExecutionGateway` view.
    """


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
    (ADR 0001). ``bootstrap.server.llm_module`` satisfies this protocol.

    The lock is typed structurally (:class:`LockLike`) so both
    ``threading.Lock`` and ``threading.RLock`` satisfy it.
    """

    llm: LLMBackend | None
    llm_lock: LockLike


class RagContextPort(Protocol):
    """Retrieval the ``rag_search`` tool needs from the live index (ADR 0001).

    The composition root injects the RAG module it owns, so the tool layer
    never imports the bootstrap module — the last service locator in
    ``tools``. A registry built without a provider leaves the tool reporting
    the index as unavailable rather than reaching for a global.
    """

    def get_context(self, query: str, query_image_b64: str | None = None, result_k: int | None = None) -> tuple[Any, ...]: ...


class AgentTaskExecutor(Protocol):
    def __call__(self, task: AgentTask) -> dict: ...


ToolEventSink = Callable[[ToolRequest, ToolResult], None]
