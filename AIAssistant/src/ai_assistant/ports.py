from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any, Protocol

from .domain.memory import MemoryRecord as MemoryRecord
from .domain.security import Principal
from .domain.tasks import AgentTask, TaskStatus
from .domain.tools import ToolRequest, ToolResult, ToolSpec


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

    def execute(self, tool_name: str, parameters: dict[str, Any]) -> dict[str, Any]: ...

    def execute_approved(
        self, tool_name: str, parameters: dict[str, Any], approval_token: str = ""
    ) -> dict[str, Any]: ...

    def preflight(
        self, tool_name: str, parameters: dict[str, Any], approval_token: str = ""
    ) -> dict[str, Any]: ...

    def approval_covers(self, tool_name: str, parameters: dict[str, Any], approval_token: str) -> bool: ...


class ApprovalGrantIssuer(Protocol):
    """Grant-minting surface, reachable only from a user-approval path.

    Only a service that represents a user decision may mint a single-use grant;
    the decision loop itself holds no reference to this port.
    """

    def issue_approval_grant(self, tool_name: str, parameters: dict[str, Any], approval_token: str) -> None: ...


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

    def run(self, task: AgentTask, principal: Principal) -> dict[str, Any]: ...


class TaskStore(Protocol):
    def commit(
        self, task: AgentTask, *,
        expected: TaskStatus | None = None,
        events: Sequence[tuple[str, dict[str, Any]]] = (),
    ) -> bool:
        """Persist ``task`` and its events as **one unit of work**.

        ``expected`` selects the mode:

        * ``None`` — insert the task; it must not exist yet.
        * a status — compare-and-set: the write happens only while the stored
          status still equals it, and the return value says whether it did.
          This is the TOCTOU guard for the A2A worker: without it a cancel
          that lands while the executor runs is overwritten by an
          unconditional ``UPDATE ... SET status='completed'``.

        The state write and the event append are the same fact observed
        twice. A subscriber that tails the event log must see a transition
        exactly once, and must never see a task whose state changed without
        its event. Two separate transactions break both halves of that
        promise: a crash between them leaves a task that is ``completed``
        with no terminal event, and a subscriber waits forever for an event
        that will never arrive (ADR 0032).
        """
        ...

    def get(self, task_id: str) -> AgentTask | None: ...

    def events(self, task_id: str, *, since: int = 0) -> Iterable[dict[str, Any]]:
        """Yield events with an ordering key strictly greater than ``since``.

        ``since`` is a store-native cursor (SQLite rowid); stores without a
        native cursor may ignore it, in which case the caller falls back to
        slicing by count.
        """
        ...

    def list(self, *, context_id: str | None = None, limit: int = 100, owner: str | None = None) -> Iterable[AgentTask]: ...

    def prune(self, retention_days: int) -> int:
        """Delete terminal tasks older than ``retention_days``; return the count."""
        ...


def store_supports_cursor(events: Any) -> bool:
    """Whether a store's ``events`` callable accepts the ``since`` cursor.

    The duck-typed fallback used to be detected by catching the ``TypeError``
    the *call* raised when the store did not know the keyword. That also
    swallowed a ``TypeError`` raised **inside** the store, and then called the
    store a second time with the old signature — a real bug in the store
    turned into "this store is old". The signature is asked instead.
    """
    try:
        parameters = inspect.signature(events).parameters
    except (TypeError, ValueError):
        return False
    return "since" in parameters


class ToolCatalogEntry(Protocol):
    """One registered tool: its policy record and the executor behind it."""

    @property
    def spec(self) -> ToolSpec: ...

    @property
    def executor(self) -> ToolExecutor: ...


class ToolCatalog(Protocol):
    """The catalogue of tools the platform is allowed to execute.

    ADR 0032: the application layer used to depend on the *concrete*
    ``plugins.registry.PluginRegistry`` — a peer layer's class, reached
    through an import-linter exception. The catalogue is a port, so the
    policy chokepoint depends on the *contract* and the plugin layer stays
    replaceable (and the HTTP adapter's own duck-typed duplicate of it
    becomes redundant).
    """

    def tool(self, name: str) -> ToolCatalogEntry | None:
        """The registered tool, or ``None`` when the name is unknown."""
        ...

    def specs(self) -> tuple[ToolSpec, ...]:
        """Every registered policy record — the catalogue's own listing."""
        ...


class AgentToolSpec(Protocol):
    """The tool record the *agent* surface reads (ADR 0032 follow-up).

    ``ToolCatalog`` covers the execution chokepoint (``tool``/``specs``); the
    prompt builder, the constrained-decoder grammar and the dispatch guard need
    the registrable record instead: the display fields, the parameter map and
    the handler-existence check. ``tools.definition.ToolDefinition`` satisfies
    this structurally, so the agent layer states the contract it consumes
    without importing the concrete tool package.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> dict[str, Any]: ...

    @property
    def requires_approval(self) -> bool: ...

    @property
    def handler(self) -> Callable[[dict[str, Any]], dict[str, Any]] | None: ...

    def to_openai_format(self) -> dict[str, Any]: ...


class AgentToolCatalog(Protocol):
    """The catalogue the agent prompt/grammar/dispatch surface needs.

    This is deliberately a *second* port, not an extension of
    :class:`ToolCatalog`: the execution chokepoint must not inherit the
    prompt-shaped listings, and a catalogue that only serves execution stays
    a valid ``ToolCatalog``. ``tools.registry.ToolRegistry`` satisfies both.
    """

    def get_all(self) -> Sequence[AgentToolSpec]:
        """Every registrable tool, in registration order."""
        ...

    def get(self, name: str) -> AgentToolSpec | None:
        """The tool with this name, or ``None`` when it is unknown."""
        ...

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """The tool list in the OpenAI function-calling shape."""
        ...

    @property
    def grammar(self) -> str:
        """The llama.cpp grammar schema for the current catalogue."""
        ...

    @property
    def models(self) -> dict[str, type]:
        """The compiled parameter models, keyed by tool name."""
        ...


class LLMRuntime(Protocol):
    """Infrastructure LLM handle injected into the agent decision loop.

    The agent layer only needs the loaded backend and the lock that guards it;
    it never imports the model loader, so the inference technology stays
    replaceable and the composition root owns the concrete runtime
    (ADR 0001). ``bootstrap.modules.LlmModule`` satisfies this protocol.

    The lock is typed structurally (:class:`LockLike`) so both
    ``threading.Lock`` and ``threading.RLock`` satisfy it.

    ``model_lease`` is the drain contract (ADR 0032): a run pins one backend
    for its whole length, so an admin reload cannot swap the model in the
    middle of it. It used to be duck-typed with a ``getattr`` probe, which
    meant a runtime that forgot the method silently lost the guarantee — no
    type error, no log. The port now states it.
    """

    llm: LLMBackend | None
    llm_lock: LockLike

    def model_lease(self) -> AbstractContextManager[LLMBackend | None]:
        """Pin the current backend for the duration of the block."""
        ...


class RagContextPort(Protocol):
    """Retrieval the ``rag_search`` tool needs from the live index (ADR 0001).

    The composition root injects the RAG module it owns, so the tool layer
    never imports the bootstrap module — the last service locator in
    ``tools``. A registry built without a provider leaves the tool reporting
    the index as unavailable rather than reaching for a global.
    """

    def get_context(self, query: str, query_image_b64: str | None = None, result_k: int | None = None) -> tuple[Any, ...]: ...


class VisionRuntimePort(Protocol):
    """What the chatbot specialist needs from the LLM runtime handle.

    The handle is a *runtime* concern (backend, lock, lease) — the prompt
    building lives behind :class:`PromptBuilder`. Declaring the one flag the
    specialist reads keeps the boundary typed: a collaborator typed ``Any``
    cannot be checked, and an unchecked collaborator is exactly how a chat
    route ends up calling a method that does not exist (ADR 0029).
    """

    is_vision_model: bool


class AgentTaskExecutor(Protocol):
    def __call__(self, task: AgentTask) -> dict[str, Any]: ...

ToolEventSink = Callable[[ToolRequest, ToolResult], None]


class PendingActionPort(Protocol):
    """Dict-like pending-approval store the agent layer runs against (ADR 0003).

    The composition root injects the file-backed adapter
    (``adapters.persistence.PendingActionStore``); tests and embeds fall back
    to the agent layer's ``MemoryPendingStore``. Agents never construct or
    import either implementation — they only ever see this contract, which is
    what removed the last ``agents → adapters`` edge from the codebase.
    """

    #: On-disk location; ``""`` when no file backs the store.
    path: str

    def __setitem__(self, key: str, value: dict[str, Any]) -> None: ...
    def get(self, key: str, default: dict[str, Any] | None = None) -> dict[str, Any] | None: ...
    def pop(self, key: str, default: dict[str, Any] | None = None) -> dict[str, Any] | None: ...
    def clear(self) -> None: ...
    def load(self) -> None: ...
    def save(self) -> None: ...
    def cleanup(self, cutoff: float) -> bool: ...


class ShortTermMemoryPort(Protocol):
    """Bounded per-session conversation buffer (ADR 0043).

    The client ships the history it still holds on every request; when that
    history is shorter than what the server already saw, this buffer fills the
    gap so a run does not lose its context. The buffer is bounded — a
    long-lived session must not grow without limit — and the eviction rule
    belongs to the implementation (``TaskCoordinator``-style), not to the
    caller. In-process by default; a durable implementation would satisfy the
    same contract without the agent layer noticing.
    """

    def record(self, session_id: str, messages: Sequence[Mapping[str, str]]) -> None: ...
    def recent(self, session_id: str) -> list[dict[str, str]]: ...
    def forget(self, session_id: str) -> None: ...
    def cleanup(self, cutoff: float) -> int: ...


class LongTermMemoryPort(Protocol):
    """Durable long-term memory store (ADR 0043).

    The store is dumb on purpose: it persists already-validated
    :class:`~ai_assistant.domain.memory.MemoryRecord` rows and hands candidate
    sets back. Ranking (keyword + recency), the redaction gate and the
    sealed-at-rest encoding are decisions the owner layers make — the adapter
    only ever sees records the service approved, sealed by the composition
    root's cipher. ``kinds`` is a plain ``frozenset`` because rehydrated rows
    cross a trust boundary: the caller filters with ``is_memory_kind`` first.
    """

    def add(self, record: MemoryRecord) -> None: ...
    def candidates(self, owner: str, kinds: frozenset[str], *, limit: int = 200) -> list[MemoryRecord]: ...
    def prune(self, retention_days: int) -> int: ...
    def close(self) -> None: ...


class StorageCryptoError(RuntimeError):
    """Base class for at-rest storage failures (ADR 0027/0028).

    The exception lives in ``ports`` so a layer that only holds a
    :class:`FieldCipher` can react to a failed seal without importing the
    adapter that implements it.
    """


class StorageKeyError(StorageCryptoError):
    """The key is missing, malformed or does not match the ciphertext."""


class StorageDecryptionError(StorageCryptoError):
    """A sealed field failed authentication or could not be decoded."""


#: The at-rest envelope marker (ADR 0027/0028/0029). It is a *contract*, not
#: an implementation detail: the persistence adapters seal with it, the
#: kernel helpers recognise it, and the Qt client produces the same bytes.
#: Keeping the literal here is what lets a kernel module (which may not
#: import an adapter) tell a sealed file from a plaintext one.
SEALED_BINARY_PREFIX = b"enc:v1:"

#: The associated-data context of a chat attachment file. The Qt client seals
#: the attachment it stores with exactly this context, so the AI service can
#: open it with the platform at-rest key (ADR 0029). It is a contract between
#: two processes, which is why it lives here.
CHAT_ATTACHMENT_CONTEXT = "chat.attachment"


class FieldCipher(Protocol):
    """Authenticating cipher for state that reaches local disk (ADR 0027/0028).

    The composition root loads one key and injects the cipher into every
    adapter that persists user content: the task store, the pending-action
    store and the RAG cache. The protocol lives in ``ports`` so those adapters
    and the ``rag`` layer share one inward-facing contract instead of
    importing each other.

    ``context`` names the field being sealed. It becomes the AEAD associated
    data, so a ciphertext moved to a different field fails authentication
    instead of being read as if it belonged there.
    """

    def seal(self, value: str, *, context: str) -> str:
        """Return an authenticated envelope for ``value`` bound to ``context``."""
        ...

    def open(self, value: str, *, context: str) -> str:
        """Return the plaintext, accepting legacy unsealed values unchanged."""
        ...

    def is_sealed(self, value: str) -> bool:
        """Return whether ``value`` already carries the sealed envelope."""
        ...

    def seal_bytes(self, value: bytes, *, context: str) -> bytes:
        """Return an authenticated binary envelope for ``value``."""
        ...

    def open_bytes(self, value: bytes, *, context: str) -> bytes:
        """Return the plaintext bytes, refusing anything not sealed."""
        ...
