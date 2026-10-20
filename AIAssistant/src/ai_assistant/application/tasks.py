from __future__ import annotations

import contextvars
import threading
from collections.abc import Mapping
from time import monotonic
from typing import Any

from ..domain.governance import redact_value
from ..domain.security import DataClassification
from ..domain.tasks import AgentTask, TaskStatus
from ..ports import AgentTaskExecutor, TaskStore
from .coordination import TaskCoordinator


class TaskService:
    """Durable A2A task lifecycle with explicit, observable state changes."""

    def __init__(self, store: TaskStore, executor: AgentTaskExecutor, capabilities: frozenset[str],
                 coordinator: TaskCoordinator | None = None) -> None:
        self._store = store
        self._executor = executor
        self._capabilities = capabilities
        # Shared with AgentService by the composition root so ``cancel`` can
        # stop the running LangGraph loop instead of only writing the DB row.
        self._coordinator = coordinator
        self._workers: set[threading.Thread] = set()
        self._worker_lock = threading.RLock()
        self._closing = False
        self._store_closed = False
        # A previous process cannot have left a live worker behind: reconcile
        # its non-terminal rows before this service accepts new work. Detection
        # is class-level so duck-typed test stores keep their old behaviour.
        reconcile = getattr(type(self._store), "fail_orphaned", None)
        if callable(reconcile):
            reconcile(self._store)

    def submit(self, capability: str, message: str, metadata: Mapping[str, Any] | None = None,
               context_id: str | None = None, classification: DataClassification = DataClassification.INTERNAL,
               *, owner: str = "default") -> AgentTask:
        task = AgentTask.new(capability, message, context_id=context_id, metadata=dict(metadata or {}),
                             classification=classification, owner=owner)
        if capability not in self._capabilities:
            task = task.transition(TaskStatus.REJECTED, error=f"Unsupported capability: {capability}")
            self._store.create(task)
            self._store.append_event(task.id, "status", {"status": task.status, "error": task.error})
            return task
        self._store.create(task)
        self._store.append_event(task.id, "status", {"status": task.status})
        self._start(task.id)
        return task

    def get(self, task_id: str, *, owner: str = "default") -> AgentTask | None:
        task = self._store.get(task_id)
        if task is None or task.owner != owner:
            return None
        return task

    @property
    def capabilities(self) -> frozenset[str]:
        return self._capabilities

    def events(self, task_id: str, since: int = 0, *, owner: str = "default") -> tuple[dict[str, Any], ...]:
        """Return only the events after the store-native ``since`` cursor.

        SQLite pages by ``rowid`` (cost is O(new events)); stores without a
        cursor are read once and sliced by count, so the SSE poll loop stops
        re-reading the whole log on every tick either way. Ownership is
        checked before streaming so a subscriber that only holds a task id
        cannot read another tenant's event log.
        """
        task = self._store.get(task_id)
        if task is None or task.owner != owner:
            return ()
        store_events = self._store.events
        try:
            events = tuple(store_events(task_id, since=since))
        except TypeError:
            # Duck-typed store with the old signature: read once, slice by count.
            events = tuple(store_events(task_id))[since:]
        return events

    def list(self, *, context_id: str | None = None, limit: int = 100,
             owner: str = "default") -> tuple[AgentTask, ...]:
        return tuple(self._store.list(context_id=context_id, limit=limit, owner=owner))

    def prune(self, retention_days: int) -> int:
        """Delete terminal tasks past the retention window (0 disables it)."""
        if retention_days <= 0:
            return 0
        prune = getattr(type(self._store), "prune", None)
        if not callable(prune):
            return 0
        return int(prune(self._store, retention_days))

    def close(self, timeout_seconds: float = 5.0) -> None:
        """Gracefully drain active workers before releasing durable storage."""
        with self._worker_lock:
            self._closing = True
            if self._workers or self._store_closed:
                workers = tuple(self._workers)
            else:
                self._close_store()
                return
        deadline = monotonic() + max(0.0, timeout_seconds)
        for worker in workers:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            worker.join(remaining)
        with self._worker_lock:
            if not self._workers:
                self._close_store()

    def cancel(self, task_id: str, *, owner: str = "default") -> AgentTask | None:
        task = self._store.get(task_id)
        if task is None or task.owner != owner:
            return None
        if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED, TaskStatus.REJECTED}:
            return None
        cancelled = task.transition(TaskStatus.CANCELED)
        if not self._save_if_status(cancelled, task.status):
            # Lost the race: the worker moved the task first (CAS guard).
            return None
        self._store.append_event(task_id, "status", {"status": cancelled.status})
        if self._coordinator is not None:
            # Session id for an A2A run is the task id (LangGraph orchestrator),
            # so this flips the flag the runner's cancel_checker polls: the
            # LangGraph loop — including already-approved tool execution — stops.
            self._coordinator.cancel(task_id, create_if_missing=True)
        return cancelled

    def _save_if_status(self, task: AgentTask, expected: TaskStatus) -> bool:
        """Persist ``task`` only if its stored status is still ``expected``.

        ``SqliteTaskStore`` answers with a single guarded UPDATE; duck-typed
        test stores without the method fall back to read-then-write. Detection
        is class-level so mock doubles keep their historical ``save`` recording
        behaviour instead of being mistaken for a real CAS implementation.
        """
        if callable(getattr(type(self._store), "save_if_status", None)):
            return bool(self._store.save_if_status(task, expected))
        current = self._store.get(task.id)
        if current is None or current.status != expected:
            return False
        self._store.save(task)
        return True

    def resume(self, task_id: str, input_data: Mapping[str, Any], *, owner: str = "default") -> AgentTask | None:
        """Resume an explicitly paused task with a durable user or desktop response."""
        task = self._store.get(task_id)
        if task is None or task.owner != owner:
            return None
        if task.status != TaskStatus.INPUT_REQUIRED:
            return None
        continuation = task.metadata.get("agent_run.continuation")
        if not isinstance(continuation, dict):
            return None
        metadata = dict(task.metadata)
        metadata["agent_run.continuation"] = continuation
        metadata["agent_run.resume"] = dict(input_data)
        resumed = task.with_metadata(metadata).transition(TaskStatus.WORKING)
        if not self._save_if_status(resumed, TaskStatus.INPUT_REQUIRED):
            # The task left input_required while we prepared the resume.
            return None
        self._store.append_event(task_id, "status", {"status": resumed.status, "resumed": True})
        self._start(task_id, already_working=True)
        return resumed

    def _start(self, task_id: str, *, already_working: bool = False) -> None:
        # A plain thread inherits no contextvars, and the OTel trace context
        # (ADR 0001: the inbound traceparent attached by the HTTP middleware)
        # lives in one. Snapshot the submitter's context here and run the
        # worker inside it, so the run's spans — including every span the
        # executor opens — join the caller's trace instead of a fresh root.
        # ``contextvars`` is stdlib, so the application layer still imports no
        # observability framework.
        submit_context = contextvars.copy_context()

        def worker() -> None:
            try:
                submit_context.run(self._run, task_id, already_working)
            finally:
                with self._worker_lock:
                    self._workers.discard(threading.current_thread())
                    if self._closing and not self._workers:
                        self._close_store()

        thread = threading.Thread(target=worker, daemon=True, name=f"a2a-{task_id[-8:]}")
        with self._worker_lock:
            if self._closing:
                msg = "Task service is shutting down"
                raise RuntimeError(msg)
            self._workers.add(thread)
        thread.start()

    def _close_store(self) -> None:
        if self._store_closed:
            return
        close = getattr(self._store, "close", None)
        if callable(close):
            close()
        self._store_closed = True

    def _run(self, task_id: str, already_working: bool = False) -> None:
        task = self._store.get(task_id)
        if task is None or task.status == TaskStatus.CANCELED:
            return
        if already_working:
            working = task
        else:
            working = task.transition(TaskStatus.WORKING)
            if not self._save_if_status(working, task.status):
                # A cancel (or another writer) won the race before the run
                # started: do not execute, do not overwrite the new status.
                return
            self._store.append_event(task_id, "status", {"status": working.status})
        try:
            result = self._executor(working)
            latest = self._store.get(task_id)
            if latest is None or latest.status == TaskStatus.CANCELED:
                return
            if result.get("status") == "input_required":
                continuation = result.get("continuation")
                if isinstance(continuation, dict):
                    metadata = dict(latest.metadata)
                    metadata["agent_run.continuation"] = continuation
                    metadata.pop("agent_run.resume", None)
                    latest = latest.with_metadata(metadata)
                public_result = {key: value for key, value in result.items() if key != "continuation"}
                waiting = latest.transition(TaskStatus.INPUT_REQUIRED, result=public_result)
                if not self._save_if_status(waiting, latest.status):
                    return
                # Events are a transport surface (SSE): they get the same
                # redaction as ``task_payload`` so the HITL continuation — which
                # carries the action id and the approved tool params — can never
                # be read back by a caller that only holds the task id.
                self._store.append_event(
                    task_id, "status",
                    {"status": waiting.status, "result": redact_value(result)},
                )
                return
            completed = latest.transition(TaskStatus.COMPLETED, result=result)
            if not self._save_if_status(completed, latest.status):
                # Cancelled while the executor ran: the CAS keeps CANCELED
                # instead of an unconditional UPDATE writing COMPLETED over it.
                return
            self._store.append_event(task_id, "artifact", {"result": redact_value(result)})
            self._store.append_event(task_id, "status", {"status": completed.status})
        except Exception as error:  # The task becomes observable failure, not a hidden fallback.
            latest = self._store.get(task_id)
            if latest is not None and latest.status != TaskStatus.CANCELED:
                failed = latest.transition(TaskStatus.FAILED, error=str(error))
                if not self._save_if_status(failed, latest.status):
                    return
                self._store.append_event(task_id, "status", {"status": failed.status, "error": failed.error})
