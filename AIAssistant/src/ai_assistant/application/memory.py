"""Long-term memory use-cases: recall before a run, consolidate after it.

One service owns both directions of the durable memory loop (ADR 0043):
``recall`` ranks what the store holds for this run's task, ``consolidate_run``
decides what a *completed* run is worth remembering. Both directions are
fail-closed — a store that throws, a record the redaction gate cannot make
safe, or a kind outside the vocabulary is dropped; a memory problem never
delays, fails or steers a run into territory the guards did not approve.

The store itself is injected through ``LongTermMemoryPort``: the composition
root wires the sealed SQLite adapter, tests and embeds use a stub or nothing
at all (``None`` = memory off).
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ..domain.governance import has_sensitive_content, redact_value
from ..domain.memory import (
    MEMORY_KINDS,
    MemoryRecord,
    build_run_records,
    is_memory_kind,
    recall_rank,
    task_signature,
)
from ..ports import LongTermMemoryPort

logger = logging.getLogger("ai_assistant.application.memory")


class MemoryService:
    """The single door the agent layer has to long-term memory (ADR 0043).

    ``owner`` is **required** and names the namespace every read and write is
    scoped to. The service used to default it to ``"default"`` and never look
    at the caller, which made every A2A tenant share one namespace: tenant B
    could recall tenant A's runs and a same-task write replaced A's plan. A
    deployment with more tenants than the local desktop passes the tenant to
    :meth:`recall`, :meth:`procedural_plan` and :meth:`consolidate_run`, so
    the owner travels with the call instead of with the process.
    """

    def __init__(self, store: LongTermMemoryPort | None, *, owner: str,
                 top_k: int = 5, retention_days: int = 0,
                 retention_max_bytes: int = 0) -> None:
        self._store = store
        self._owner = self._validated_owner(owner)
        self._top_k = max(0, int(top_k))
        self._retention_days = max(0, int(retention_days))
        self._retention_max_bytes = max(0, int(retention_max_bytes))

    @staticmethod
    def _validated_owner(owner: str) -> str:
        """Normalise an owner, refusing the empty namespace.

        An empty owner is not "no owner" — it is a namespace no record was
        written under, so a call that reached the store with it would quietly
        read nothing while consolidation wrote unreachable rows. Fail at the
        call that got it wrong instead of hiding it in the data.
        """
        resolved = owner.strip()
        if not resolved:
            msg = "long-term memory requires an owner"
            raise ValueError(msg)
        return resolved

    def _resolve_owner(self, owner: str | None) -> str:
        """The owner of this call: the caller's override, else the configured one."""
        return self._owner if owner is None else self._validated_owner(owner)

    @property
    def enabled(self) -> bool:
        """True when a run can actually recall something."""
        return self._store is not None and self._top_k > 0

    def recall(self, query: str, *, owner: str | None = None) -> list[MemoryRecord]:
        """The records worth injecting for *query*, best first.

        Fails closed in three places, each pinned by a test: no store (or a
        zero ``top_k``) recalls nothing; a store that raises is logged and
        recalls nothing (memory must never fail a run); a candidate whose
        kind is outside the vocabulary is dropped, whatever the store
        filtered on — the vocabulary is re-derived here, at the trust
        crossing, exactly like ``validate_steps`` does for the step log
        (ADR 0019).

        ``owner`` scopes the lookup to one caller. ``None`` means the owner
        the service was built with; an A2A execution passes the task's
        tenant so two tenants never see each other's records.
        """
        if self._store is None or self._top_k <= 0:
            return []
        target = self._resolve_owner(owner)
        try:
            candidates = self._store.candidates(target, MEMORY_KINDS)
        except Exception as error:
            logger.warning("Long-term memory recall failed: %s", error)
            return []
        trusted = [record for record in candidates if is_memory_kind(record.get("kind"))]
        return cast(list[MemoryRecord], recall_rank(trusted, query, self._top_k))

    def procedural_plan(self, task: str, *, owner: str | None = None) -> list[str] | None:
        """The remembered plan written under *task*'s exact signature.

        Procedural memory is an exact-key lookup, not a rank (ADR 0043 §7):
        the record was written under ``task_signature(task)`` and only that
        signature reads it back — same signature, same recipe; anything else
        yields ``None`` and the planner runs as usual. Fails closed like
        ``recall``: memory off, a store that raises, a row whose kind is not
        ``procedural`` (the store's kind filter is a hint, not a guarantee),
        metadata that is not a mapping, or text that does not split into
        steps all yield ``None`` rather than a plan shaped by guesswork.
        """
        if self._store is None or self._top_k <= 0:
            return None
        target = self._resolve_owner(owner)
        signature = task_signature(task)
        try:
            candidates = self._store.candidates(target, frozenset({"procedural"}))
        except Exception as error:
            logger.warning("Procedural memory lookup failed: %s", error)
            return None
        for record in candidates:
            if record.get("kind") != "procedural":
                continue
            metadata = record.get("metadata")
            if not isinstance(metadata, Mapping) or metadata.get("task_signature") != signature:
                continue
            text = record.get("text")
            if not isinstance(text, str):
                continue
            steps = [part.strip() for part in text.split(" | ") if part.strip()]
            if steps:
                return steps
        return None

    def consolidate_run(self, task: str, session_id: str,
                        steps: Sequence[Mapping[str, Any]], *,
                        owner: str | None = None) -> int:
        """Persist what a completed run is worth remembering; return the count.

        The redaction gate decides, in this order (ADR 0043 §5): the text is
        passed through ``redact_value`` (credential-shaped material masked),
        and a record that is **still** sensitive after that is dropped whole —
        there is no "store it and scrub on read", because a record that
        survives in the store is a record a future recall can leak straight
        back into a prompt. A write that raises is logged and counted as a
        miss; consolidation runs on the response path and must not turn a
        succeeded run into a failed one.

        ``owner`` scopes the write exactly like :meth:`recall`: an A2A run
        writes under its tenant, so its records carry the same namespace the
        next run of that tenant will read. A successful write also runs the
        configured retention policy, because a desktop install that only
        pruned at startup would grow between restarts — a prune failure is
        contained like every other memory failure.
        """
        if self._store is None:
            return 0
        target = self._resolve_owner(owner)
        stored = 0
        for record in build_run_records(task, session_id, list(steps), target):
            safe_text = redact_value(record["text"])
            if not isinstance(safe_text, str) or has_sensitive_content(safe_text):
                logger.info("Dropping sensitive memory record %s", record["id"])
                continue
            record["text"] = safe_text
            try:
                self._store.add(record)
            except Exception as error:
                logger.warning("Long-term memory write failed for %s: %s",
                               record["id"], error)
                continue
            stored += 1
        if stored and self._retention_days > 0:
            try:
                self._store.prune(self._retention_days,
                                  max_bytes=self._retention_max_bytes)
            except Exception as error:
                logger.warning("Long-term memory retention failed: %s", error)
        return stored


__all__ = ["MemoryService"]
