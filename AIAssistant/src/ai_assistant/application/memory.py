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
    """The single door the agent layer has to long-term memory (ADR 0043)."""

    def __init__(self, store: LongTermMemoryPort | None, *, top_k: int = 5,
                 owner: str = "default") -> None:
        self._store = store
        self._top_k = max(0, int(top_k))
        self._owner = owner

    @property
    def enabled(self) -> bool:
        """True when a run can actually recall something."""
        return self._store is not None and self._top_k > 0

    def recall(self, query: str) -> list[MemoryRecord]:
        """The records worth injecting for *query*, best first.

        Fails closed in three places, each pinned by a test: no store (or a
        zero ``top_k``) recalls nothing; a store that raises is logged and
        recalls nothing (memory must never fail a run); a candidate whose
        kind is outside the vocabulary is dropped, whatever the store
        filtered on — the vocabulary is re-derived here, at the trust
        crossing, exactly like ``validate_steps`` does for the step log
        (ADR 0019).
        """
        if self._store is None or self._top_k <= 0:
            return []
        try:
            candidates = self._store.candidates(self._owner, MEMORY_KINDS)
        except Exception as error:
            logger.warning("Long-term memory recall failed: %s", error)
            return []
        trusted = [record for record in candidates if is_memory_kind(record.get("kind"))]
        return cast(list[MemoryRecord], recall_rank(trusted, query, self._top_k))

    def procedural_plan(self, task: str) -> list[str] | None:
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
        signature = task_signature(task)
        try:
            candidates = self._store.candidates(self._owner, frozenset({"procedural"}))
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
                        steps: Sequence[Mapping[str, Any]]) -> int:
        """Persist what a completed run is worth remembering; return the count.

        The redaction gate decides, in this order (ADR 0043 §5): the text is
        passed through ``redact_value`` (credential-shaped material masked),
        and a record that is **still** sensitive after that is dropped whole —
        there is no "store it and scrub on read", because a record that
        survives in the store is a record a future recall can leak straight
        back into a prompt. A write that raises is logged and counted as a
        miss; consolidation runs on the response path and must not turn a
        succeeded run into a failed one.
        """
        if self._store is None:
            return 0
        stored = 0
        for record in build_run_records(task, session_id, list(steps), self._owner):
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
        return stored


__all__ = ["MemoryService"]
