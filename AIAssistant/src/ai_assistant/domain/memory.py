"""Memory: short-term context policy and long-term record contracts (ADR 0043).

Two halves, one owner:

* **Short-term (STM)** — which conversation turns survive into the next model
  call. ``trim_history`` (drop-oldest by token budget), ``summarize_messages``
  (character-budget compaction protecting system/user/recency) and the budget
  constants behind them used to live in three modules (``llm.prompts``,
  ``orchestration.nodes.summarize``, ``orchestration.state``). Those decisions
  are about the agent's context, not about a transport or a framework node, so
  the domain owns them here; the historical import paths keep working as thin
  delegates (the ADR 0019 ``Step`` precedent).

* **Long-term (LTM)** — three kinds of durable records: *semantic* (facts),
  *episodic* (what a past run did) and *procedural* (how it did it). This
  module owns the vocabulary, the record shape and the deterministic ranking
  used to recall them. Persistence, sealing and redaction stay in the layers
  that own IO — an LTM record is rehydrated JSON crossing a trust boundary, so
  the runtime gate :func:`is_memory_kind` re-derives from the vocabulary
  itself, exactly like ``validate_steps`` does for the step log (ADR 0019).

Framework-free and stdlib-only: no IO, no LLM, no LangGraph — the import
contracts forbid the domain from reaching any of them.
"""
from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

from .steps import has_step_type

# ── Long-term memory vocabulary ────────────────────────────────────────────────

#: The three long-term memory kinds (ADR 0043). Closed on purpose: a record
#: whose ``kind`` is not one of these would be recalled by a path that has no
#: policy for it.
MemoryKind = Literal["semantic", "episodic", "procedural"]

MEMORY_KINDS: frozenset[str] = frozenset({"semantic", "episodic", "procedural"})


class MemoryRecord(TypedDict):
    """One durable memory. ``text`` is the payload the recall path injects.

    ``owner`` scopes the record to an account (the persistence layer filters
    on it); ``salience`` (0..1) is a writer-provided prior the ranker may
    weigh; ``created_at``/``last_used_at`` are unix seconds and drive the
    recency half of the ranking.
    """

    id: str
    kind: MemoryKind
    owner: str
    text: str
    created_at: float
    last_used_at: float
    salience: float
    metadata: dict[str, Any]


def is_memory_kind(value: object) -> bool:
    """True when *value* is a vocabulary kind — the runtime twin of
    :data:`MemoryKind`.

    Rehydrated records cross the trust boundary as plain JSON, so the
    compiler cannot see their ``kind``; callers drop the record when this
    returns ``False`` rather than persisting or recalling an unknown kind.
    """
    return isinstance(value, str) and value in MEMORY_KINDS


#: Recall ranking half-life in days: a record loses half of its keyword-score
#: weight after this age. Deterministic given ``now``, so the ranker is testable
#: without a clock dependency.
MEMORY_HALF_LIFE_DAYS = 30.0


def _normalize(text: str) -> str:
    """Casefold, strip diacritics, collapse separators — accent-insensitive so
    ``dự án`` matches ``du an`` the way the rest of the codebase normalises."""
    decomposed = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return " ".join("".join(ch if ch.isalnum() else " " for ch in stripped).split())


def recall_rank(records: Sequence[Mapping[str, Any]], query: str, top_k: int,
                *, now: float | None = None) -> list[Mapping[str, Any]]:
    """Rank *records* for *query* by keyword overlap weighted by recency.

    Keyword + recency is the whole model (ADR 0043 v1 — no embeddings): the
    score is ``overlap * (0.5 + 0.5 * decay) + 0.01 * salience * decay`` where
    *decay* halves every :data:`MEMORY_HALF_LIFE_DAYS` days.

    Rules, each pinned by a test:

    * ``top_k <= 0`` returns ``[]`` — the caller asked for nothing;
    * a non-empty query with **no** keyword hit is dropped instead of ranked
      last: injecting an unrelated "memory" into the prompt is noise, and a
      ranker that returned zero-overlap rows would fire it constantly;
    * an **empty** query is a recency browse (rank by recency + salience),
      which is what "what do we remember about this session" asks for;
    * ties break by ``created_at`` descending, then ``id`` ascending, so the
      order never depends on dict or set iteration.
    """
    if top_k <= 0:
        return []
    moment = time.time() if now is None else now
    query_tokens = set(_normalize(query).split())
    scored: list[tuple[float, float, str, Mapping[str, Any]]] = []
    for record in records:
        text_tokens = set(_normalize(str(record.get("text", ""))).split())
        overlap = (len(query_tokens & text_tokens) / len(query_tokens)
                   if query_tokens else 0.0)
        if query_tokens and overlap == 0.0:
            continue
        created_at = float(record.get("created_at", moment))
        age_days = max(0.0, (moment - created_at) / 86400.0)
        decay = 0.5 ** (age_days / MEMORY_HALF_LIFE_DAYS)
        salience = float(record.get("salience", 0.0))
        score = overlap * (0.5 + 0.5 * decay) + 0.01 * salience * decay
        scored.append((score, created_at, str(record.get("id", "")), record))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [record for _, _, _, record in scored[:top_k]]


# ── Short-term policy (ADR 0043) ──────────────────────────────────────────────
#
# The numbers keep their historical values (``orchestration.state`` re-exports
# them); only the address changed.

#: Summarise observations after this many tool calls.
OBSERVATION_SUMMARY_THRESHOLD = 4   # summarise observations after N tool calls
#: Keep last N tool-result messages through a compaction.
OBSERVATION_KEEP_LAST = 10          # keep last N tool-result messages
#: Character budgets enforced by :func:`summarize_messages`.
CONTEXT_SYSTEM_LIMIT = 3500         # reason node adds ~1k REASONER_PROMPT on top
CONTEXT_MESSAGE_LIMIT = 4000        # user + evidence must both fit
CONTEXT_TOTAL_LIMIT = 11000         # hard compaction limit before calling LLM
#: Token budget for :func:`trim_history` (drop-oldest window).
TRIM_DEFAULT_MAX_TOKENS = 2000
#: Characters per estimated token (the historical default of the trimmer).
CHARS_PER_TOKEN_DEFAULT = 2.2


def estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN_DEFAULT) -> int:
    """Rough token count for budget arithmetic — never used as a real tokenizer."""
    return int(len(text) / chars_per_token)


def trim_history(messages: list[dict[str, Any]], max_tokens: int = TRIM_DEFAULT_MAX_TOKENS,
                 chars_per_token: float = CHARS_PER_TOKEN_DEFAULT) -> list[dict[str, Any]]:
    """Drop oldest turns until the window fits ``max_tokens``; keep at least one.

    The drop-oldest half of the short-term policy. The character compaction
    half is :func:`summarize_messages` — one module owns both so the two
    budgets cannot drift apart (ADR 0043).
    """
    trimmed = list(messages)
    total = sum(estimate_tokens(m.get("content", ""), chars_per_token)
                for m in trimmed if isinstance(m.get("content"), str))
    while total > max_tokens and len(trimmed) > 1:
        removed = trimmed.pop(0)
        content = removed.get("content", "")
        if isinstance(content, str):
            total -= estimate_tokens(content, chars_per_token)
    return trimmed


def summarize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep a bounded, useful context for the next Reasoner turn.

    The compactor preserves the original task and the latest observations
    while enforcing a character budget independent of the number or domain of
    tools used. Body moved verbatim from ``orchestration.nodes.summarize``
    (ADR 0043): the orchestration entry point delegates here so its import
    path and tests stay put.
    """
    if len(messages) <= 4 and sum(len(m.get("content", "")) for m in messages) <= CONTEXT_TOTAL_LIMIT:
        return messages

    def clipped(message: dict[str, str], limit: int) -> dict[str, str]:
        content = message.get("content", "")
        if len(content) <= limit:
            return message
        marker = "\n… [context clipped]"
        return {**message, "content": content[:max(0, limit - len(marker))] + marker}

    first_system_index = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
    first_user_index = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
    recent_count = OBSERVATION_KEEP_LAST * 2
    recent_indices = list(range(max(0, len(messages) - recent_count), len(messages)))
    protected = set(recent_indices)
    if first_system_index is not None:
        protected.add(first_system_index)
    if first_user_index is not None:
        protected.add(first_user_index)

    middle = [messages[i] for i in range(len(messages)) if i not in protected]
    parts = [
        f"[{m.get('role', '')}] {m.get('content', '')[:300]}"
        for m in middle
    ]
    result: list[dict[str, str]] = []
    if first_system_index is not None:
        result.append(clipped(messages[first_system_index], CONTEXT_SYSTEM_LIMIT))
    if first_user_index is not None and first_user_index != first_system_index:
        result.append(clipped(messages[first_user_index], CONTEXT_MESSAGE_LIMIT))
    if parts:
        result.append({
            "role": "system",
            "content": "[Tom tat cac buoc da thuc hien]\n" + "\n".join(parts) + "\n[Het tom tat]",
        })
    for index in recent_indices:
        if index not in {first_system_index, first_user_index}:
            result.append(clipped(messages[index], CONTEXT_MESSAGE_LIMIT))

    # Preserve the latest observation if the fixed total budget is still tight.
    total = sum(len(m.get("content", "")) for m in result)
    if total > CONTEXT_TOTAL_LIMIT:
        for index in range(1, len(result)):
            message = result[index]
            excess = total - CONTEXT_TOTAL_LIMIT
            content = message.get("content", "")
            new_length = max(240, len(content) - excess)
            result[index] = clipped(message, new_length)
            total = sum(len(m.get("content", "")) for m in result)
            if total <= CONTEXT_TOTAL_LIMIT:
                break
    return result


def conversation_turns(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Keep only the turns a client would ship back (``user``/``assistant``).

    The system prompt is rebuilt on every run, and tool observations are
    re-derivable from the run itself: neither belongs in the short-term buffer
    a request tops its history up from (ADR 0043).
    """
    return [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
        for m in messages
        if m.get("role") in {"user", "assistant"}
    ]


# ── Long-term extraction (ADR 0043 §4/§7) ─────────────────────────────────────

#: Semantic records are the most opinionated kind (they steer future answers),
#: so one run may contribute at most this many.
MAX_SEMANTIC_PER_RUN = 3

#: Preference-shaped statements, Vietnamese and English, accent-tolerant on the
#: Vietnamese side. Deliberately narrow: only an explicit preference verb may
#: turn a sentence into a durable fact — a wide pattern would memorise
#: everything the user says and drown the store in noise.
_PREFERENCE_RE = re.compile(
    r"(?:th[ií]ch|mu[oố]n|prefer(?:s|red)?|likes?|wants?)\s+([^.;\n]{3,120})",
    re.IGNORECASE,
)


def task_signature(task: str) -> str:
    """Stable identity of a task — the procedural-memory key (ADR 0043 §7).

    Normalised (accents stripped, casefolded, separators collapsed) and
    hashed, so ``"Tìm file main.cpp"`` and ``"tim file main.cpp"`` share one
    plan while a genuinely different task gets a different key. 64 hex chars:
    long enough to never collide in practice, short enough to key a row.
    """
    return hashlib.sha256(_normalize(task).encode("utf-8")).hexdigest()


def _preference_facts(text: str) -> list[str]:
    """The preference statements *text* makes, at most
    :data:`MAX_SEMANTIC_PER_RUN` of them."""
    facts = [
        match.group(1).strip()
        for match in _PREFERENCE_RE.finditer(text)
        if match.group(1).strip()
    ]
    return facts[:MAX_SEMANTIC_PER_RUN]


def build_run_records(
    task: str,
    session_id: str,
    steps: Sequence[Mapping[str, Any]],
    owner: str,
    *,
    now: float | None = None,
) -> list[MemoryRecord]:
    """Extract the durable records of one completed run (ADR 0043 §4).

    Deterministic, no LLM, no model lease: the records are read off the step
    log the run already produced.

    * **episodic** — always one: what was asked, whether it answered and
      which tools it used;
    * **procedural** — only when the run produced a plan: the plan itself,
      keyed by :func:`task_signature` so a future run of the same task can
      reuse it;
    * **semantic** — up to :data:`MAX_SEMANTIC_PER_RUN` preference facts
      found in the task or the final answer.

    Only *completed* runs reach here — the caller gates on the status, so a
    failed or parked run never writes to the store (ADR 0043 §4).
    """
    moment = time.time() if now is None else now
    signature = task_signature(task)
    answer = ""
    plan_items: list[str] = []
    tools_used: list[str] = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        if has_step_type(step, "final_answer"):
            answer = str(step.get("content", ""))
        elif has_step_type(step, "plan"):
            raw_plan = step.get("steps")
            if isinstance(raw_plan, list):
                plan_items = [str(item) for item in raw_plan]
        elif has_step_type(step, "tool_call"):
            tool = step.get("tool")
            if isinstance(tool, str) and tool:
                tools_used.append(tool)
    records: list[MemoryRecord] = [{
        "id": f"ep-{signature[:16]}-{int(moment)}",
        "kind": "episodic",
        "owner": owner,
        "text": f"Task: {task[:160]}" + (f" — answer: {answer[:240]}" if answer else ""),
        "created_at": moment,
        "last_used_at": moment,
        "salience": 0.3,
        "metadata": {"session_id": session_id, "task_signature": signature,
                     "tools": tools_used[:20]},
    }]
    if plan_items:
        # The id is derived from the signature on purpose: rerunning the same
        # task replaces its procedural record instead of piling up copies
        # (``INSERT OR REPLACE`` in the store).
        records.append({
            "id": f"proc-{signature[:24]}",
            "kind": "procedural",
            "owner": owner,
            "text": " | ".join(plan_items)[:600],
            "created_at": moment,
            "last_used_at": moment,
            "salience": 0.5,
            "metadata": {"session_id": session_id, "task_signature": signature,
                         "steps": len(plan_items)},
        })
    for fact in _preference_facts(f"{task}\n{answer}"):
        records.append({
            "id": f"sem-{hashlib.sha256(fact.encode('utf-8')).hexdigest()[:24]}",
            "kind": "semantic",
            "owner": owner,
            "text": fact,
            "created_at": moment,
            "last_used_at": moment,
            "salience": 0.6,
            "metadata": {"session_id": session_id, "task_signature": signature},
        })
    return records


def format_memory_block(records: Sequence[Mapping[str, Any]]) -> str:
    """Render recalled records as the block appended to the system prompt.

    An empty list renders ``""`` — an empty ``[Long-term memory]`` header
    would tell the model there is memory when there is none.
    """
    if not records:
        return ""
    lines = [f"- [{record.get('kind', '')}] {record.get('text', '')}" for record in records]
    return "[Long-term memory]\n" + "\n".join(lines)


def _is_window(needle: Sequence[Mapping[str, str]], haystack: Sequence[Mapping[str, str]]) -> bool:
    """True when *needle* occurs contiguously inside *haystack*.

    An empty needle is a window of anything (``haystack[0:0] == []``), which
    the generator derives without a separate branch to cover.
    """
    n, h = len(needle), len(haystack)
    return h >= n and any(haystack[i:i + n] == list(needle) for i in range(h - n + 1))


def _continuation_overlap(buffered: Sequence[Mapping[str, str]],
                          incoming: Sequence[Mapping[str, str]]) -> int:
    """Largest ``k`` where *buffered* already ends with the first ``k`` turns
    of *incoming* — the part the merge must not repeat."""
    for k in range(min(len(buffered), len(incoming)), 0, -1):
        if buffered[-k:] == incoming[:k]:
            return k
    return 0


def merge_history(buffered: Sequence[Mapping[str, str]],
                  incoming: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Top the client's shipped history up from the short-term buffer (ADR 0043).

    The client stays authoritative for everything it sends — a turn it shipped
    is never dropped — and the buffer only adds what the client no longer
    holds. Two rules produce every case, each pinned by a test:

    * *incoming* is a **window** of *buffered* → return the buffer: the
      client re-shipped a slice of what the server already holds (including a
      client that restarted its transcript and ships nothing), so neither
      duplicating nor forgetting is right;
    * otherwise → the buffer plus the part of *incoming* that does not repeat
      the buffer's tail: an empty buffer passes *incoming* through, a
      continuing conversation is appended exactly once, and a wholly new one
      is kept after the remembered context.
    """
    if _is_window(incoming, buffered):
        return [dict(m) for m in buffered]
    overlap = _continuation_overlap(buffered, incoming)
    return [dict(m) for m in buffered] + [dict(m) for m in incoming[overlap:]]


__all__ = [
    "CHARS_PER_TOKEN_DEFAULT",
    "CONTEXT_MESSAGE_LIMIT",
    "CONTEXT_SYSTEM_LIMIT",
    "CONTEXT_TOTAL_LIMIT",
    "MAX_SEMANTIC_PER_RUN",
    "MEMORY_HALF_LIFE_DAYS",
    "MEMORY_KINDS",
    "OBSERVATION_KEEP_LAST",
    "OBSERVATION_SUMMARY_THRESHOLD",
    "TRIM_DEFAULT_MAX_TOKENS",
    "MemoryKind",
    "MemoryRecord",
    "build_run_records",
    "conversation_turns",
    "estimate_tokens",
    "format_memory_block",
    "is_memory_kind",
    "merge_history",
    "recall_rank",
    "summarize_messages",
    "task_signature",
    "trim_history",
]
