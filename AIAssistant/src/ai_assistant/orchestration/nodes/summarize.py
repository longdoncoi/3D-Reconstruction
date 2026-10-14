"""Observation Summarization node/helpers.

These utilities compact the conversation history and extract citations,
ensuring the ReAct prompt doesn't overflow the LLM context window.
"""
from __future__ import annotations

import os

from ..state import (
    CONTEXT_MESSAGE_LIMIT,
    CONTEXT_SYSTEM_LIMIT,
    CONTEXT_TOTAL_LIMIT,
    OBSERVATION_KEEP_LAST,
    Step,
    has_step_type,
)


def summarize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep a bounded, useful context for the next Reasoner turn.

    The compactor preserves the original task and the latest observations
    while enforcing a character budget independent of the number or domain of tools used.
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


def format_code_citation(messages: list[dict[str, str]], steps: list[Step]) -> str | None:
    """Return a complete, syntax-highlighted citation for symbol requests."""
    user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    normalized = user_msg.casefold()
    # A request that merely names a function may ask for analysis, review or a
    # fix.  Only an explicit citation request may replace the model's final
    # explanation with a source fence.
    citation_terms = ("trích dẫn", "trich dan", "cite", "quote", "show code", "source code")
    if not any(term in normalized for term in citation_terms):
        return None
    for step in reversed(steps):
        if not has_step_type(step, "tool_result") or step.get("tool") != "read_file":
            continue
        result = step.get("result", {})
        content = result.get("content") if isinstance(result, dict) else None
        if not isinstance(content, str) or not content.strip() or result.get("error"):
            continue
        path = str(result.get("path", "source"))
        ext = os.path.splitext(path)[1].lower()
        language = {".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".h": "cpp",
                    ".hpp": "cpp", ".c": "c", ".py": "python"}.get(ext, "text")
        showing = str(result.get("showing", ""))
        source_ref = f"`{path}`" + (f" ({showing})" if showing else "")
        return f"Đoạn mã đầy đủ của hàm được trích dẫn từ {source_ref}:\n\n```{language}\n{content.rstrip()}\n```"
    return None
