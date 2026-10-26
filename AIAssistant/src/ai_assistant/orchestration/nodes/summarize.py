"""Observation Summarization node/helpers.

These utilities compact the conversation history and extract citations,
ensuring the ReAct prompt doesn't overflow the LLM context window.
"""
from __future__ import annotations

import os

from ai_assistant.domain.memory import summarize_messages as summarize_messages

from ..state import Step, has_step_type


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
