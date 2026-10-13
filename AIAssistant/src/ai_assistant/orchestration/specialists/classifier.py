"""Sprint 2: classifier replaces keyword heuristics for task routing."""
from __future__ import annotations


def classify_task(task: str) -> str:
    """Simple rule-based classifier; replace with model when golden set ready."""
    t = task.lower()
    if any(w in t for w in ("sửa", "lỗi", "fix", "build", "compile", "test", "runnable")):
        return "coding"
    if any(w in t for w in ("tìm", "tài liệu", "hướng dẫn", "search", "read")):
        return "rag"
    return "direct"
