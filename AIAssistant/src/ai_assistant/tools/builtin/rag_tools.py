"""RAG tools for retrieval."""
from __future__ import annotations

from typing import Any


def tool_rag_search(params: dict[str, Any]) -> dict[str, Any]:
    """Dynamic RAG search tool — called by the agent at runtime.
    Note: Requires a RAG engine to be injected or used via globals in the caller layer.
    For this module, it forwards to the old rag module for compatibility until rag is fully integrated.
    """
    from ai_assistant.legacy import rag_module
    
    query = params.get("query", "").strip()
    if not query:
        return {"error": "Query không được để trống."}
    top_k = min(int(params.get("top_k", 5)), 10)
    
    if not hasattr(rag_module, "get_context"):
        return {"error": "RAG index chưa sẵn sàng."}
    try:
        doc_ctx, code_ctx, _ = rag_module.get_context(query, result_k=top_k)
        results = []
        if doc_ctx:
            results.append({"source": "documentation", "content": doc_ctx[:3000]})
        if code_ctx:
            results.append({"source": "source_code", "content": code_ctx[:3000]})
        if not results:
            return {"query": query, "found": False, "message": "Không tìm thấy kết quả phù hợp."}
        return {"query": query, "found": True, "top_k": top_k, "results": results}
    except Exception as e:
        return {"error": f"Lỗi RAG search: {e}"}
