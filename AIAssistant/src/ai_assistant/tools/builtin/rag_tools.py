"""RAG tools for retrieval."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_assistant.ports import RagContextPort

#: Shape every handler in the tool registry must satisfy.
Handler = Callable[[dict[str, Any]], dict[str, Any]]


def rag_search_tool(provider: RagContextPort | None) -> Handler:
    """Build the ``rag_search`` handler bound to a retrieval port.

    The composition root injects the index it owns (ADR 0001); this module
    never imports the bootstrap module the tool used to reach for at call time,
    which is what made the handler untestable without faking a whole package in
    ``sys.modules``. A registry built without a provider reports the index as
    unavailable instead of raising.
    """

    def tool_rag_search(params: dict[str, Any]) -> dict[str, Any]:
        query = params.get("query", "").strip()
        if not query:
            return {"error": "Query không được để trống."}
        top_k = min(int(params.get("top_k", 5)), 10)
        if provider is None:
            return {"error": "RAG index chưa sẵn sàng."}
        try:
            doc_ctx, code_ctx, _ = provider.get_context(query, result_k=top_k)
        except Exception as e:
            return {"error": f"Lỗi RAG search: {e}"}
        results = []
        if doc_ctx:
            results.append({"source": "documentation", "content": doc_ctx[:3000]})
        if code_ctx:
            results.append({"source": "source_code", "content": code_ctx[:3000]})
        if not results:
            return {"query": query, "found": False, "message": "Không tìm thấy kết quả phù hợp."}
        return {"query": query, "found": True, "top_k": top_k, "results": results}

    return tool_rag_search
