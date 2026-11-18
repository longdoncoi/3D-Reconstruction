"""Core RAG domain models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkResult:
    """A semantic chunk of text or an image extracted from a document."""
    text: str
    source_path: str
    loader_type: str
    is_image: bool = False
    image_b64: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # ── Hierarchical Chunking fields ──────────────────────────────────────────
    # parent_text: the broader context that contains this chunk (e.g. the
    # enclosing class body for a method chunk, or the H1 section for a
    # sub-heading chunk).
    parent_text: str | None = None
    hierarchy_level: int = 0   # 0 = flat/root, 1 = class/H1, 2 = method/H2+


@dataclass(frozen=True, slots=True)
class RagHit:
    """One ranked, attributable piece of project evidence.

    Retrieval used to flatten all chunks into two prompt strings.  Keeping the
    identity and scores until the prompt boundary lets the agent verify that a
    result is evidence, apply an explicit no-answer gate, and cite a concrete
    source when the product policy permits it.
    """

    chunk_id: int
    text: str
    source_path: str
    loader_type: str
    rank: int
    fusion_score: float
    confidence: float
    rerank_score: float | None = None
    symbol: str | None = None
    heading: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    is_image: bool = False
    parent_text: str | None = None

    def to_dict(self, *, text_limit: int = 1_200) -> dict[str, Any]:
        """Return the bounded wire representation used by ``rag_search``."""
        payload = {
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "loader_type": self.loader_type,
            "rank": self.rank,
            "fusion_score": self.fusion_score,
            "confidence": self.confidence,
            "rerank_score": self.rerank_score,
            "symbol": self.symbol,
            "heading": self.heading,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "is_image": self.is_image,
            "text": self.text[:text_limit],
        }
        if self.parent_text is not None:
            payload["parent_text"] = self.parent_text[:text_limit]
        return payload


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Structured retrieval outcome, including an explicit no-answer state."""

    query: str
    hits: tuple[RagHit, ...]
    confidence: float
    answerable: bool
    reason: str

    @classmethod
    def empty(cls, query: str, reason: str) -> "RetrievalResult":
        return cls(query=query, hits=(), confidence=0.0, answerable=False, reason=reason)
