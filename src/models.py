from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ContentComposition(str, Enum):
    TEXT = "text"
    TABLES = "tables"
    MIXED = "mixed"


class ChunkingStrategy(str, Enum):
    NAIVE = "naive"
    STRUCTURE_AWARE = "structure_aware"


# ── Raw extraction output ──────────────────────────────────────────

class RawPage(BaseModel):
    """Output from document loaders — one per page/sheet."""
    page_number: int
    text: str
    tables: list[list[dict[str, Any]]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RawDocument(BaseModel):
    """Complete raw extraction before analysis."""
    file_name: str
    file_type: str
    pages: list[RawPage]
    total_pages: int
    extraction_timestamp: datetime = Field(default_factory=datetime.now)


# ── AI document analysis ───────────────────────────────────────────

# ── Document metadata (stored with each chunk) ────────────────────

class DocumentMetadata(BaseModel):
    """Metadata attached to every chunk for filtering and citation."""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    file_name: str
    file_type: str
    document_type: str = "Unknown"
    total_pages: int = 0


# ── Chunks ─────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """A single chunk ready for embedding and storage."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    document_id: str
    file_name: str
    document_type: str = "Unknown"
    text: str
    page_number: int | None = None
    section: str = "Unknown"
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.NAIVE
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_vectorstore_metadata(self) -> dict[str, Any]:
        """Flatten for ChromaDB metadata storage (no nested objects)."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "file_name": self.file_name,
            "document_type": self.document_type,
            "page_number": self.page_number or 0,
            "section": self.section,
            "chunking_strategy": self.chunking_strategy.value,
        }


# ── Parsed sections (intermediate between raw + chunks) ───────────

class ParsedSection(BaseModel):
    """A document section with preserved structure."""
    title: str = "Untitled"
    level: int = 1
    content: str = ""
    page_number: int | None = None
    is_table: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Query analysis ─────────────────────────────────────────────────

class QueryAnalysis(BaseModel):
    """LLM-inferred understanding of the user's question."""
    original_query: str
    intent: str = "general"
    target_document_types: list[str] = Field(default_factory=list)
    target_sections: list[str] = Field(default_factory=list)
    is_multi_document: bool = False
    is_comparison: bool = False
    keywords: list[str] = Field(default_factory=list)


# ── Retrieval ──────────────────────────────────────────────────────

class RetrievalResult(BaseModel):
    """A single retrieved chunk with its relevance score."""
    chunk: DocumentChunk
    score: float
    rank: int


# ── RAG response ───────────────────────────────────────────────────

class RAGResponse(BaseModel):
    """Complete response from the RAG pipeline."""
    answer: str
