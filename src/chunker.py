from __future__ import annotations

import logging
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import (
    ChunkingStrategy,
    DocumentChunk,
    DocumentMetadata,
    ParsedSection,
)

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Creates chunks from parsed sections using configurable strategies."""

    def __init__(self):
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_naive(
        self,
        sections: list[ParsedSection],
        doc_meta: DocumentMetadata,
    ) -> list[DocumentChunk]:
        """
        Naive strategy: concatenate all text, then split with
        RecursiveCharacterTextSplitter. Structure is ignored.
        """
        full_text = "\n\n".join(s.content for s in sections if s.content)

        if not full_text.strip():
            return []

        splits = self._text_splitter.split_text(full_text)

        chunks = []
        for i, text in enumerate(splits):
            # Approximate page number from character position
            page_num = self._estimate_page(text, sections)

            chunks.append(DocumentChunk(
                document_id=doc_meta.document_id,
                file_name=doc_meta.file_name,
                document_type=doc_meta.document_type,
                text=text,
                page_number=page_num,
                section="Unknown",
                chunking_strategy=ChunkingStrategy.NAIVE,
            ))

        logger.info(
            "Naive chunking: %d chunks from %s", len(chunks), doc_meta.file_name
        )
        return chunks

    def chunk_structure_aware(
        self,
        sections: list[ParsedSection],
        doc_meta: DocumentMetadata,
    ) -> list[DocumentChunk]:
        """
        Structure-aware strategy: respect section boundaries, keep tables
        atomic, and split long sections with overlap.
        """
        chunks: list[DocumentChunk] = []

        for section in sections:
            if not section.content.strip():
                continue

            section_chunks = self._chunk_section(section, doc_meta)
            chunks.extend(section_chunks)

        logger.info(
            "Structure-aware chunking: %d chunks from %s",
            len(chunks),
            doc_meta.file_name,
        )
        return chunks

    def _chunk_section(
        self, section: ParsedSection, doc_meta: DocumentMetadata
    ) -> list[DocumentChunk]:
        """Chunk a single section, keeping tables atomic."""
        content = section.content.strip()

        # Tables should never be split — they are atomic chunks
        if section.is_table:
            return [DocumentChunk(
                document_id=doc_meta.document_id,
                file_name=doc_meta.file_name,
                document_type=doc_meta.document_type,
                text=f"[Table: {section.title}]\n{content}",
                page_number=section.page_number,
                section=section.title,
                chunking_strategy=ChunkingStrategy.STRUCTURE_AWARE,
            )]

        # Short sections stay as single chunks
        if len(content) <= self.chunk_size:
            return [DocumentChunk(
                document_id=doc_meta.document_id,
                file_name=doc_meta.file_name,
                document_type=doc_meta.document_type,
                text=content,
                page_number=section.page_number,
                section=section.title,
                chunking_strategy=ChunkingStrategy.STRUCTURE_AWARE,
            )]

        # Long sections: split with RecursiveCharacterTextSplitter
        splits = self._text_splitter.split_text(content)
        return [
            DocumentChunk(
                document_id=doc_meta.document_id,
                file_name=doc_meta.file_name,
                document_type=doc_meta.document_type,
                text=text,
                page_number=section.page_number,
                section=section.title,
                chunking_strategy=ChunkingStrategy.STRUCTURE_AWARE,
            )
            for text in splits
        ]

    def _estimate_page(self, chunk_text: str, sections: list[ParsedSection]) -> int | None:
        """Best-effort page estimation for naive chunks."""
        for section in sections:
            if chunk_text[:100] in section.content:
                return section.page_number
        return sections[0].page_number if sections else None
