from __future__ import annotations

import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.models import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages ChromaDB operations: indexing, querying, deletion."""

    def __init__(self, embeddings: Embeddings, collection_name: str = "documents"):
        self.embeddings = embeddings
        self.collection_name = collection_name
        self._store: Chroma | None = None

    @property
    def store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        return self._store

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Add document chunks to the vector store. Returns count added."""
        if not chunks:
            return 0

        documents = [
            Document(
                page_content=chunk.text,
                metadata=chunk.to_vectorstore_metadata(),
            )
            for chunk in chunks
        ]

        self.store.add_documents(documents)
        logger.info("Added %d chunks to vector store", len(documents))
        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Search for similar documents with optional metadata filtering.
        Returns list of (Document, similarity_score) tuples.
        """
        try:
            results = self.store.similarity_search_with_relevance_scores(
                query,
                k=k,
                filter=filter_dict,
            )
            return results
        except Exception as e:
            logger.error("Similarity search failed: %s", e)
            # Retry without filter if filtering caused the error
            if filter_dict:
                logger.info("Retrying without metadata filter")
                return self.store.similarity_search_with_relevance_scores(query, k=k)
            return []

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to a specific document."""
        try:
            collection = self.store._collection
            collection.delete(where={"document_id": document_id})
            logger.info("Deleted chunks for document_id: %s", document_id)
        except Exception as e:
            logger.error("Failed to delete document %s: %s", document_id, e)

    def get_all_metadata(self) -> list[dict[str, Any]]:
        """Retrieve metadata for all stored chunks (for UI display)."""
        try:
            collection = self.store._collection
            results = collection.get(include=["metadatas"])
            return results.get("metadatas", [])
        except Exception:
            return []

    def get_document_ids(self) -> set[str]:
        """Get unique document IDs in the store."""
        metadata_list = self.get_all_metadata()
        return {m.get("document_id", "") for m in metadata_list if m}

    def get_chunk_count(self) -> int:
        """Total number of chunks in the store."""
        try:
            return self.store._collection.count()
        except Exception:
            return 0

    def reset(self) -> None:
        """Clear all data from the vector store."""
        try:
            if self._store is not None:
                self._store.delete_collection()
                self._store = None
            logger.info("Vector store reset")
        except Exception as e:
            logger.error("Failed to reset vector store: %s", e)
