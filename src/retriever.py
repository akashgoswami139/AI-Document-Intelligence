from __future__ import annotations

import logging
import os
from typing import Any

from src.models import (
    DocumentChunk,
    DocumentMetadata,
    QueryAnalysis,
    RetrievalResult,
)
from src.query_analyzer import QueryAnalyzer
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Orchestrates query understanding and vector retrieval."""

    def __init__(self, vector_store: VectorStore, query_analyzer: QueryAnalyzer):
        self.vector_store = vector_store
        self.query_analyzer = query_analyzer
        self.top_k = int(os.getenv("TOP_K", "5"))
        # Over-fetch ratio: retrieve more than top_k, then trim
        self.fetch_k = self.top_k + 3

    def retrieve(
        self,
        query: str,
        documents: list[DocumentMetadata],
        use_query_analysis: bool = True,
    ) -> tuple[list[RetrievalResult], QueryAnalysis | None]:
        """
        Full retrieval pipeline: analyze → filter → search → rank.
        Returns (results, query_analysis).
        """
        query_analysis = None
        metadata_filter = None

        # Step 1: Query analysis for smart filtering
        if use_query_analysis and documents:
            query_analysis = self.query_analyzer.analyze(query, documents)
            metadata_filter = self._build_filter(query_analysis, documents)
            logger.info(
                "Query analysis — intent: %s, targets: %s",
                query_analysis.intent,
                query_analysis.target_document_types,
            )

        # Step 2: Filtered similarity search
        raw_results = self.vector_store.similarity_search(
            query=query,
            k=self.fetch_k,
            filter_dict=metadata_filter,
        )

        # Step 3: Fallback if filtered search returned nothing
        if not raw_results and metadata_filter:
            logger.info("Filtered search empty, retrying without filter")
            raw_results = self.vector_store.similarity_search(
                query=query,
                k=self.fetch_k,
            )

        # Step 4: Convert to RetrievalResult and rank
        results = []
        for rank, (doc, score) in enumerate(raw_results[: self.top_k]):
            meta = doc.metadata
            chunk = DocumentChunk(
                chunk_id=meta.get("chunk_id", ""),
                document_id=meta.get("document_id", ""),
                file_name=meta.get("file_name", ""),
                document_type=meta.get("document_type", "Unknown"),
                text=doc.page_content,
                page_number=meta.get("page_number"),
                section=meta.get("section", "Unknown"),
            )
            results.append(RetrievalResult(
                chunk=chunk,
                score=float(score),
                rank=rank + 1,
            ))

        logger.info("Retrieved %d chunks for query: %s", len(results), query[:50])
        return results, query_analysis

    def _build_filter(
        self,
        analysis: QueryAnalysis,
        documents: list[DocumentMetadata],
    ) -> dict[str, Any] | None:
        """
        Build ChromaDB metadata filter from query analysis.
        Only filter on document_type if the query clearly targets specific types.
        """
        if not analysis.target_document_types:
            return None

        # Map target types to actual document types in the store
        available_types = {d.document_type.lower() for d in documents}
        matched_types = [
            t for t in analysis.target_document_types
            if t.lower() in available_types
        ]

        if not matched_types:
            return None

        if len(matched_types) == 1:
            return {"document_type": matched_types[0]}

        # ChromaDB $in filter for multiple types
        return {"document_type": {"$in": matched_types}}
