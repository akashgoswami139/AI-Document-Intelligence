from __future__ import annotations

import json
import logging
from src.models import DocumentMetadata, QueryAnalysis
from src.prompts import QUERY_ANALYSIS_PROMPT
from src.llm import get_gemini_model

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyzes user queries to drive smarter retrieval."""

    def __init__(self):
        self.llm = get_gemini_model()

    def analyze(
        self,
        query: str,
        documents: list[DocumentMetadata],
    ) -> QueryAnalysis:
        """Analyze a query against available documents."""
        doc_list = self._format_document_list(documents)

        try:
            prompt = QUERY_ANALYSIS_PROMPT.format(
                document_list=doc_list,
                query=query,
            )
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Strip markdown code fences
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            data = json.loads(content)
            # Changelog: normalize malformed LLM lists and fall back to general query analysis.
            if not isinstance(data, dict):
                raise ValueError("Query analysis response must be a JSON object")

            def string_list(value: object) -> list[str]:
                return [str(item) for item in value if item is not None] if isinstance(value, list) else []

            return QueryAnalysis(
                original_query=query,
                intent=data.get("intent", "general"),
                target_document_types=string_list(data.get("target_document_types", [])),
                target_sections=string_list(data.get("target_sections", [])),
                is_multi_document=bool(data.get("is_multi_document", False)),
                is_comparison=bool(data.get("is_comparison", False)),
                keywords=string_list(data.get("keywords", [])),
            )

        except Exception as e:
            logger.error("Query analysis failed: %s", e)
            return QueryAnalysis(
                original_query=query,
                intent="general",
                keywords=query.split()[:5],
            )

    def _format_document_list(self, documents: list[DocumentMetadata]) -> str:
        """Format document metadata for the prompt."""
        if not documents:
            return "No documents uploaded."

        lines = []
        for doc in documents:
            doc_type = doc.document_type
            lines.append(
                f"- {doc.file_name} (Type: {doc_type})"
            )
        return "\n".join(lines)
