from __future__ import annotations

import json
import logging
from src.models import (
    ContentComposition,
    DetectedEntity,
    DetectedSection,
    DocumentAnalysis,
    RawDocument,
)
from src.prompts import DOCUMENT_ANALYSIS_PROMPT, SUMMARY_PROMPT
from src.llm import get_gemini_model

logger = logging.getLogger(__name__)

# Max characters sent to LLM for analysis (controls cost/latency)
MAX_ANALYSIS_CHARS = 8000


class DocumentAnalyzer:
    """Analyzes documents using an LLM to infer type, structure, and entities."""

    def __init__(self):
        self.llm = get_gemini_model()

    def analyze(self, raw_doc: RawDocument) -> DocumentAnalysis:
        """Run LLM analysis on a raw document and return structured results."""
        full_text = self._get_document_text(raw_doc)

        if not full_text.strip():
            logger.warning("Empty document: %s", raw_doc.file_name)
            return DocumentAnalysis(
                document_type="Empty Document",
                summary="This document appears to be empty.",
                confidence=0.0,
            )

        # Truncate for analysis to control token usage
        analysis_text = full_text[:MAX_ANALYSIS_CHARS]

        try:
            prompt = DOCUMENT_ANALYSIS_PROMPT.format(document_text=analysis_text)
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            analysis_data = json.loads(content)
            return self._parse_analysis(analysis_data)

        except (json.JSONDecodeError, Exception) as e:
            logger.error("Analysis failed for %s: %s", raw_doc.file_name, e)
            return self._fallback_analysis(raw_doc)

    def generate_summary(self, raw_doc: RawDocument) -> str:
        """Generate a concise summary of the document."""
        text = self._get_document_text(raw_doc)[:MAX_ANALYSIS_CHARS]

        if not text.strip():
            return "This document appears to be empty."

        try:
            prompt = SUMMARY_PROMPT.format(document_text=text)
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error("Summary generation failed: %s", e)
            return "Summary could not be generated."

    def _get_document_text(self, raw_doc: RawDocument) -> str:
        """Concatenate all pages into a single string for analysis."""
        parts = []
        for page in raw_doc.pages:
            if page.text:
                parts.append(f"[Page {page.page_number}]\n{page.text}")
        return "\n\n".join(parts)

    def _parse_analysis(self, data: dict) -> DocumentAnalysis:
        """Parse LLM JSON output into a DocumentAnalysis model."""
        sections = [
            DetectedSection(
                title=s.get("title", "Unknown"),
                level=s.get("level", 1),
                page_number=s.get("page_number"),
            )
            for s in data.get("sections", [])
        ]

        entities = [
            DetectedEntity(
                name=e.get("name", "Unknown"),
                entity_type=e.get("entity_type", "unknown"),
            )
            for e in data.get("entities", [])
        ]

        composition_raw = data.get("content_composition", "text").lower()
        try:
            composition = ContentComposition(composition_raw)
        except ValueError:
            composition = ContentComposition.TEXT

        return DocumentAnalysis(
            document_type=data.get("document_type", "Unknown"),
            main_topic=data.get("main_topic", "Unknown"),
            purpose=data.get("purpose", "Unknown"),
            language=data.get("language", "English"),
            content_composition=composition,
            sections=sections,
            entities=entities,
            summary=data.get("summary", ""),
            key_dates=data.get("key_dates", []),
            organizations=data.get("organizations", []),
            people=data.get("people", []),
            confidence=float(data.get("confidence", 0.5)),
        )

    def _fallback_analysis(self, raw_doc: RawDocument) -> DocumentAnalysis:
        """Minimal analysis when LLM call fails."""
        return DocumentAnalysis(
            document_type=f"{raw_doc.file_type.upper()} Document",
            main_topic="Unknown",
            purpose="Unknown",
            summary=f"A {raw_doc.file_type.upper()} document with {raw_doc.total_pages} page(s).",
            confidence=0.1,
        )
