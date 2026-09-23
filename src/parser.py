from __future__ import annotations

import logging
import re

from src.models import DocumentAnalysis, ParsedSection, RawDocument

logger = logging.getLogger(__name__)

# Patterns for detecting headings in plain text
HEADING_PATTERNS = [
    # Markdown headings
    (re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE), "markdown"),
    # ALL CAPS lines (likely headings)
    (re.compile(r"^([A-Z][A-Z\s]{4,})$", re.MULTILINE), "caps"),
    # Numbered headings like "1. Introduction" or "1.1 Background"
    (re.compile(r"^(\d+\.[\d.]*)\s+(.+)$", re.MULTILINE), "numbered"),
]


class DocumentParser:
    """Parses raw documents into structured sections."""

    def parse(
        self, raw_doc: RawDocument, analysis: DocumentAnalysis | None = None
    ) -> list[ParsedSection]:
        """Parse a raw document into structured sections."""
        all_sections: list[ParsedSection] = []

        for page in raw_doc.pages:
            page_sections = self._parse_page_text(page.text, page.page_number)

            # Add tables as separate sections
            for table_rows in page.tables:
                table_text = self._format_table(table_rows)
                if table_text.strip():
                    page_sections.append(ParsedSection(
                        title="Table",
                        level=2,
                        content=table_text,
                        page_number=page.page_number,
                        is_table=True,
                    ))

            all_sections.extend(page_sections)

        # If analysis detected sections, try to align with parsed sections
        if analysis and analysis.sections:
            all_sections = self._enrich_with_analysis(all_sections, analysis)

        # Fallback: if no sections detected, wrap entire text as one section
        if not all_sections:
            full_text = "\n\n".join(p.text for p in raw_doc.pages if p.text)
            all_sections = [ParsedSection(
                title="Full Document",
                content=full_text,
                page_number=1,
            )]

        return all_sections

    def _parse_page_text(self, text: str, page_number: int) -> list[ParsedSection]:
        """Split page text into sections based on detected headings."""
        if not text.strip():
            return []

        # Find all headings and their positions
        headings: list[tuple[int, str, int]] = []  # (position, title, level)

        for pattern, pattern_type in HEADING_PATTERNS:
            for match in pattern.finditer(text):
                if pattern_type == "markdown":
                    level = len(match.group(1))
                    title = match.group(2).strip()
                elif pattern_type == "caps":
                    level = 1
                    title = match.group(1).strip().title()
                else:  # numbered
                    level = match.group(1).count(".") + 1
                    title = match.group(2).strip()

                # Skip very short "headings" that are likely not headings
                if len(title) < 3:
                    continue

                headings.append((match.start(), title, level))

        # Sort by position
        headings.sort(key=lambda h: h[0])

        # If no headings found, return entire page as one section
        if not headings:
            return [ParsedSection(
                title="Content",
                content=text.strip(),
                page_number=page_number,
            )]

        # Split text at heading boundaries
        sections: list[ParsedSection] = []

        # Content before the first heading
        first_pos = headings[0][0]
        if first_pos > 0:
            pre_content = text[:first_pos].strip()
            if pre_content:
                sections.append(ParsedSection(
                    title="Introduction",
                    content=pre_content,
                    page_number=page_number,
                ))

        # Content for each heading
        for i, (pos, title, level) in enumerate(headings):
            end_pos = headings[i + 1][0] if i + 1 < len(headings) else len(text)
            # Get content after the heading line
            heading_end = text.find("\n", pos)
            if heading_end == -1:
                heading_end = len(text)
            content = text[heading_end:end_pos].strip()

            sections.append(ParsedSection(
                title=title,
                level=level,
                content=content,
                page_number=page_number,
            ))

        return sections

    def _format_table(self, table_rows: list[dict]) -> str:
        """Convert table data to a pipe-delimited string that preserves relationships."""
        if not table_rows:
            return ""

        headers = list(table_rows[0].keys())
        lines = [" | ".join(str(h) for h in headers)]
        lines.append(" | ".join("---" for _ in headers))

        for row in table_rows:
            lines.append(" | ".join(str(row.get(h, "")) for h in headers))

        return "\n".join(lines)

    def _enrich_with_analysis(
        self, sections: list[ParsedSection], analysis: DocumentAnalysis
    ) -> list[ParsedSection]:
        """Use LLM analysis to improve section titles when detection was weak."""
        analysis_titles = {s.title.lower() for s in analysis.sections}

        for section in sections:
            # If a section is titled generically ("Content"), check if analysis
            # has a better title that matches the content
            if section.title in ("Content", "Introduction", "Untitled"):
                content_lower = section.content[:200].lower()
                for a_title in analysis_titles:
                    if a_title in content_lower:
                        section.title = a_title.title()
                        break

        return sections
