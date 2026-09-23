"""Generate the complete project guide as a PDF."""

from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "docs" / "PROJECT_GUIDE.md"
OUTPUT = ROOT / "docs" / "PROJECT_GUIDE.pdf"


def build_pdf() -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="GuideTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        leading=28,
        spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        name="GuideHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="GuideBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="GuideBullet",
        parent=styles["BodyText"],
        leftIndent=14,
        firstLineIndent=-8,
        fontSize=9.5,
        leading=13,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="GuideCode",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        leftIndent=10,
        rightIndent=10,
        spaceBefore=4,
        spaceAfter=8,
    ))

    story = []
    in_code = False
    code_lines = []

    for raw_line in SOURCE.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["GuideCode"]))
                code_lines = []
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line:
            story.append(Spacer(1, 3))
        elif line.startswith("# "):
            story.append(Paragraph(line[2:], styles["GuideTitle"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["GuideHeading"]))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:], styles["Heading3"]))
        elif line.startswith("- "):
            story.append(Paragraph("&#8226; " + line[2:], styles["GuideBullet"]))
        else:
            escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            escaped = escaped.replace("`", "")
            story.append(Paragraph(escaped, styles["GuideBody"]))

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=LETTER,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Document Intelligence RAG Complete Project Guide",
        author="Document Intelligence RAG",
    )
    document.build(story)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
