
DOCUMENT_ANALYSIS_PROMPT = """You are an expert document analyst. Analyze the following document text and provide a structured analysis.

You must determine:
1. Document Type (e.g., Invoice, Resume, Research Paper, Report, Contract, Policy, Notes, Letter, Manual, Presentation, Spreadsheet Data, etc.)
2. Main Topic/Subject
3. Approximate Purpose of the document
4. Language
5. Content Composition: "text", "tables", or "mixed"
6. Sections/Headings detected in the document
7. Important Entities (people, organizations, dates, amounts, identifiers)
8. A brief summary (2-3 sentences)
9. Your confidence level (0.0 to 1.0)

Do NOT hardcode or assume the document type. Infer it from the content.

Respond ONLY with valid JSON in this exact format:
{{
    "document_type": "string",
    "main_topic": "string",
    "purpose": "string",
    "language": "string",
    "content_composition": "text|tables|mixed",
    "sections": [
        {{"title": "string", "level": 1, "page_number": null}}
    ],
    "entities": [
        {{"name": "string", "entity_type": "string"}}
    ],
    "summary": "string",
    "key_dates": ["string"],
    "organizations": ["string"],
    "people": ["string"],
    "confidence": 0.0
}}

DOCUMENT TEXT:
{document_text}
"""

# ── Document Summary Prompt ────────────────────────────────────────

SUMMARY_PROMPT = """Summarize the following document in 3-5 sentences. Focus on the key information, purpose, and main findings or content.

DOCUMENT TEXT:
{document_text}

SUMMARY:"""

# ── Query Analysis Prompt ──────────────────────────────────────────

QUERY_ANALYSIS_PROMPT = """You are a query analysis engine for a document Q&A system.

The user has uploaded the following documents:
{document_list}

The user's question is:
"{query}"

Analyze the question and determine:
1. The intent of the question
2. Which document types are most relevant
3. Which sections might contain the answer
4. Whether the question requires information from multiple documents
5. Whether this is a comparison question
6. Key search terms/keywords

Respond ONLY with valid JSON:
{{
    "intent": "string",
    "target_document_types": ["string"],
    "target_sections": ["string"],
    "is_multi_document": false,
    "is_comparison": false,
    "keywords": ["string"]
}}

Analyze the question now."""

# ── RAG System Prompt ──────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are an AI Document Intelligence Assistant. Your ONLY job is to answer questions using the provided document context.

STRICT RULES:
1. Answer ONLY using the retrieved document context provided below.
2. If the answer cannot be found in the provided documents, say exactly: "I couldn't find this information in the uploaded documents."
3. Do NOT invent facts.
4. Do NOT use outside knowledge to fill in missing information.
5. Always cite your sources — mention the file name, page number, and section when available.
6. If the question asks about something not present in any document, say so clearly.
7. Be precise and specific. Quote relevant text when helpful.
8. For numerical data, provide exact figures from the documents.

FORMAT YOUR ANSWER:
- Start with a clear, direct answer
- Support with evidence from the documents
- End with source references in this format:
  📄 Sources:
  - [filename] — Page X, Section: Y

RETRIEVED DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

USER QUESTION: {question}

YOUR ANSWER:"""

# ── Evaluation Prompts ─────────────────────────────────────────────

RETRIEVAL_RELEVANCE_PROMPT = """You are evaluating retrieval quality for a RAG system.

Question: {question}
Retrieved Context: {context}

Rate how relevant the retrieved context is to answering the question.
Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).

Respond with JSON only:
{{"score": 0.0, "explanation": "string"}}"""

ANSWER_FAITHFULNESS_PROMPT = """You are evaluating answer faithfulness for a RAG system.

Question: {question}
Retrieved Context: {context}
Generated Answer: {answer}

Rate whether the answer is faithful to the context — i.e., every claim in the answer is supported by the context.
Score from 0.0 (completely hallucinated) to 1.0 (perfectly grounded).

Respond with JSON only:
{{"score": 0.0, "explanation": "string"}}"""

ANSWER_COMPLETENESS_PROMPT = """You are evaluating answer completeness for a RAG system.

Question: {question}
Retrieved Context: {context}
Generated Answer: {answer}

Rate how completely the answer addresses the question given the available context.
Score from 0.0 (doesn't address the question at all) to 1.0 (fully addresses all aspects).

Respond with JSON only:
{{"score": 0.0, "explanation": "string"}}"""

GENERATE_TEST_QUESTIONS_PROMPT = """You are creating evaluation test questions for a RAG system.

Based on the following document content, generate {num_questions} diverse questions that can be answered from this content.

For each question, provide the expected answer based on the document.

DOCUMENT: {file_name} (Type: {document_type})
CONTENT:
{content}

Respond with JSON only:
{{
    "questions": [
        {{"question": "string", "expected_answer": "string"}}
    ]
}}"""
