from __future__ import annotations

import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.models import (
    DocumentMetadata,
    RAGResponse,
    RetrievalResult,
)
from src.prompts import RAG_SYSTEM_PROMPT
from src.retriever import Retriever
from src.llm import get_gemini_model

logger = logging.getLogger(__name__)


class RAGChain:
    """Orchestrates retrieval → context → LLM → cited answer."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.llm = get_gemini_model()
        self.parser = StrOutputParser()
        self.chat_history: list[dict[str, str]] = []
        self.max_history = 10  # Keep last N exchanges

    def query(
        self,
        question: str,
        documents: list[DocumentMetadata],
    ) -> RAGResponse:
        """Full RAG pipeline: retrieve → format → generate → cite."""

        # Step 1: Retrieve relevant chunks
        results, query_analysis = self.retriever.retrieve(question, documents)

        # Step 2: Format context from retrieved chunks
        context = self._format_context(results)

        # Step 3: Format conversation history
        history_str = self._format_history()

        # Step 4: Build prompt and invoke LLM
        prompt = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)
        chain = prompt | self.llm | self.parser

        answer = chain.invoke({
            "context": context,
            "chat_history": history_str,
            "question": question,
        })

        # Step 5: Update conversation memory
        self._update_history(question, answer)

        return RAGResponse(answer=answer)

    def _format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        if not results:
            return "No relevant documents were found."

        parts = []
        for r in results:
            chunk = r.chunk
            header = (
                f"[Source: {chunk.file_name} | "
                f"Page: {chunk.page_number or 'N/A'} | "
                f"Section: {chunk.section} | "
                f"Type: {chunk.document_type}]"
            )
            parts.append(f"{header}\n{chunk.text}")

        return "\n\n---\n\n".join(parts)

    def _format_history(self) -> str:
        """Format recent conversation history for context."""
        if not self.chat_history:
            return "No previous conversation."

        lines = []
        for entry in self.chat_history[-self.max_history:]:
            lines.append(f"User: {entry['question']}")
            lines.append(f"Assistant: {entry['answer'][:200]}...")
        return "\n".join(lines)

    def _update_history(self, question: str, answer: str) -> None:
        """Add the latest exchange to conversation memory."""
        self.chat_history.append({
            "question": question,
            "answer": answer,
        })
        # Trim to max history
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

    def clear_history(self) -> None:
        """Reset conversation memory."""
        self.chat_history.clear()
