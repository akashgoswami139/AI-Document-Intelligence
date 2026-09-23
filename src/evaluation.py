from __future__ import annotations

import json
import logging
from src.models import (
    DocumentMetadata,
    EvaluationMetrics,
    EvaluationResult,
    SourceCitation,
)
from src.prompts import (
    ANSWER_COMPLETENESS_PROMPT,
    ANSWER_FAITHFULNESS_PROMPT,
    GENERATE_TEST_QUESTIONS_PROMPT,
    RETRIEVAL_RELEVANCE_PROMPT,
)
from src.rag_chain import RAGChain
from src.llm import get_gemini_model

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates RAG pipeline quality using LLM-as-judge."""

    def __init__(self, rag_chain: RAGChain):
        self.rag_chain = rag_chain
        self.judge_llm = get_gemini_model()

    def generate_test_questions(
        self,
        documents: list[DocumentMetadata],
        raw_texts: dict[str, str],
        num_per_doc: int = 3,
    ) -> list[dict[str, str]]:
        """Generate test questions from document content."""
        all_questions = []

        for doc in documents:
            text = raw_texts.get(doc.document_id, "")[:4000]
            if not text:
                continue

            try:
                prompt = GENERATE_TEST_QUESTIONS_PROMPT.format(
                    num_questions=num_per_doc,
                    file_name=doc.file_name,
                    document_type=doc.document_type,
                    content=text,
                )
                response = self.judge_llm.invoke(prompt)
                content = response.content.strip()

                if content.startswith("```"):
                    content = content.split("\n", 1)[-1]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                data = json.loads(content)
                for q in data.get("questions", []):
                    q["source_document"] = doc.file_name
                    all_questions.append(q)

            except Exception as e:
                logger.error("Failed to generate questions for %s: %s", doc.file_name, e)

        return all_questions

    def evaluate(
        self,
        questions: list[dict[str, str]],
        documents: list[DocumentMetadata],
    ) -> list[EvaluationResult]:
        """Run full evaluation on a set of test questions."""
        results = []

        for q_data in questions:
            question = q_data["question"]
            expected = q_data.get("expected_answer", "")

            try:
                # Run the RAG pipeline
                rag_response = self.rag_chain.query(question, documents)

                # Format retrieved context for evaluation
                context = "\n\n".join(
                    r.chunk.text for r in rag_response.retrieved_chunks
                )

                # Evaluate each dimension
                metrics = EvaluationMetrics()

                # 1. Retrieval relevance
                metrics.retrieval_relevance = self._judge_metric(
                    RETRIEVAL_RELEVANCE_PROMPT, question=question, context=context
                )

                # 2. Answer faithfulness
                metrics.answer_faithfulness = self._judge_metric(
                    ANSWER_FAITHFULNESS_PROMPT,
                    question=question,
                    context=context,
                    answer=rag_response.answer,
                )

                # 3. Answer completeness
                metrics.answer_completeness = self._judge_metric(
                    ANSWER_COMPLETENESS_PROMPT,
                    question=question,
                    context=context,
                    answer=rag_response.answer,
                )

                # 4. Context relevance (same as retrieval for now)
                metrics.context_relevance = metrics.retrieval_relevance

                # 5. Citation correctness (programmatic)
                metrics.citation_correctness = self._check_citations(
                    rag_response.sources,
                    rag_response.retrieved_chunks,
                )

                results.append(EvaluationResult(
                    question=question,
                    expected_answer=expected,
                    generated_answer=rag_response.answer,
                    retrieved_context=context[:1000],
                    sources=rag_response.sources,
                    metrics=metrics,
                ))

            except Exception as e:
                logger.error("Evaluation failed for '%s': %s", question[:50], e)
                results.append(EvaluationResult(
                    question=question,
                    expected_answer=expected,
                    generated_answer=f"Error: {e}",
                    metrics=EvaluationMetrics(),
                    explanation=f"Evaluation failed: {e}",
                ))

        return results

    def _judge_metric(self, prompt_template: str, **kwargs) -> float:
        """Ask the LLM judge to score a metric."""
        try:
            prompt = prompt_template.format(**kwargs)
            response = self.judge_llm.invoke(prompt)
            content = response.content.strip()

            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            data = json.loads(content)
            return float(data.get("score", 0.0))

        except Exception as e:
            logger.error("Judge metric failed: %s", e)
            return 0.0

    def _check_citations(
        self,
        sources: list[SourceCitation],
        retrieved_chunks: list,
    ) -> float:
        """Programmatic check: do cited sources exist in retrieved chunks?"""
        if not sources:
            return 0.0

        retrieved_files = {r.chunk.file_name for r in retrieved_chunks}
        matches = sum(1 for s in sources if s.file_name in retrieved_files)
        return matches / len(sources) if sources else 0.0

    @staticmethod
    def compute_summary(results: list[EvaluationResult]) -> dict[str, float]:
        """Compute average metrics across all evaluation results."""
        if not results:
            return {}

        n = len(results)
        return {
            "avg_retrieval_relevance": sum(r.metrics.retrieval_relevance for r in results) / n,
            "avg_context_relevance": sum(r.metrics.context_relevance for r in results) / n,
            "avg_answer_faithfulness": sum(r.metrics.answer_faithfulness for r in results) / n,
            "avg_answer_completeness": sum(r.metrics.answer_completeness for r in results) / n,
            "avg_citation_correctness": sum(r.metrics.citation_correctness for r in results) / n,
            "total_questions": n,
        }
