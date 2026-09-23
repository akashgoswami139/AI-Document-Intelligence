import io
import logging
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

PROJECT_GUIDE_PDF = PROJECT_ROOT / "docs" / "PROJECT_GUIDE.pdf"

from src.chunker import DocumentChunker
from src.document_analyzer import DocumentAnalyzer
from src.document_loader import DocumentLoader
from src.embeddings import get_embedding_model
from src.evaluation import RAGEvaluator
from src.models import DocumentMetadata
from src.parser import DocumentParser
from src.query_analyzer import QueryAnalyzer
from src.rag_chain import RAGChain
from src.retriever import Retriever
from src.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page Config ────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
        background: #ffffff;
        color: #111827;
    }

    section[data-testid="stAppViewContainer"],
    .main,
    .block-container {
        background: #ffffff;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #ffffff;
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #111827;
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
    }

    .metric-card h3 {
        color: #4b5563;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0 0 8px 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-card .value {
        color: #111827;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }

    /* Source citation card */
    .source-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.9rem;
    }

    /* Analysis panel */
    .analysis-panel {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
    }

    .analysis-panel h2 {
        color: #111827;
        font-size: 1.3rem;
        margin-bottom: 16px;
    }

    .entity-tag {
        display: inline-block;
        background: #f3f4f6;
        border: 1px solid #d1d5db;
        border-radius: 20px;
        padding: 4px 12px;
        margin: 3px;
        font-size: 0.8rem;
        color: #374151;
    }

    /* Score bar */
    .score-bar {
        height: 6px;
        border-radius: 3px;
        background: #e5e7eb;
        margin-top: 4px;
        overflow: hidden;
    }

    .score-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }

    /* Chat enhancements */
    .stChatMessage {
        border-radius: 12px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }

    /* Document status pill */
    .status-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-ready {
        background: #ecfdf5;
        color: #166534;
        border: 1px solid #bbf7d0;
    }

    .status-processing {
        background: #fefce8;
        color: #854d0e;
        border: 1px solid #fde68a;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ───────────────────────────────────


def init_session_state():
    defaults = {
        "documents": {},           # doc_id -> DocumentMetadata
        "raw_documents": {},       # doc_id -> RawDocument
        "raw_texts": {},           # doc_id -> full text string
        "analyses": {},            # doc_id -> DocumentAnalysis
        "chat_history": [],        # list of {"role", "content", "sources"}
        "last_response": None,     # last RAGResponse
        "chunking_strategy": "Structure-Aware",
        "initialized": False,
        "processing": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ── Initialize Pipeline Components ─────────────────────────────────

@st.cache_resource
def get_pipeline():
    """Initialize all pipeline components (cached across reruns)."""
    embeddings = get_embedding_model()
    vector_store = VectorStore(embeddings)
    query_analyzer = QueryAnalyzer()
    retriever = Retriever(vector_store, query_analyzer)
    rag_chain = RAGChain(retriever)
    evaluator = RAGEvaluator(rag_chain)

    return {
        "loader": DocumentLoader(),
        "analyzer": DocumentAnalyzer(),
        "parser": DocumentParser(),
        "chunker": DocumentChunker(),
        "embeddings": embeddings,
        "vector_store": vector_store,
        "retriever": retriever,
        "rag_chain": rag_chain,
        "evaluator": evaluator,
    }


pipeline = get_pipeline()


# ── Document Processing Pipeline ───────────────────────────────────

def process_document(uploaded_file) -> str | None:
    """Run the full document processing pipeline on an uploaded file."""
    try:
        file_bytes = uploaded_file.read()
        file_obj = io.BytesIO(file_bytes)
        file_name = uploaded_file.name

        # Step 1: Load
        raw_doc = pipeline["loader"].load(file_name, file_obj)

        # Step 2: Analyze
        analysis = pipeline["analyzer"].analyze(raw_doc)
        summary = pipeline["analyzer"].generate_summary(raw_doc)
        analysis.summary = summary

        # Step 3: Parse
        sections = pipeline["parser"].parse(raw_doc, analysis)

        # Step 4: Create metadata
        doc_meta = DocumentMetadata(
            file_name=file_name,
            file_type=raw_doc.file_type,
            document_type=analysis.document_type,
            total_pages=raw_doc.total_pages,
            analysis=analysis,
        )

        # Step 5: Chunk
        strategy = st.session_state.get("chunking_strategy", "Structure-Aware")
        if strategy == "Naive":
            chunks = pipeline["chunker"].chunk_naive(sections, doc_meta)
        else:
            chunks = pipeline["chunker"].chunk_structure_aware(sections, doc_meta)

        # Step 6: Index in vector store
        pipeline["vector_store"].add_chunks(chunks)

        # Step 7: Store in session state
        doc_id = doc_meta.document_id
        st.session_state.documents[doc_id] = doc_meta
        st.session_state.raw_documents[doc_id] = raw_doc
        st.session_state.analyses[doc_id] = analysis

        full_text = "\n\n".join(p.text for p in raw_doc.pages if p.text)
        st.session_state.raw_texts[doc_id] = full_text

        return doc_id

    except Exception as e:
        logger.error("Failed to process %s: %s", uploaded_file.name, e)
        st.error(f"Error processing {uploaded_file.name}: {e}")
        return None


# ── Sidebar ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# AI Document Intelligence")
    st.markdown("*Upload any document. Let AI understand it.*")
    if PROJECT_GUIDE_PDF.exists():
        st.download_button(
            "Download Project Guide",
            data=PROJECT_GUIDE_PDF.read_bytes(),
            file_name=PROJECT_GUIDE_PDF.name,
            mime="application/pdf",
            use_container_width=True,
        )
    st.markdown("---")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt", "md", "csv", "xlsx"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    # Chunking strategy selector
    st.markdown("### Settings")
    st.session_state.chunking_strategy = st.radio(
        "Chunking Strategy",
        ["Structure-Aware", "Naive"],
        index=0,
        help="Structure-Aware preserves headings, tables, and sections. Naive is a simple baseline.",
    )

    # Process uploaded files
    if uploaded_files:
        already_processed = {
            d.file_name for d in st.session_state.documents.values()
        }
        new_files = [f for f in uploaded_files if f.name not in already_processed]

        if new_files:
            st.markdown("---")
            progress_bar = st.progress(0, text="Processing documents...")

            for i, file in enumerate(new_files):
                progress_bar.progress(
                    (i + 1) / len(new_files),
                    text=f"Processing {file.name}...",
                )
                process_document(file)

            progress_bar.progress(1.0, text="All documents processed!")

    # Document list
    if st.session_state.documents:
        st.markdown("---")
        st.markdown("### Uploaded Documents")

        for doc_meta in st.session_state.documents.values():
            doc_type = doc_meta.document_type

            st.markdown(
                f'<div style="padding:8px 12px; margin:4px 0; '
                f'background:rgba(99,102,241,0.1); border-radius:8px; '
                f'border:1px solid rgba(99,102,241,0.2);">'
                f'<strong style="color:#e0e0ff;">{doc_meta.file_name}</strong><br/>'
                f'<span style="color:#a5b4fc; font-size:0.8rem;">{doc_type} · {doc_meta.total_pages} page(s)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Reset button
    if st.session_state.documents:
        st.markdown("---")
        if st.button("Clear All Documents", type="secondary", use_container_width=True):
            pipeline["vector_store"].reset()
            pipeline["rag_chain"].clear_history()
            for key in ["documents", "raw_documents", "raw_texts", "analyses", "chat_history", "last_response"]:
                st.session_state[key] = {} if isinstance(st.session_state[key], dict) else []
            st.session_state.last_response = None
            st.rerun()


# ── Helper Functions ───────────────────────────────────────────────

def render_score_bar(score: float, color: str = "#6366f1") -> str:
    pct = max(0, min(100, score * 100))
    return (
        f'<div class="score-bar">'
        f'<div class="score-fill" style="width:{pct}%; background:{color};"></div>'
        f'</div>'
    )


# ── Main Area — Tabs ───────────────────────────────────────────────

tab_chat, tab_docs, tab_analysis, tab_sources, tab_eval = st.tabs([
    "Chat", "Documents", "Document Analysis", "Sources", "Evaluation"
])


# ── Tab 1: Chat ────────────────────────────────────────────────────

with tab_chat:
    st.markdown("## Ask Your Documents")

    if not st.session_state.documents:
        st.info("Upload documents using the sidebar to get started.")
    else:
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # Show inline sources for assistant messages
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("View Sources", expanded=False):
                        for src in msg["sources"]:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<strong>{src.file_name}</strong> — '
                                f'Page {src.page_number or "N/A"}, '
                                f'Section: {src.section}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Show user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt,
            })
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating answer..."):
                    doc_list = list(st.session_state.documents.values())
                    response = pipeline["rag_chain"].query(prompt, doc_list)
                    st.session_state.last_response = response

                st.markdown(response.answer)

                # Show sources
                if response.sources:
                    with st.expander("View Sources", expanded=True):
                        for src in response.sources:
                            score_color = "#22c55e" if src.relevance_score > 0.7 else "#eab308" if src.relevance_score > 0.4 else "#ef4444"
                            st.markdown(
                                f'<div class="source-card">'
                                f'<strong>{src.file_name}</strong> — '
                                f'Page {src.page_number or "N/A"}, '
                                f'Section: {src.section} '
                                f'<span style="color:{score_color}; font-weight:600;">'
                                f'({src.relevance_score:.2f})</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

            # Save to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
            })


# ── Tab 2: Documents ───────────────────────────────────────────────

with tab_docs:
    st.markdown("## Document Management")

    if not st.session_state.documents:
        st.info("No documents uploaded yet. Use the sidebar to upload files.")
    else:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        total_docs = len(st.session_state.documents)
        total_chunks = pipeline["vector_store"].get_chunk_count()
        doc_types = set(d.document_type for d in st.session_state.documents.values())

        with col1:
            st.markdown(
                f'<div class="metric-card"><h3>Total Documents</h3>'
                f'<p class="value">{total_docs}</p></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card"><h3>Total Chunks</h3>'
                f'<p class="value">{total_chunks}</p></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="metric-card"><h3>Document Types</h3>'
                f'<p class="value">{len(doc_types)}</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Document table
        for doc_id, doc_meta in st.session_state.documents.items():
            analysis = st.session_state.analyses.get(doc_id)
            col_a, col_b, col_c, col_d = st.columns([3, 2, 2, 1])
            with col_a:
                st.markdown(f"**{doc_meta.file_name}**")
            with col_b:
                st.markdown(f"`{doc_meta.document_type}`")
            with col_c:
                st.markdown(f"{doc_meta.total_pages} page(s)")
            with col_d:
                st.markdown(
                    '<span class="status-pill status-ready">Ready</span>',
                    unsafe_allow_html=True,
                )


# ── Tab 3: Document Analysis ──────────────────────────────────────

with tab_analysis:
    st.markdown("## AI Document Analysis")

    if not st.session_state.analyses:
        st.info("Upload documents to see AI analysis results.")
    else:
        for doc_id, analysis in st.session_state.analyses.items():
            doc_meta = st.session_state.documents.get(doc_id)
            if not doc_meta:
                continue

            st.markdown(
                f'<div class="analysis-panel">'
                f'<h2>{doc_meta.file_name}</h2>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Key info in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Document Type:** `{analysis.document_type}`")
                st.markdown(f"**Main Topic:** {analysis.main_topic}")
            with col2:
                st.markdown(f"**Purpose:** {analysis.purpose}")
                st.markdown(f"**Language:** {analysis.language}")
            with col3:
                st.markdown(f"**Content:** `{analysis.content_composition.value}`")
                confidence_pct = f"{analysis.confidence * 100:.0f}%"
                st.markdown(f"**Confidence:** {confidence_pct}")

            # Summary
            if analysis.summary:
                st.markdown("#### AI Summary")
                st.markdown(f"> {analysis.summary}")

            # Sections
            if analysis.sections:
                st.markdown("#### Detected Sections")
                for section in analysis.sections:
                    indent = "  " * (section.level - 1)
                    page_info = f" (Page {section.page_number})" if section.page_number else ""
                    st.markdown(f"{indent}• **{section.title}**{page_info}")

            # Entities
            if analysis.entities:
                st.markdown("#### Entities")
                entity_html = ""
                for entity in analysis.entities:
                    entity_html += (
                        f'<span class="entity-tag">'
                        f'{entity.name} <small>({entity.entity_type})</small>'
                        f'</span>'
                    )
                st.markdown(entity_html, unsafe_allow_html=True)

            # Key dates, orgs, people
            detail_cols = st.columns(3)
            with detail_cols[0]:
                if analysis.key_dates:
                    st.markdown("**Key Dates**")
                    for d in analysis.key_dates:
                        st.markdown(f"- {d}")
            with detail_cols[1]:
                if analysis.organizations:
                    st.markdown("**Organizations**")
                    for o in analysis.organizations:
                        st.markdown(f"- {o}")
            with detail_cols[2]:
                if analysis.people:
                    st.markdown("**People**")
                    for p in analysis.people:
                        st.markdown(f"- {p}")

            st.markdown("---")



        # Query analysis info
        if response.query_analysis:
            qa = response.query_analysis
            st.markdown("### Query Understanding")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Intent:** `{qa.intent}`")
                st.markdown(f"**Multi-document:** {'Yes' if qa.is_multi_document else 'No'}")
            with col2:
                if qa.target_document_types:
                    st.markdown(f"**Target Types:** {', '.join(qa.target_document_types)}")
                if qa.keywords:
                    st.markdown(f"**Keywords:** {', '.join(qa.keywords)}")

        st.markdown("---")

        # Retrieved chunks
        if response.retrieved_chunks:
            st.markdown(f"### Retrieved Chunks ({len(response.retrieved_chunks)})")

            for i, result in enumerate(response.retrieved_chunks):
                chunk = result.chunk
                score = result.score
                score_color = "#22c55e" if score > 0.7 else "#eab308" if score > 0.4 else "#ef4444"

                with st.expander(
                    f"#{result.rank} — {chunk.file_name} | "
                    f"Page {chunk.page_number or 'N/A'} | "
                    f"Section: {chunk.section} | "
                    f"Score: {score:.3f}",
                    expanded=(i == 0),
                ):
                    st.markdown(
                        f"**Document Type:** `{chunk.document_type}` | "
                        f"**Chunk ID:** `{chunk.chunk_id}`"
                    )
                    st.markdown(
                        f'<div style="margin-top:4px;">Relevance: '
                        f'<strong style="color:{score_color}">{score:.3f}</strong>'
                        f'{render_score_bar(score, score_color)}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")
                    st.text(chunk.text)
        else:
            st.warning("No chunks were retrieved for the last query.")


# ── Tab 5: Evaluation ─────────────────────────────────────────────

with tab_eval:
    st.markdown("## RAG Evaluation")

    if not st.session_state.documents:
        st.info("Upload documents first, then run evaluation.")
    else:
        st.markdown(
            "Generate test questions from your documents and evaluate "
            "retrieval quality, answer faithfulness, and citation accuracy."
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            num_q = st.number_input("Questions per document", min_value=1, max_value=5, value=2)
        with col2:
            run_eval = st.button("Run Evaluation", type="primary", use_container_width=True)

        if run_eval:
            doc_list = list(st.session_state.documents.values())
            evaluator = pipeline["evaluator"]

            with st.spinner("Generating test questions..."):
                questions = evaluator.generate_test_questions(
                    doc_list, st.session_state.raw_texts, num_per_doc=num_q
                )

            if not questions:
                st.error("Could not generate test questions. Check your API key and documents.")
            else:
                st.success(f"Generated {len(questions)} test questions.")

                with st.spinner("Running evaluation (this may take a minute)..."):
                    results = evaluator.evaluate(questions, doc_list)

                # Summary metrics
                summary = RAGEvaluator.compute_summary(results)
                st.markdown("### Evaluation Summary")

                metric_cols = st.columns(5)
                metric_names = [
                    ("Retrieval\nRelevance", "avg_retrieval_relevance", "#6366f1"),
                    ("Context\nRelevance", "avg_context_relevance", "#8b5cf6"),
                    ("Answer\nFaithfulness", "avg_answer_faithfulness", "#22c55e"),
                    ("Answer\nCompleteness", "avg_answer_completeness", "#0ea5e9"),
                    ("Citation\nCorrectness", "avg_citation_correctness", "#f59e0b"),
                ]

                for col, (label, key, color) in zip(metric_cols, metric_names):
                    val = summary.get(key, 0)
                    with col:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<h3>{label}</h3>'
                            f'<p class="value" style="color:{color}">{val:.2f}</p>'
                            f'{render_score_bar(val, color)}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Detailed results
                st.markdown("### Detailed Results")
                for i, result in enumerate(results):
                    with st.expander(
                        f"Q{i+1}: {result.question[:80]}{'...' if len(result.question) > 80 else ''}",
                        expanded=False,
                    ):
                        st.markdown(f"**Expected Answer:** {result.expected_answer}")
                        st.markdown(f"**Generated Answer:** {result.generated_answer}")

                        if result.sources:
                            st.markdown("**Sources:**")
                            for src in result.sources:
                                st.markdown(
                                    f"- {src.file_name} — Page {src.page_number or 'N/A'}, "
                                    f"Section: {src.section}"
                                )

                        # Metrics
                        m = result.metrics
                        st.markdown("**Metrics:**")
                        metrics_data = {
                            "Retrieval Relevance": m.retrieval_relevance,
                            "Context Relevance": m.context_relevance,
                            "Answer Faithfulness": m.answer_faithfulness,
                            "Answer Completeness": m.answer_completeness,
                            "Citation Correctness": m.citation_correctness,
                        }
                        for name, val in metrics_data.items():
                            color = "#22c55e" if val > 0.7 else "#eab308" if val > 0.4 else "#ef4444"
                            st.markdown(
                                f'{name}: <strong style="color:{color}">{val:.2f}</strong>',
                                unsafe_allow_html=True,
                            )

st.markdown(
    '<div style="text-align:center; color:#6b7280; padding:24px 0 8px;">'
    'Built by Akash Goswami ❤️</div>',
    unsafe_allow_html=True,
)
