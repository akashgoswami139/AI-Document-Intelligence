import io
import logging
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()


from src.chunker import DocumentChunker
from src.document_loader import DocumentLoader
from src.embeddings import get_embedding_model
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
    page_icon="📄",
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
        "chat_history": [],        # list of {"role", "content"}
        "chunking_strategy": "Structure-Aware",
        "initialized": False,
        "processing": False,
        "uploader_version": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ── Initialize Pipeline Components ─────────────────────────────────

@st.cache_resource
def get_cached_embeddings():
    """Cache only the stateless embedding client shared by sessions."""
    return get_embedding_model()


def init_pipeline_components() -> None:
    """Create mutable pipeline components once per browser session."""
    if "vector_store" not in st.session_state:
        embeddings = get_cached_embeddings()
        st.session_state.vector_store = VectorStore(
            embeddings,
            collection_name=f"documents_{uuid.uuid4().hex}",
        )
        st.session_state.query_analyzer = QueryAnalyzer()
        st.session_state.retriever = Retriever(
            st.session_state.vector_store,
            st.session_state.query_analyzer,
        )
        st.session_state.rag_chain = RAGChain(st.session_state.retriever)
    for key, factory in {
        "loader": DocumentLoader,
        "parser": DocumentParser,
        "chunker": DocumentChunker,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = factory()


try:
    init_pipeline_components()
except Exception as exc:
    logger.exception("Pipeline initialization failed")
    st.error(f"Unable to initialize the document intelligence pipeline: {exc}")
    st.stop()


# ── Document Processing Pipeline ───────────────────────────────────

def process_document(uploaded_file) -> str | None:
    """Run the full document processing pipeline on an uploaded file."""
    try:
        file_bytes = uploaded_file.read()
        file_obj = io.BytesIO(file_bytes)
        file_name = uploaded_file.name

        # Step 1: Load
        raw_doc = st.session_state.loader.load(file_name, file_obj)

        # Step 2: Parse
        sections = st.session_state.parser.parse(raw_doc)

        # Step 4: Create metadata
        doc_meta = DocumentMetadata(
            file_name=file_name,
            file_type=raw_doc.file_type,
            document_type="Unknown",
            total_pages=raw_doc.total_pages,
        )

        # Step 5: Chunk
        strategy = st.session_state.get("chunking_strategy", "Structure-Aware")
        if strategy == "Naive":
            chunks = st.session_state.chunker.chunk_naive(sections, doc_meta)
        else:
            chunks = st.session_state.chunker.chunk_structure_aware(sections, doc_meta)

        # Step 6: Index in vector store
        st.session_state.vector_store.add_chunks(chunks)

        # Step 7: Store in session state
        doc_id = doc_meta.document_id
        st.session_state.documents[doc_id] = doc_meta
        st.session_state.raw_documents[doc_id] = raw_doc

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
    st.markdown("---")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt", "md", "csv", "xlsx"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_version}",
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

        for doc_id, doc_meta in st.session_state.documents.items():
            doc_type = doc_meta.document_type
            doc_col, remove_col = st.columns([4, 1])
            with doc_col:
                st.markdown(
                    f'<div style="padding:8px 12px; margin:4px 0; '
                    f'background:rgba(99,102,241,0.1); border-radius:8px; '
                    f'border:1px solid rgba(99,102,241,0.2);">'
                    f'<strong style="color:#e0e0ff;">{doc_meta.file_name}</strong><br/>'
                    f'<span style="color:#a5b4fc; font-size:0.8rem;">{doc_type} · {doc_meta.total_pages} page(s)</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with remove_col:
                if st.button("🗑 Remove", key=f"remove_{doc_id}", help="Remove this document"):
                    # Changelog: remove one session document and reset the uploader to prevent re-upload.
                    st.session_state.vector_store.remove_document(doc_id)
                    for key in ["documents", "raw_documents", "raw_texts"]:
                        st.session_state[key].pop(doc_id, None)
                    st.session_state.uploader_version += 1
                    st.rerun()

    # Reset button
    if st.session_state.documents:
        st.markdown("---")
        if st.button("Clear All Documents", type="secondary", use_container_width=True):
            # Changelog: version the uploader so cleared files are not silently reprocessed.
            st.session_state.vector_store.reset()
            st.session_state.rag_chain.clear_history()
            for key in ["documents", "raw_documents", "raw_texts"]:
                st.session_state[key] = {}
            st.session_state.chat_history = []
            st.session_state.uploader_version += 1
            st.rerun()


# ── Main Area — Tabs ───────────────────────────────────────────────

with st.container():
    st.markdown("## Ask Your Documents")

    if not st.session_state.documents:
        st.info(" Please upload documents to start chat")
    else:
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask me anything about your documents..."):
            # Show user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt,
            })
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Please wait few second"):
                    doc_list = list(st.session_state.documents.values())
                    try:
                        response = st.session_state.rag_chain.query(prompt, doc_list)
                    except Exception as exc:
                        logger.exception("Chat request failed")
                        st.error(f"Unable to answer this question: {exc}")
                        response = None
                if response is not None:
                    st.markdown(response.answer)

            if response is not None:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.answer,
                })
st.markdown(
    '<div style="text-align:center; color:#6b7280; padding:24px 0 8px;">'
    'Built with ❤️ by Akash Goswami</div>',
    unsafe_allow_html=True,
)
