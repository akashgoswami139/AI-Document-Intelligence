# 🧠 AI Document Intelligence RAG System

> **Upload any document → AI automatically understands it → Ask questions → Get grounded answers with citations.**

An intelligent Retrieval-Augmented Generation (RAG) system that goes beyond simple "PDF → chunks → chatbot" pipelines. The AI dynamically analyzes every uploaded document to determine its type, structure, entities, and purpose — then uses this intelligence for smarter retrieval and more accurate answers.

---

## 🏗️ Architecture

```
                ┌──────────────────┐
                │ Upload Documents │
                │ (PDF/DOCX/TXT/   │
                │  MD/CSV/XLSX)    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Document Loader  │
                │ Format-specific  │
                │ extraction       │
                └────────┬─────────┘
                         │
                         ▼
              ┌───────────────────────┐
              │  AI Document Analyzer │
              │  • Type inference     │
              │  • Structure detection│
              │  • Entity extraction  │
              │  • Summary generation │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Structure-Preserving  │
              │ Parser                │
              │ • Headings            │
              │ • Tables              │
              │ • Page boundaries     │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Intelligent Chunking  │
              │ • Structure-Aware     │
              │ • Naive (baseline)    │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Embedding Generation  │
              │ Jina Embeddings       │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    ChromaDB Vector    │
              │    Store              │
              │  (embeddings + meta)  │
              └───────────┬───────────┘
                          │
                          ▼
User Question ──→ Query Understanding ──→ Retrieval ──→ RAG Chain ──→ Answer + Citations
```

---

## ✨ Features

### Document Intelligence
- **Automatic document classification** — the LLM infers document type (invoice, resume, research paper, contract, etc.) without hardcoded rules
- **Structure detection** — identifies sections, headings, entities, dates, organizations
- **AI summary** — generates a concise summary of every uploaded document
- **Multi-format support** — PDF, DOCX, TXT, Markdown, CSV, XLSX

### Smart Processing
- **Structure-preserving parsing** — headings, tables, page boundaries maintained
- **Dual chunking strategies** — structure-aware (production) and naive (baseline) for comparison
- **Rich metadata** — every chunk carries document_id, file_name, document_type, page_number, section

### Intelligent Retrieval
- **Query understanding** — LLM analyzes questions to determine intent and target documents
- **Metadata-filtered search** — questions about invoices prioritize invoice chunks
- **Multi-document queries** — supports questions spanning multiple uploaded documents
- **Score transparency** — similarity scores exposed for every retrieved chunk

### Anti-Hallucination
- **Strict grounding** — system prompt enforces answers from retrieved context only
- **Source citations** — every answer includes file name, page number, and section
- **Honest uncertainty** — system explicitly says "I couldn't find this information" when appropriate

### Conversation
- **Follow-up questions** — conversation memory maintains context
- **Sliding window** — keeps last 10 exchanges for relevant context without token bloat

### Evaluation
- **LLM-as-judge** — automated evaluation of retrieval relevance, answer faithfulness, completeness
- **Citation correctness** — programmatic verification that cited sources exist
- **Auto-generated test questions** — creates evaluation data from uploaded documents

---

## 🛠️ Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Orchestration** | LangChain (LCEL) | Modern pipe-based chains, provider-agnostic |
| **LLM** | Google Gemini | Analysis, query understanding, answers, and evaluation |
| **Embeddings** | Jina API | Document and query vectors |
| **Vector DB** | ChromaDB | Zero-config, metadata filtering, LangChain integration |
| **UI** | Streamlit | Rapid prototyping, native chat components |
| **PDF** | pypdf | Pure Python, page-level extraction |
| **DOCX** | python-docx | Heading/table structure preservation |
| **Data** | pandas + openpyxl | CSV/XLSX handling |
| **Models** | Pydantic v2 | Type-safe data contracts across pipeline stages |

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Gemini API key and Jina API key

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd document-intelligence-rag

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY and JINA_API_KEY
```

### Run

```bash
streamlit run app.py
```

---

## 🔑 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *required* | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Gemini model for analysis and Q&A |
| `JINA_API_KEY` | *required* | Jina API key for embeddings |
| `JINA_EMBEDDING_MODEL` | `jina-embeddings-v5-text-small` | Jina embedding model |
| `CHUNK_SIZE` | `1000` | Maximum chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Number of chunks to retrieve per query |

---

## 📖 How RAG Works in This System

### 1. Document Processing Pipeline

```
Upload → Load → Analyze → Parse → Chunk → Embed → Store
```

1. **Load**: Format-specific extraction preserving page/sheet structure
2. **Analyze**: LLM examines the first ~8000 chars to infer type, structure, entities
3. **Parse**: Detect headings (markdown, ALL CAPS, numbered) and split into sections
4. **Chunk**: Structure-aware splitting respects section boundaries; tables stay atomic
5. **Embed**: Convert chunks to vectors using the configured embedding model
6. **Store**: Index in ChromaDB with rich metadata for filtered retrieval

### 2. Query Pipeline

```
Question → Analyze → Filter → Search → Format → LLM → Answer + Citations
```

1. **Query Analysis**: LLM determines intent, target document types, and relevant sections
2. **Metadata Filtering**: Build ChromaDB filters from query analysis
3. **Similarity Search**: Retrieve top-K chunks matching the query within filters
4. **Context Formatting**: Structure retrieved chunks with source annotations
5. **LLM Generation**: Anti-hallucination prompt enforces grounded answering
6. **Citation Extraction**: Deduplicate and format source references

---

## ✂️ Chunking Strategy

### Why Structure-Aware Chunking?

**Problem:** Naive fixed-size chunking can split a table mid-row, put a heading in one chunk and its content in another, or merge unrelated sections.

**Solution:** Structure-aware chunking:
- Never splits across section boundaries
- Keeps tables as atomic chunks
- Preserves heading → content relationships
- Uses `RecursiveCharacterTextSplitter` within long sections for controlled splitting

### Comparison

| Aspect | Naive | Structure-Aware |
|--------|-------|-----------------|
| Section boundaries | Ignored | Respected |
| Table preservation | Tables split randomly | Tables kept atomic |
| Metadata quality | Generic ("Unknown" section) | Rich (actual section titles) |
| Retrieval precision | Lower | Higher |
| Use case | Baseline/comparison | Production |

---

## 🔎 Retrieval Strategy

### Multi-Stage Retrieval

1. **Over-fetch**: Retrieve `top_k + 3` chunks for better recall
2. **Metadata filtering**: Narrow search space using query analysis (document type, sections)
3. **Graceful fallback**: If filtered search returns nothing, retry without filters
4. **Score inspection**: All similarity scores are exposed for transparency

### Why Query Understanding?

Without query analysis, "What is the invoice total?" might retrieve chunks from a co-uploaded research paper. By analyzing the query first, we:
- Build metadata filters targeting invoice chunks
- Prioritize sections containing "total" or "summary"
- Improve precision dramatically

---

## 🤖 AI Document Classification

The system does **not** hardcode document types. Instead:

1. The first ~8000 characters of each document are sent to the LLM
2. The LLM returns structured JSON with:
   - Document type (Invoice, Resume, Research Paper, etc.)
   - Main topic and purpose
   - Detected sections and entities
   - Content composition (text/tables/mixed)
3. This analysis drives chunking strategy and retrieval filtering

**Example outputs:**

```
Invoice → Type: Invoice, Sections: [Vendor Info, Items, Taxes, Total]
Resume → Type: Resume, Sections: [Education, Experience, Skills, Projects]
Research Paper → Type: Research Paper, Sections: [Abstract, Methodology, Results]
```

---

## 🧪 Evaluation Methodology

### Metrics

| Metric | Method | Description |
|--------|--------|-------------|
| **Retrieval Relevance** | LLM judge | Are retrieved chunks relevant to the question? |
| **Context Relevance** | LLM judge | Does the context support the answer? |
| **Answer Faithfulness** | LLM judge | Is the answer grounded in context (no hallucination)? |
| **Answer Completeness** | LLM judge | Does the answer fully address the question? |
| **Citation Correctness** | Programmatic | Do cited sources exist in retrieved chunks? |

### Process
1. LLM generates test questions from uploaded documents
2. Each question runs through the full RAG pipeline
3. LLM judges evaluate each dimension independently
4. Average scores are computed and displayed

---

## ⚠️ Limitations

1. **No OCR** — Scanned PDFs with image-only content won't extract text
2. **LLM cost** — Every document upload and question incurs Gemini API costs
3. **No persistence** — Vector store is in-memory per session
4. **No authentication** — Single-user system
5. **English-centric** — Prompts optimized for English documents
6. **Large documents** — Files >100 pages may hit token limits during analysis
7. **No reranking** — Uses similarity scores only (no cross-encoder reranking)

---

## 🔮 Future Improvements

- **Hybrid retrieval** — Combine BM25 keyword search with dense embeddings using Reciprocal Rank Fusion
- **Cross-encoder reranking** — Cohere Rerank or BGE for re-scoring retrieved chunks
- **OCR support** — PyMuPDF + Tesseract for scanned documents
- **Persistent storage** — ChromaDB persistent directory or Qdrant server
- **Streaming responses** — Token-by-token streaming for better UX
- **Multi-modal** — Image/chart understanding within documents
- **Authentication** — Multi-user support with isolated document collections
- **Docker deployment** — Containerized production deployment
- **LangSmith tracing** — Full observability for production debugging

---

## 📁 Project Structure

```
document-intelligence-rag/
│
├── app.py                      # Streamlit UI (5 tabs)
├── requirements.txt            # Dependencies
├── .env.example                # Environment variable template
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── models.py               # Pydantic data models
│   ├── prompts.py              # All prompt templates
│   ├── document_loader.py      # Multi-format file loading
│   ├── document_analyzer.py    # LLM-powered analysis
│   ├── parser.py               # Structure-preserving parsing
│   ├── chunker.py              # Naive + structure-aware chunking
│   ├── embeddings.py           # Jina embedding factory
│   ├── jina_embeddings.py      # Jina API client
│   ├── llm.py                  # Shared Gemini model factory
│   ├── vector_store.py         # ChromaDB operations
│   ├── query_analyzer.py       # Query intent understanding
│   ├── retriever.py            # Multi-stage retrieval pipeline
│   ├── rag_chain.py            # LLM chain + memory
│   └── evaluation.py           # LLM-as-judge evaluation
│
└── data/
    └── uploads/                # Temporary uploaded files
```

---

## 📄 License

MIT
