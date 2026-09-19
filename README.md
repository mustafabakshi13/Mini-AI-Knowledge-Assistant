# Mini AI Knowledge Assistant

A lightweight, transparent, and educational **Retrieval-Augmented Generation (RAG)** application built with Python, Google Gemini, NumPy, and Streamlit.

---

## 1. Overview

The **Mini AI Knowledge Assistant** allows users to ask natural-language questions and receive factual answers grounded strictly in user-supplied documents (PDFs). Instead of relying on general model pretraining or hallucinating unknown facts, the application extracts and chunks document text, computes dense semantic embeddings, indexes vectors locally, retrieves the most relevant passages via cosine similarity, and synthesizes answers with Google Gemini 2.5 Flash while providing exact document and page citations.

---

## 2. Problem Statement

Standard Large Language Models (LLMs) suffer from two core limitations:
1. **Knowledge Cutoffs**: They cannot know information outside their training datasets or private organization data.
2. **Hallucination**: When asked about specialized or unfamiliar facts, LLMs frequently generate plausible-sounding but factually incorrect statements.

**Retrieval-Augmented Generation (RAG)** solves both problems by separating *factual retrieval* from *linguistic synthesis*. By providing relevant source excerpts directly in the prompt context and enforcing strict anti-hallucination rules, the assistant ensures factual correctness and verifiable citations.

---

## 3. Features

* **PDF Document Ingestion**: Multi-page PDF extraction using `pypdf` with text normalization, whitespace cleanup, and extraction artifact removal.
* **Intelligent Boundary-Aware Chunking**: Configurable chunk size and sliding-window overlap with boundary snapping (paragraphs, sentences, punctuation, word boundaries) to prevent cut words and context fragmentation.
* **Semantic Embeddings**: Dense vector representations generated with Google Gemini (`gemini-embedding-2`) using the official `google-genai` SDK.
* **Lightweight NumPy Vector Store**: Zero-bloat vector storage computing exact cosine similarity:
  $$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
  Persisted locally to compressed binary (`embeddings.npz`), JSON metadata (`chunks.json`), and file hashes (`hashes.json`).
* **Strict Grounded Generation**: Context-bound prompt engineering using `gemini-2.5-flash` at `temperature=0.0`. Out-of-domain or unanswerable queries cleanly produce: *"I cannot find the answer to this question in the provided documents."*
* **Verifiable Source Citations**: Every response cites the originating document name, page number, chunk ID, and similarity score.
* **Multiple Document Support**: Ingest, search, and manage multiple PDF documents simultaneously.
* **Interactive Document Upload**: Drag-and-drop PDF file upload directly from the Streamlit sidebar.
* **SHA-256 Duplicate Detection**: Cryptographic hashing prevents redundant indexing and duplicate embedding API calls.
* **Session Conversation History**: Chronological chat history preserved in Streamlit session state, with each question evaluated as an independent query to prevent prompt bloat and context contamination.
* **Dynamic Retrieval Controls**: Live sliders for Top-K chunks (1–8) and relevance score threshold (0.0–1.0).
* **Index Management**: Sidebar controls to rebuild the entire knowledge base from scratch or clear chat history.
* **Retrieval Evaluation Benchmark**: Built-in benchmark suite (`evaluate_retrieval.py` and `evaluation/questions.json`) computing Source Hit@K and Page-Exact Hit@K accuracy.

---

## 4. Architecture

```
                    ┌────────────────────────┐
                    │   Streamlit Web UI     │
                    │       (app.py)         │
                    └───────────┬────────────┘
                                │ User Question
                                ▼
                    ┌────────────────────────┐
                    │       RAGEngine        │
                    │   (src/rag_engine.py)  │
                    └───────────┬────────────┘
                                │
                      Query Text│
                                ▼
                    ┌────────────────────────┐
                    │     GeminiEmbedder     │
                    │  (src/embeddings.py)   │
                    └───────────┬────────────┘
                                │
                 Query Vector [d]
                                ▼
                    ┌────────────────────────┐
                    │      VectorStore       │
                    │  (src/vector_store.py) │
                    │   [NumPy Cosine Sim]   │
                    └───────────┬────────────┘
                                │
                 Top-K Chunks (Score >= Threshold)
                                ▼
                    ┌────────────────────────┐
                    │ Strict Grounding Prompt│
                    │  + System Instruction  │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │    Gemini 2.5 Flash    │
                    │     (google-genai)     │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │  Grounded Answer +     │
                    │  Source Citations      │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │  Streamlit UI Display  │
                    │  (Answer + Sources)    │
                    └────────────────────────┘

Document Ingestion Pipeline:
PDF Document(s) (data/ or data/uploads/)
      ↓
src/loader.py (pypdf text extraction, cleaning & SHA-256 duplicate detection)
      ↓
src/chunker.py (boundary-snapped chunking with overlap & metadata preservation)
      ↓
src/embeddings.py (GeminiEmbedder batch embedding via google-genai)
      ↓
src/vector_store.py (NumPy vector storage -> embeddings.npz, chunks.json, hashes.json)
```

---

## 5. Technology Stack

* **Language**: Python 3.10+ (tested on Python 3.14.3, Windows 10/11)
* **LLM & Embeddings**: Google Gemini API via official `google-genai` SDK
  * LLM Model: `gemini-2.5-flash`
  * Embedding Model: `gemini-embedding-2`
* **Document Processing**: `pypdf` (v6+)
* **Vector Indexing & Math**: `numpy` (v2+)
* **User Interface**: `streamlit` (v1.30+)
* **Configuration**: `python-dotenv`
* **Testing**: Python `unittest` (47 automated offline tests)

---

## 6. Project Structure

```
mini-ai-knowledge-assistant/
├── .env.example             # Template for API keys and configuration
├── .env                     # Local environment settings (git-ignored)
├── .gitignore               # Excludes virtual environments, secrets, caches, and uploaded docs
├── requirements.txt         # Pinned production dependencies
├── README.md                # Comprehensive documentation
├── main.py                  # Entry point for environment verification
├── app.py                   # Streamlit web application with sidebar controls & upload
├── evaluate_retrieval.py    # Retrieval evaluation benchmark script (Hit@K metrics)
├── verify_ingestion.py      # Standalone verification for PDF ingestion & chunking
├── verify_retrieval.py      # Standalone verification for embedding & vector search
├── verify_rag.py            # Standalone verification for RAG answering & grounding
├── data/                    # Knowledge documents & vector store
│   ├── create_sample_pdf.py # Helper script to generate sample test PDFs
│   ├── sample.pdf           # Sample multi-page RAG test document
│   ├── sample_ml_primer.pdf # Sample multi-page Machine Learning test document
│   ├── uploads/             # User-uploaded PDF storage (git-ignored)
│   └── vector_store/        # Persisted vector index (git-ignored)
│       ├── embeddings.npz   # Compressed numpy array of vectors
│       ├── chunks.json      # Chunk text and metadata records
│       └── hashes.json      # File SHA-256 hashes for duplicate prevention
├── evaluation/              # Benchmark evaluation datasets
│   └── questions.json       # Ground-truth retrieval evaluation questions
├── src/                     # Core application modules
│   ├── __init__.py
│   ├── config.py            # Centralized configuration and .env loading
│   ├── loader.py            # PDF document extraction, cleaning, and SHA-256 hashing
│   ├── chunker.py           # Boundary-aware text chunking with overlap
│   ├── embeddings.py        # Gemini embedding generation using google-genai SDK
│   ├── vector_store.py      # Lightweight NumPy vector store with cosine similarity
│   ├── rag_engine.py        # Grounded RAG answering & prompt orchestration
│   └── ui_helpers.py        # UI presentation, validation, and status utilities
└── tests/                   # Modular offline unit test suite (47 tests)
    ├── __init__.py
    ├── test_setup.py        # Environment verification tests
    ├── test_loader.py       # PDF extraction, cleaning, and corrupt file tests
    ├── test_chunker.py      # Chunking, overlap, and boundary tests
    ├── test_embeddings.py   # Embedding parameter & offline validation tests
    ├── test_vector_store.py # Vector store, similarity math, & persistence tests
    ├── test_rag_engine.py   # Grounding prompt, threshold, and RAG logic tests
    ├── test_ui_helpers.py   # UI validation and status formatting tests
    └── test_phase6.py       # Multi-doc, upload deduplication, evaluation & UI controls tests
```

---

## 7. Setup & Installation

### Prerequisites
* Python 3.10 or higher.
* A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/).

### 1. Clone the Repository
```bash
git clone <repository_url>
cd mini-ai-knowledge-assistant
```

### 2. Create and Activate Virtual Environment
* **On Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **On macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the example environment file and configure your API key:
```bash
# Windows PowerShell:
Copy-Item .env.example .env

# macOS / Linux:
cp .env.example .env
```
Open `.env` and set your key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=50
DEFAULT_TOP_K=4
RETRIEVAL_SCORE_THRESHOLD=0.2
```

### 5. Verify Setup
Run the environment verification script:
```bash
python main.py
```

---

## 8. Indexing Documents

Before running queries, documents must be processed and indexed into the vector store.

### Automated Indexing via CLI
Run the retrieval verification script to index the sample documents (`data/sample.pdf` and `data/sample_ml_primer.pdf`):
```bash
python verify_retrieval.py
```
*(If no API key is provided, the script runs a deterministic semantic vector simulation for offline demonstration).*

### Rebuilding Index from Web UI
You can rebuild the entire vector store at any time from the Streamlit UI by clicking the **Rebuild Index** button in the sidebar.

---

## 9. Running the Application

Launch the Streamlit web interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Usage Walkthrough:
1. **Inspect Knowledge Base**: The sidebar confirms ready status, document count, and chunk count.
2. **Upload Documents**: Use the file uploader in the sidebar to upload additional PDFs. Duplicate files are detected via SHA-256 and rejected automatically.
3. **Tune Retrieval**: Adjust the **Top-K** slider (1–8 chunks) and **Relevance Score Threshold** slider (0.0–1.0).
4. **Ask Questions**: Enter a question (e.g., *"What is Retrieval-Augmented Generation?"*).
5. **Inspect Answers & Sources**: View the grounded answer and expand source cards to inspect originating document, page number, chunk ID, similarity score, and verbatim snippet.
6. **Review Conversation**: Past questions and answers remain visible in session history. Each query is evaluated independently to avoid context drift.

---

## 10. Retrieval Evaluation Benchmark

The project includes an offline-capable retrieval evaluation script:
```bash
python evaluate_retrieval.py
```
* **Dataset**: `evaluation/questions.json` (5 benchmark questions mapped to ground-truth document pages).
* **Metrics**:
  * **Source Retrieval Accuracy (Hit@4)**: Measures whether the correct source document is retrieved in the Top-4 chunks.
  * **Page-Exact Retrieval Accuracy (Hit@4)**: Measures whether the exact source page is retrieved.
* **Results**:
  ```text
  Questions Evaluated          : 5
  Source Retrieval Accuracy    : 100.0% (5/5)
  Page-Exact Retrieval Accuracy: 100.0% (5/5)
  ```

---

## 11. Automated Test Suite

The project includes 47 automated unit tests covering all modules, edge cases, and failure modes. The test suite runs **100% offline** without requiring an external API key:
```bash
python -m unittest discover tests
```
* **Total Tests**: 47
* **Failures**: 0
* **Errors**: 0
* **Status**: OK (100% pass)

---

## 12. Known Limitations

* **Scanned / Image PDFs**: Text extraction relies on `pypdf`. Image-only or scanned PDFs without embedded text layers require OCR (e.g. Tesseract), which is not included to keep dependencies lightweight.
* **Small Evaluation Set**: The included evaluation benchmark comprises 5 focused questions designed to verify retrieval correctness across multiple documents. It is not an industry-scale benchmark.
* **Local In-Memory Vector Store**: Vector storage uses local NumPy files (`.npz` and `.json`). While ideal for student demonstrations and small collections (thousands of chunks), it is not designed for distributed enterprise scale (millions of vectors).
* **API Connectivity**: Live semantic embeddings and answer generation require outbound internet connectivity to the Google Gemini API.

---

## 13. Future Improvements

* **OCR Integration**: Integrate OCR for scanned and image-based PDFs.
* **Hybrid Retrieval**: Combine dense semantic vector search with sparse keyword search (BM25) and reciprocal rank fusion (RRF).
* **Cross-Encoder Reranking**: Apply a neural reranker to reorder Top-K retrieved chunks before prompt generation.
* **Cloud Vector Storage**: Optional adapter for Pinecone, Qdrant, or ChromaDB for large-scale production deployments.
* **User Authentication**: Add user accounts and multi-tenant document isolation.
