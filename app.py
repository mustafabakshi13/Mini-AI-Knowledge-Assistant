"""Streamlit application for Mini AI Knowledge Assistant (Phase 6).

Provides an interactive interface supporting:
- Multi-document knowledge base inspection
- Dynamic Top-K and similarity threshold retrieval controls
- PDF upload and duplicate detection (SHA-256)
- Knowledge base reindexing
- Conversation history tracking in session state
- Expandable source citation inspection
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

# Guarantee that the repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.config import (
    DATA_DIR,
    DEFAULT_TOP_K,
    GEMINI_API_KEY,
    RETRIEVAL_SCORE_THRESHOLD,
    UPLOADS_DIR,
    VECTOR_STORE_DIR,
    get_gemini_api_key,
)
from src.loader import compute_file_hash, load_pdf
from src.chunker import chunk_document_pages
from src.rag_engine import RAGEngine, RAGResponse
from src.vector_store import VectorStore
from src.ui_helpers import format_source_display, get_kb_status, validate_question


# 1. Page Configuration
st.set_page_config(
    page_title="Mini AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
)


def index_pdf_file(
    file_bytes: bytes,
    filename: str,
    store: VectorStore,
    has_active_key: bool = False,
    api_key: Optional[str] = None,
) -> tuple[bool, str]:
    """Processes, chunks, embeds, and indexes a single PDF document into the VectorStore."""
    if not file_bytes:
        return False, f"File '{filename}' contains no data (empty buffer)."

    file_hash = compute_file_hash(file_bytes)
    if store.is_document_indexed(file_hash):
        return False, f"Document '{filename}' is already indexed (duplicate detected)."

    # Save to controlled uploads directory
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = UPLOADS_DIR / filename
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    try:
        pages = load_pdf(temp_path)
        chunks = chunk_document_pages(pages)

        if not chunks:
            temp_path.unlink(missing_ok=True)
            return False, f"No extractable text chunks found in '{filename}'."

        if has_active_key:
            from src.embeddings import GeminiEmbedder
            effective_key = api_key or get_gemini_api_key()
            embedder = GeminiEmbedder(api_key=effective_key)
            chunks_with_emb = embedder.embed_chunks(chunks)
        else:
            # Deterministic simulation vectors when running offline
            import numpy as np
            chunks_with_emb = []
            keywords = ["rag", "retrieval", "chunk", "learning", "gradient", "optimization", "model", "vector"]
            target_dim = store.dimension if store.dimension > 0 else len(keywords)
            for c in chunks:
                lower = c.text.lower()
                v = [float(lower.count(k) + 0.1) for k in keywords]
                if target_dim != len(keywords):
                    if len(v) < target_dim:
                        v = v + [0.1] * (target_dim - len(v))
                    else:
                        v = v[:target_dim]
                norm = float(np.linalg.norm(v))
                norm = norm if norm > 0.0 else 1.0
                chunks_with_emb.append((c, [float(x / norm) for x in v]))

        store.add_chunks(chunks_with_emb)
        store.register_document_hash(file_hash, filename)
        store.save()
        return True, f"Successfully indexed '{filename}' ({len(chunks)} chunks)."
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        return False, f"Failed to process '{filename}': {exc}"


# 2. Resource Caching: Load VectorStore and RAGEngine once per session
@st.cache_resource
def load_rag_components():
    """Loads the persisted vector store and initializes the RAGEngine.

    Validates that stored vectors match the active embedding model's dimensions.
    If a dimension mismatch is detected (e.g. offline 8D vectors vs. real 3072D Gemini embeddings),
    automatically rebuilds the index from documents found in DATA_DIR and UPLOADS_DIR.
    """
    store = VectorStore(storage_dir=VECTOR_STORE_DIR)
    is_loaded = False
    current_key = get_gemini_api_key()
    has_active_key = bool(current_key) and current_key != "your_gemini_api_key_here"
    expected_dim = 3072 if has_active_key else 8

    try:
        store.load()
        is_loaded = True
        # Verify dimension consistency with active provider
        if store.count > 0 and store.dimension != expected_dim:
            is_loaded = False
    except Exception:
        is_loaded = False

    if not is_loaded:
        store.clear()
        base_pdfs = sorted(list(DATA_DIR.glob("*.pdf")))
        if UPLOADS_DIR.is_dir():
            base_pdfs.extend(sorted(list(UPLOADS_DIR.glob("*.pdf"))))
        if base_pdfs:
            for pdf_path in base_pdfs:
                try:
                    with open(pdf_path, "rb") as f:
                        index_pdf_file(
                            f.read(),
                            pdf_path.name,
                            store,
                            has_active_key=has_active_key,
                            api_key=current_key,
                        )
                except Exception:
                    pass
            is_loaded = store.count > 0

    engine = RAGEngine(vector_store=store, api_key=current_key) if is_loaded else None
    return store, engine, is_loaded


def main():
    # 3. Session State Initialization for Conversation History
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    current_key = get_gemini_api_key()
    has_active_key = bool(current_key) and current_key != "your_gemini_api_key_here"

    # 4. System Initialization & Knowledge Base Status
    store, engine, is_loaded = load_rag_components()
    kb_info = get_kb_status(store, is_loaded)

    # 5. Sidebar Controls & Management
    with st.sidebar:
        st.header("Knowledge Base")
        if kb_info["is_ready"]:
            st.success(f"✅ Ready ({kb_info['document_count']} docs, {kb_info['chunk_count']} chunks)")
            with st.expander("Indexed Documents", expanded=True):
                for doc in kb_info["documents"]:
                    st.markdown(f"- 📄 `{doc}`")
        else:
            st.error("❌ No documents indexed")
            st.info("Run `python verify_retrieval.py` or upload a PDF below.")

        st.markdown("---")
        st.subheader("Retrieval Controls")
        top_k = st.slider(
            "Top-K Retrieved Chunks",
            min_value=1,
            max_value=8,
            value=DEFAULT_TOP_K,
            help="Number of most relevant chunks retrieved for prompt context.",
        )
        score_threshold = st.slider(
            "Relevance Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=RETRIEVAL_SCORE_THRESHOLD,
            step=0.05,
            help="Minimum cosine similarity required for a chunk to be accepted.",
        )

        st.markdown("---")
        st.subheader("Add Documents")

        # Display persistent feedback from previous indexing action across reruns
        if "indexing_results" in st.session_state:
            for success, msg in st.session_state.pop("indexing_results", []):
                if success:
                    st.success(msg)
                else:
                    st.warning(msg)

        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDFs to expand the knowledge base.",
        )

        if uploaded_files:
            if st.button("Index Uploaded Files", type="secondary"):
                with st.spinner("Processing and indexing documents..."):
                    progress_bar = st.progress(0)
                    indexing_results = []
                    for i, uploaded_file in enumerate(uploaded_files):
                        file_bytes = (
                            uploaded_file.getvalue()
                            if hasattr(uploaded_file, "getvalue")
                            else uploaded_file.read()
                        )
                        success, msg = index_pdf_file(
                            file_bytes,
                            uploaded_file.name,
                            store,
                            has_active_key=has_active_key,
                            api_key=current_key,
                        )
                        indexing_results.append((success, msg))
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    st.session_state["indexing_results"] = indexing_results
                    load_rag_components.clear()
                    st.rerun()

        st.markdown("---")
        st.subheader("Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Rebuild Index"):
                with st.spinner("Rebuilding knowledge base index..."):
                    store.clear()
                    # Collect all sample and uploaded PDFs
                    all_pdfs = list(DATA_DIR.glob("*.pdf"))
                    if UPLOADS_DIR.is_dir():
                        all_pdfs.extend(list(UPLOADS_DIR.glob("*.pdf")))
                    for pdf_path in all_pdfs:
                        with open(pdf_path, "rb") as f:
                            index_pdf_file(
                                f.read(),
                                pdf_path.name,
                                store,
                                has_active_key=has_active_key,
                                api_key=current_key,
                            )
                    load_rag_components.clear()
                    st.session_state["indexing_results"] = [(True, "Index rebuilt successfully!")]
                    st.rerun()

        with col2:
            if st.button("Clear Chat"):
                st.session_state.conversation = []
                st.rerun()

    # 6. Main Content Area
    st.title("Mini AI Knowledge Assistant")
    st.caption("Ask questions grounded strictly in the provided document knowledge base.")
    st.markdown("---")

    if not has_active_key:
        st.warning(
            "⚠️ **Gemini API Key is not configured.**\n\n"
            "Configure `GEMINI_API_KEY` in Streamlit Cloud Secrets (App Settings → Secrets) or in `.env` locally "
            "to enable live Gemini embeddings and answers."
        )

    # Question Input
    st.subheader("Ask a Question")
    question_input = st.text_area(
        label="Question",
        placeholder="Enter your question (e.g., 'What is Retrieval-Augmented Generation?' or 'How does gradient descent work?')...",
        height=90,
        label_visibility="collapsed",
    )

    ask_button = st.button("Ask Question", type="primary")

    if ask_button:
        is_valid, clean_q = validate_question(question_input)
        if not is_valid:
            st.warning(clean_q)
        elif not is_loaded or store.count == 0:
            st.error("Knowledge base is empty. Please index documents first.")
        else:
            with st.spinner("Retrieving relevant information and generating answer..."):
                try:
                    # Note: Each question is answered independently by the RAG engine
                    response: RAGResponse = engine.answer_question(
                        clean_q, top_k=top_k, score_threshold=score_threshold
                    )

                    # Append to conversation history (session-level UI display)
                    st.session_state.conversation.append({
                        "question": clean_q,
                        "answer": response.answer,
                        "sources": response.sources,
                    })
                except Exception as exc:
                    st.error(f"Error answering question: {exc}")

    # 7. Conversation History Display
    if st.session_state.conversation:
        st.markdown("---")
        st.subheader("Conversation History")
        st.caption(
            "Note: Each question is evaluated as an independent RAG query to prevent context contamination."
        )

        # Display history in reverse chronological order (newest first)
        for idx, turn in enumerate(reversed(st.session_state.conversation)):
            turn_num = len(st.session_state.conversation) - idx
            with st.container():
                st.markdown(f"#### Question {turn_num}: *{turn['question']}*")
                st.markdown(turn["answer"])

                if turn["sources"]:
                    with st.expander(f"Inspect Sources ({len(turn['sources'])} chunks retrieved)"):
                        for src in turn["sources"]:
                            title = format_source_display(src)
                            st.markdown(f"**{title}**")
                            st.markdown(f"- **Chunk ID:** `{src['chunk_id']}`")
                            st.markdown(f"> *{src['preview']}*")
                else:
                    st.caption("ℹ️ No authoritative sources found for this question.")
                st.markdown("---")


if __name__ == "__main__":
    main()
