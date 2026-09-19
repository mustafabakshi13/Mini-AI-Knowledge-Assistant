"""Helper utilities for Streamlit UI presentation and validation (Phase 5)."""

from typing import Any, Dict, List, Optional, Tuple
from src.config import (
    DEFAULT_TOP_K,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_MODEL,
)
from src.vector_store import VectorStore


def validate_question(question: Optional[str]) -> Tuple[bool, str]:
    """Validates user question input before submission.

    Returns:
        Tuple of (is_valid, error_or_clean_message)
    """
    if question is None:
        return False, "Please enter a question."

    clean = question.strip()
    if not clean:
        return False, "Please enter a question."

    return True, clean


def get_kb_status(store: Optional[VectorStore], is_loaded: bool) -> Dict[str, Any]:
    """Extracts status summary information for the knowledge base."""
    if not is_loaded or store is None or store.count == 0:
        return {
            "is_ready": False,
            "document_count": 0,
            "documents": [],
            "chunk_count": 0,
            "embedding_model": GEMINI_EMBEDDING_MODEL,
            "llm_model": GEMINI_MODEL,
            "top_k": DEFAULT_TOP_K,
        }

    unique_docs = sorted(
        list(set(c.get("metadata", {}).get("source", "Unknown") for c in store.chunks_data))
    )

    return {
        "is_ready": True,
        "document_count": len(unique_docs),
        "documents": unique_docs,
        "chunk_count": store.count,
        "embedding_model": GEMINI_EMBEDDING_MODEL,
        "llm_model": GEMINI_MODEL,
        "top_k": DEFAULT_TOP_K,
    }


def format_source_display(source_dict: Dict[str, Any]) -> str:
    """Formats a source citation title for expander presentation."""
    doc = source_dict.get("source", "Unknown")
    page = source_dict.get("page", "?")
    score = source_dict.get("similarity_score", 0.0)
    return f"📄 {doc} — Page {page} (Similarity: {score:.4f})"
