"""Retrieval Evaluation Script (Phase 6).

Evaluates the retrieval accuracy of the Mini AI Knowledge Assistant
against a defined benchmark dataset (evaluation/questions.json).
Measures Source Hit@K and Page Hit@K metrics.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from src.config import (
    BASE_DIR,
    DATA_DIR,
    DEFAULT_TOP_K,
    GEMINI_API_KEY,
    VECTOR_STORE_DIR,
)
from src.loader import compute_file_hash, load_pdf
from src.chunker import chunk_document_pages
from src.vector_store import VectorStore


def run_evaluation(top_k: int = 4) -> Dict[str, Any]:
    print("=" * 70)
    print("  RAG Retrieval Evaluation Benchmark")
    print("=" * 70)

    eval_file = BASE_DIR / "evaluation" / "questions.json"
    if not eval_file.is_file():
        raise FileNotFoundError(f"Evaluation file not found at {eval_file}")

    with open(eval_file, "r", encoding="utf-8") as f:
        benchmark_questions: List[Dict[str, Any]] = json.load(f)

    print(f"Benchmark File     : {eval_file.name}")
    print(f"Total Questions    : {len(benchmark_questions)}")
    print(f"Evaluation Metric  : Top-{top_k} Retrieval Accuracy (Hit@{top_k})")
    print("-" * 70)

    # 1. Ensure test documents are indexed in VectorStore
    store = VectorStore(storage_dir=VECTOR_STORE_DIR)
    has_active_key = bool(GEMINI_API_KEY) and GEMINI_API_KEY != "your_gemini_api_key_here"

    # Multi-document check: sample.pdf and sample_ml_primer.pdf
    pdf_paths = [DATA_DIR / "sample.pdf", DATA_DIR / "sample_ml_primer.pdf"]
    for p in pdf_paths:
        if not p.is_file():
            from data.create_sample_pdf import main as create_samples
            create_samples()
            break

    try:
        store.load()
    except Exception:
        store.clear()

    # Index any missing documents into store
    _index_missing_documents(store, pdf_paths, has_active_key)

    print(f"Vector Store Status: {store.count} chunks across {len(store.indexed_hashes)} document(s)")
    print("-" * 70)

    # 2. Run queries and compute retrieval metrics
    source_hits = 0
    page_hits = 0
    detailed_results = []

    embedder = None
    if has_active_key:
        from src.embeddings import GeminiEmbedder
        embedder = GeminiEmbedder()

    for idx, item in enumerate(benchmark_questions, start=1):
        q_text = item["question"]
        expected_src = item["expected_source"]
        expected_page = item["expected_page"]

        # Generate query vector
        if has_active_key and embedder is not None:
            q_vec = embedder.embed_text(q_text, task_type="RETRIEVAL_QUERY")
        else:
            # Deterministic simulation vector based on keyword projections
            q_vec = _simulate_query_vector(q_text, store.dimension)

        results = store.search(q_vec, top_k=top_k)

        # Check hits
        source_matched = any(r.metadata.get("source") == expected_src for r in results)
        page_matched = any(
            r.metadata.get("source") == expected_src and r.metadata.get("page") == expected_page
            for r in results
        )

        if source_matched:
            source_hits += 1
        if page_matched:
            page_hits += 1

        top_result = results[0] if results else None
        top_src = top_result.metadata.get("source", "None") if top_result else "None"
        top_page = top_result.metadata.get("page", "?") if top_result else "?"
        top_score = top_result.similarity_score if top_result else 0.0

        status_str = "PASS" if page_matched else ("PARTIAL (Source Only)" if source_matched else "FAIL")
        print(f"Q{idx}: \"{q_text[:45]}...\"")
        print(f"    Expected : {expected_src} (Page {expected_page})")
        print(f"    Top-1    : {top_src} (Page {top_page}) [Score: {top_score:.4f}]")
        print(f"    Status   : [{status_str}]\n")

        detailed_results.append({
            "id": item.get("id"),
            "question": q_text,
            "expected_source": expected_src,
            "expected_page": expected_page,
            "source_hit": source_matched,
            "page_hit": page_matched,
            "top_1_source": top_src,
            "top_1_page": top_page,
            "top_1_score": top_score,
        })

    total_q = len(benchmark_questions)
    source_acc = (source_hits / total_q) * 100
    page_acc = (page_hits / total_q) * 100

    print("=" * 70)
    print("  Evaluation Summary Results")
    print("=" * 70)
    print(f"Questions Evaluated          : {total_q}")
    print(f"Source Retrieval Accuracy    : {source_acc:.1f}% ({source_hits}/{total_q})")
    print(f"Page-Exact Retrieval Accuracy: {page_acc:.1f}% ({page_hits}/{total_q})")
    print("=" * 70)

    return {
        "total_questions": total_q,
        "source_accuracy_pct": round(source_acc, 1),
        "page_accuracy_pct": round(page_acc, 1),
        "detailed_results": detailed_results,
    }


def _index_missing_documents(store: VectorStore, paths: List[Path], has_active_key: bool):
    """Indexes documents that are not yet registered in the vector store."""
    expected_dim = 3072 if has_active_key else 8  # 3072 for Gemini, 8 for offline simulation
    if store.count > 0 and store.dimension != 0 and store.dimension != expected_dim:
        store.clear()

    for p in paths:
        if not p.is_file():
            continue
        file_hash = compute_file_hash(p)
        if store.is_document_indexed(file_hash):
            continue

        pages = load_pdf(p)
        chunks = chunk_document_pages(pages)

        if has_active_key:
            from src.embeddings import GeminiEmbedder
            emb = GeminiEmbedder()
            chunks_with_emb = emb.embed_chunks(chunks)
        else:
            # Deterministic offline vector projection
            chunks_with_emb = []
            keywords = ["rag", "retrieval", "chunk", "learning", "gradient", "optimization", "model", "vector"]
            for c in chunks:
                lower = c.text.lower()
                v = [float(lower.count(k) + 0.1) for k in keywords]
                norm = np.linalg.norm(v)
                norm_v = [float(x / norm) for x in v]
                chunks_with_emb.append((c, norm_v))

        store.add_chunks(chunks_with_emb)
        store.register_document_hash(file_hash, p.name)

    if store.count > 0:
        store.save()


def _simulate_query_vector(q_text: str, dim: int) -> List[float]:
    """Generates a deterministic vector projection for offline evaluation."""
    keywords = ["rag", "retrieval", "chunk", "learning", "gradient", "optimization", "model", "vector"]
    lower = q_text.lower()
    v = [float(lower.count(k) + 0.1) for k in keywords]
    norm = np.linalg.norm(v)
    return [float(x / norm) for x in v]


if __name__ == "__main__":
    run_evaluation()
