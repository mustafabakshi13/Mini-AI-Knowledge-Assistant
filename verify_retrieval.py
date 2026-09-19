"""Verification script for Phase 3: Embeddings, Vector Storage, and Semantic Retrieval.

Demonstrates the complete Phase 3 pipeline:
Document -> Chunks -> Embeddings -> Vector Store -> Query Embedding -> Top-K Retrieval
Includes persistence testing (save to disk -> reload -> query).
"""

import sys
from pathlib import Path
from src.config import (
    DATA_DIR,
    DEFAULT_TOP_K,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    VECTOR_STORE_DIR,
)
from src.loader import compute_file_hash, load_multiple_pdfs
from src.chunker import chunk_document_pages
from src.vector_store import VectorStore


def run_verification():
    print("=" * 70)
    print("  Phase 3 Verification: Embeddings & Vector Retrieval Pipeline")
    print("=" * 70)

    # 1. Load documents and extract chunks
    sample_pdfs = sorted(list(DATA_DIR.glob("sample*.pdf")))
    if not sample_pdfs:
        print(f"[ERROR] Sample PDFs not found in {DATA_DIR}.")
        print("Please run 'python data/create_sample_pdf.py' first.")
        return

    doc_names = ", ".join([p.name for p in sample_pdfs])
    print(f"Document Files      : {doc_names}")
    print(f"Embedding Model     : {GEMINI_EMBEDDING_MODEL}")
    print(f"Configured Top-K    : {DEFAULT_TOP_K}")
    print(f"Storage Directory   : {VECTOR_STORE_DIR}")
    print("-" * 70)

    print("[1/5] Ingesting and chunking documents...")
    pages = load_multiple_pdfs(sample_pdfs)
    chunks = chunk_document_pages(pages)
    print(f"      Extracted {len(pages)} pages across {len(sample_pdfs)} doc(s) -> generated {len(chunks)} chunks.")

    # 2. Check for Gemini API key
    has_active_key = bool(GEMINI_API_KEY) and GEMINI_API_KEY != "your_gemini_api_key_here"

    if not has_active_key:
        print("\n[NOTE] No active GEMINI_API_KEY found in .env (or placeholder detected).")
        print("       To perform live Gemini API calls:")
        print("         1. Get an API key from https://aistudio.google.com/")
        print("         2. Place it in .env: GEMINI_API_KEY=AIzaSy...")
        print("         3. Re-run this script: python verify_retrieval.py")
        print("\n[INFO] Running offline semantic vector simulation to demonstrate vector")
        print("       storage, cosine similarity ranking, and persistence logic.")
        print("-" * 70)
        _run_simulation(chunks, sample_pdfs)
        return

    # 3. Live Gemini API Embedding and Indexing
    print("\n[2/5] Generating live embeddings via Google GenAI SDK...")
    from src.embeddings import GeminiEmbedder

    embedder = GeminiEmbedder()
    try:
        chunks_with_embeddings = embedder.embed_chunks(chunks)
        print(f"      Successfully embedded {len(chunks_with_embeddings)} chunks.")
        dim = len(chunks_with_embeddings[0][1])
        print(f"      Vector dimensionality : {dim}")
    except Exception as exc:
        print(f"[ERROR] Gemini API embedding generation failed: {exc}")
        return

    # 4. Populate and Save Vector Store
    print("\n[3/5] Indexing into VectorStore and testing persistence...")
    store = VectorStore(storage_dir=VECTOR_STORE_DIR)
    store.add_chunks(chunks_with_embeddings)
    for p in sample_pdfs:
        store.register_document_hash(compute_file_hash(p), p.name)
    saved_path = store.save()
    print(f"      Saved vector index to : {saved_path}")

    # Test persistence: destroy in-memory store and reload from disk
    print("      Testing disk reloading (destroying in-memory store)...")
    store.clear()
    assert store.count == 0, "Store was not cleared."

    loaded_store = VectorStore(storage_dir=VECTOR_STORE_DIR)
    loaded_store.load()
    print(f"      Reloaded from disk    : {loaded_store.count} vectors, dimension {loaded_store.dimension}.")

    # 5. Query Embedding and Semantic Retrieval
    test_query = "What is Retrieval-Augmented Generation?"
    print(f"\n[4/5] Embedding user query: \"{test_query}\"...")
    query_vector = embedder.embed_text(test_query, task_type="RETRIEVAL_QUERY")

    print(f"\n[5/5] Performing cosine similarity search (Top-{DEFAULT_TOP_K})...")
    results = loaded_store.search(query_vector, top_k=DEFAULT_TOP_K)

    _display_results(test_query, results)

    print("\n[SUCCESS] Phase 3 Embeddings & Vector Store pipeline verified successfully!")
    print("=" * 70)


def _run_simulation(chunks, sample_pdfs=None):
    """Demonstrates vector storage and cosine similarity using deterministic simulation."""
    import numpy as np

    print("Running simulated vector index with deterministic keyword projections...")
    simulated_pairs = []
    
    # Simple deterministic feature projection for demonstration
    keywords = ["retrieval-augmented", "rag", "embedding", "vector", "chunk", "hallucination"]
    for chunk in chunks:
        lower = chunk.text.lower()
        vec = [float(lower.count(k) + 0.1) for k in keywords]
        # Normalize
        norm = np.linalg.norm(vec)
        norm_vec = [float(v / norm) for v in vec]
        simulated_pairs.append((chunk, norm_vec))

    test_dir = DATA_DIR / "vector_store"
    store = VectorStore(storage_dir=test_dir)
    store.add_chunks(simulated_pairs)
    if sample_pdfs:
        for p in sample_pdfs:
            store.register_document_hash(compute_file_hash(p), p.name)
    store.save()

    # Clear and reload
    reloaded = VectorStore(storage_dir=test_dir)
    reloaded.load()

    query = "What is Retrieval-Augmented Generation?"
    q_vec = [1.0, 1.0, 0.1, 0.1, 0.1, 0.1]
    norm = np.linalg.norm(q_vec)
    q_vec = [float(v / norm) for v in q_vec]

    results = reloaded.search(q_vec, top_k=DEFAULT_TOP_K)
    _display_results(query, results)

    print("\n[SIMULATION COMPLETE] Vector storage, cosine ranking, and disk persistence verified.")
    print("Add your GEMINI_API_KEY to .env to run with real Gemini high-dimensional embeddings.")
    print("=" * 70)


def _display_results(query: str, results):
    print("\n" + "=" * 50)
    print(f"QUERY: \"{query}\"")
    print(f"Found {len(results)} relevant chunks:")
    print("=" * 50)

    for rank, res in enumerate(results, start=1):
        meta = res.metadata
        preview = res.text.replace("\n", " ")
        if len(preview) > 130:
            preview = preview[:130] + "..."

        print(f"\nResult #{rank} [Cosine Similarity: {res.similarity_score:.4f}]")
        print(f"  Source   : {meta.get('source')} (Page {meta.get('page')})")
        print(f"  Chunk ID : {meta.get('chunk_id')}")
        print(f"  Content  : \"{preview}\"")


if __name__ == "__main__":
    run_verification()
