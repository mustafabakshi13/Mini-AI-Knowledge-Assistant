"""Automated unit tests for Phase 6 improvements and bonus features.

Verifies multiple-document ingestion, duplicate detection via SHA-256,
dynamic Top-K overrides, index clearing/rebuilding, and evaluation execution.
Runs completely offline without requiring a live Gemini API key.
"""

from pathlib import Path
from unittest.mock import MagicMock
import tempfile
import unittest
from src.loader import compute_file_hash, load_multiple_pdfs
from src.chunker import chunk_document_pages
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from src.config import DATA_DIR
from evaluate_retrieval import run_evaluation


class TestPhase6Features(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf1 = DATA_DIR / "sample.pdf"
        cls.pdf2 = DATA_DIR / "sample_ml_primer.pdf"
        if not cls.pdf1.is_file() or not cls.pdf2.is_file():
            from data.create_sample_pdf import main as create_samples
            create_samples()

    def test_multiple_document_loading_and_coexistence(self):
        """Test 1: Multiple PDFs can be loaded and their metadata coexists cleanly."""
        pages = load_multiple_pdfs([self.pdf1, self.pdf2])
        self.assertGreater(len(pages), 0)

        sources = set(p.metadata["source"] for p in pages)
        self.assertEqual(sources, {"sample.pdf", "sample_ml_primer.pdf"})

        chunks = chunk_document_pages(pages)
        chunk_sources = set(c.metadata["source"] for c in chunks)
        self.assertEqual(chunk_sources, {"sample.pdf", "sample_ml_primer.pdf"})

    def test_duplicate_detection_via_sha256(self):
        """Test 2: Verifies SHA-256 hash calculation and duplicate detection in VectorStore."""
        hash1 = compute_file_hash(self.pdf1)
        hash1_repeat = compute_file_hash(self.pdf1)
        hash2 = compute_file_hash(self.pdf2)

        self.assertEqual(hash1, hash1_repeat)
        self.assertNotEqual(hash1, hash2)

        store = VectorStore()
        self.assertFalse(store.is_document_indexed(hash1))

        store.register_document_hash(hash1, "sample.pdf")
        self.assertTrue(store.is_document_indexed(hash1))
        self.assertFalse(store.is_document_indexed(hash2))

    def test_dynamic_top_k_control_in_rag_engine(self):
        """Test 3: Verify that passing dynamic top_k directly alters retrieval limit."""
        store = VectorStore()
        store.add("Chunk 1", [1.0, 0.0, 0.0], {"source": "doc1.pdf", "page": 1, "chunk_id": 0})
        store.add("Chunk 2", [0.9, 0.1, 0.0], {"source": "doc1.pdf", "page": 2, "chunk_id": 1})
        store.add("Chunk 3", [0.8, 0.2, 0.0], {"source": "doc2.pdf", "page": 1, "chunk_id": 2})

        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [1.0, 0.0, 0.0]

        mock_llm = MagicMock()
        mock_llm.models.generate_content.return_value = MagicMock(text="Grounded answer.")

        engine = RAGEngine(
            vector_store=store,
            embedder=mock_embedder,
            llm_client=mock_llm,
            top_k=3,
            score_threshold=0.1,
        )

        # Query with default top_k (3)
        res_top3 = engine.answer_question("Test query")
        self.assertEqual(len(res_top3.sources), 3)

        # Query with dynamic top_k override (1)
        res_top1 = engine.answer_question("Test query", top_k=1)
        self.assertEqual(len(res_top1.sources), 1)

        # Query with dynamic top_k override (2)
        res_top2 = engine.answer_question("Test query", top_k=2)
        self.assertEqual(len(res_top2.sources), 2)

    def test_vector_store_rebuilding_and_clearing(self):
        """Test 4: Verify that vector store can be cleared and rebuilt cleanly."""
        store = VectorStore()
        store.add("Chunk 1", [1.0, 0.0], {"source": "doc.pdf"})
        store.register_document_hash("abc123hash", "doc.pdf")
        self.assertEqual(store.count, 1)
        self.assertTrue(store.is_document_indexed("abc123hash"))

        store.clear()
        self.assertEqual(store.count, 0)
        self.assertFalse(store.is_document_indexed("abc123hash"))

    def test_retrieval_evaluation_benchmark_executes(self):
        """Test 5: Verify that the retrieval evaluation script runs and returns valid metrics."""
        result = run_evaluation(top_k=4)
        self.assertIn("total_questions", result)
        self.assertIn("source_accuracy_pct", result)
        self.assertIn("page_accuracy_pct", result)
        self.assertGreaterEqual(result["source_accuracy_pct"], 80.0)

    def test_upload_indexing_and_duplicate_rejection(self):
        """Test 6: Verify document upload indexing and duplicate rejection via index_pdf_file."""
        from app import index_pdf_file
        from src.config import UPLOADS_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VectorStore(storage_dir=temp_dir)
            with open(self.pdf1, "rb") as f:
                pdf_bytes = f.read()

            try:
                # First indexing attempt
                success1, msg1 = index_pdf_file(pdf_bytes, "uploaded_test_doc.pdf", store, has_active_key=False)
                self.assertTrue(success1)
                self.assertIn("Successfully indexed", msg1)
                self.assertGreater(store.count, 0)

                # Second indexing attempt with exact same content (duplicate)
                success2, msg2 = index_pdf_file(pdf_bytes, "uploaded_test_doc_copy.pdf", store, has_active_key=False)
                self.assertFalse(success2)
                self.assertIn("duplicate detected", msg2)
            finally:
                (UPLOADS_DIR / "uploaded_test_doc.pdf").unlink(missing_ok=True)
                (UPLOADS_DIR / "uploaded_test_doc_copy.pdf").unlink(missing_ok=True)

    def test_append_upload_to_existing_knowledge_base(self):
        """Test 7: Regression test for appending an uploaded PDF to an existing multi-document knowledge base.

        Verifies:
        - Initial store has existing indexed documents (sample.pdf, sample_ml_primer.pdf).
        - Uploading a new PDF appends new chunks without replacing or clearing existing ones.
        - Old chunks and metadata are strictly preserved.
        - The uploaded document chunks become retrievable via vector search.
        - Re-uploading the same content is rejected by duplicate detection.
        - Persisted vector store on disk reloads with the combined chunk count.
        """
        from app import index_pdf_file
        from src.config import UPLOADS_DIR
        from data.create_sample_pdf import generate_pdf

        with tempfile.TemporaryDirectory() as temp_dir:
            store = VectorStore(storage_dir=temp_dir)
            # Step 1: Pre-populate store with sample.pdf and sample_ml_primer.pdf
            for pdf_path in [self.pdf1, self.pdf2]:
                with open(pdf_path, "rb") as f:
                    ok, _ = index_pdf_file(f.read(), pdf_path.name, store, has_active_key=False)
                    self.assertTrue(ok)

            initial_count = store.count
            self.assertGreater(initial_count, 0)
            initial_sources = set(c["metadata"]["source"] for c in store.chunks_data)
            self.assertEqual(initial_sources, {"sample.pdf", "sample_ml_primer.pdf"})

            # Step 2: Create a distinct new PDF
            new_pdf_name = "quantum_computing_intro.pdf"
            new_pdf_path = Path(temp_dir) / new_pdf_name
            distinct_text = (
                "Quantum computing leverages superposition and entanglement of qubits "
                "to perform complex calculations exponentially faster than classical computers."
            )
            generate_pdf([distinct_text], new_pdf_path)
            with open(new_pdf_path, "rb") as f:
                new_pdf_bytes = f.read()

            try:
                # Step 3: Index the uploaded PDF into the existing store
                success, msg = index_pdf_file(
                    new_pdf_bytes, new_pdf_name, store, has_active_key=False
                )
                self.assertTrue(success)
                self.assertIn("Successfully indexed", msg)

                # Step 4: Verify count increased (appended, NOT replaced)
                self.assertGreater(store.count, initial_count)
                new_count = store.count

                # Step 5: Verify all original chunks remain intact
                current_sources = set(c["metadata"]["source"] for c in store.chunks_data)
                self.assertTrue(initial_sources.issubset(current_sources))
                self.assertIn(new_pdf_name, current_sources)
                self.assertEqual(len(current_sources), 3)

                # Step 6: Verify duplicate upload is rejected without mutating count
                dup_success, dup_msg = index_pdf_file(
                    new_pdf_bytes, "copy_of_quantum.pdf", store, has_active_key=False
                )
                self.assertFalse(dup_success)
                self.assertIn("duplicate detected", dup_msg)
                self.assertEqual(store.count, new_count)

                # Step 7: Verify search retrievability
                query_vec = [0.1] * store.dimension
                results = store.search(query_vec, top_k=store.count)
                retrieved_sources = set(r.metadata.get("source") for r in results)
                self.assertIn(new_pdf_name, retrieved_sources)

                # Step 8: Verify persistence across load
                reloaded_store = VectorStore(storage_dir=temp_dir)
                reloaded_store.load()
                self.assertEqual(reloaded_store.count, new_count)
                self.assertTrue(reloaded_store.is_document_indexed(compute_file_hash(new_pdf_bytes)))
            finally:
                (UPLOADS_DIR / new_pdf_name).unlink(missing_ok=True)
                (UPLOADS_DIR / "copy_of_quantum.pdf").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()


