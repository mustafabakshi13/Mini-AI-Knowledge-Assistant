"""Unit tests for VectorStore cosine similarity, ranking, and persistence (Phase 3).

These tests run completely offline and do NOT require an external Gemini API key.
"""

import tempfile
import unittest
from pathlib import Path
from src.chunker import TextChunk
from src.vector_store import SearchResult, VectorStore


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = Path(self.temp_dir.name)
        self.store = VectorStore(storage_dir=self.storage_path)

        # Known deterministic 3D vectors
        # v1: aligned with X axis (cosine with [1,0,0] = 1.0)
        # v2: nearly aligned with X axis (cosine with [1,0,0] ~ 0.994)
        # v3: orthogonal on Y axis (cosine with [1,0,0] = 0.0)
        self.items = [
            ("RAG is an AI framework.", [1.0, 0.0, 0.0], {"source": "doc.pdf", "page": 1, "chunk_id": 0}),
            ("Machine learning models.", [0.9, 0.1, 0.0], {"source": "doc.pdf", "page": 1, "chunk_id": 1}),
            ("Baking sourdough bread.", [0.0, 1.0, 0.0], {"source": "recipe.pdf", "page": 3, "chunk_id": 2}),
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test vector store initializes in an empty state."""
        self.assertEqual(self.store.count, 0)
        self.assertEqual(self.store.dimension, 0)
        self.assertEqual(self.store.search([1.0, 0.0, 0.0]), [])

    def test_adding_vectors(self):
        """Test adding individual and batch items."""
        for text, vec, meta in self.items:
            self.store.add(text, vec, meta)

        self.assertEqual(self.store.count, 3)
        self.assertEqual(self.store.dimension, 3)

    def test_cosine_similarity_ranking(self):
        """Test that cosine similarity ranks items correctly by semantic relevance."""
        for text, vec, meta in self.items:
            self.store.add(text, vec, meta)

        query = [1.0, 0.0, 0.0]
        results = self.store.search(query, top_k=3)

        self.assertEqual(len(results), 3)

        # 1st rank: Exact match ([1, 0, 0])
        self.assertEqual(results[0].text, "RAG is an AI framework.")
        self.assertAlmostEqual(results[0].similarity_score, 1.0, places=4)

        # 2nd rank: Close match ([0.9, 0.1, 0])
        self.assertEqual(results[1].text, "Machine learning models.")
        self.assertGreater(results[1].similarity_score, 0.95)

        # 3rd rank: Orthogonal ([0, 1, 0])
        self.assertEqual(results[2].text, "Baking sourdough bread.")
        self.assertAlmostEqual(results[2].similarity_score, 0.0, places=4)

    def test_top_k_limiting(self):
        """Test that top_k parameter strictly limits the number of returned results."""
        for text, vec, meta in self.items:
            self.store.add(text, vec, meta)

        results_k1 = self.store.search([1.0, 0.0, 0.0], top_k=1)
        self.assertEqual(len(results_k1), 1)

        results_k2 = self.store.search([1.0, 0.0, 0.0], top_k=2)
        self.assertEqual(len(results_k2), 2)

    def test_metadata_preservation(self):
        """Test that source, page, and chunk_id metadata are preserved on search results."""
        for text, vec, meta in self.items:
            self.store.add(text, vec, meta)

        results = self.store.search([1.0, 0.0, 0.0], top_k=1)
        meta = results[0].metadata
        self.assertEqual(meta["source"], "doc.pdf")
        self.assertEqual(meta["page"], 1)
        self.assertEqual(meta["chunk_id"], 0)

    def test_persistence_save_and_load(self):
        """Test saving vector store to disk, destroying memory, and reloading."""
        for text, vec, meta in self.items:
            self.store.add(text, vec, meta)

        # Save to disk
        save_path = self.store.save()
        self.assertTrue((save_path / "embeddings.npz").is_file())
        self.assertTrue((save_path / "chunks.json").is_file())

        # Clear from memory
        self.store.clear()
        self.assertEqual(self.store.count, 0)

        # Re-load into a brand new VectorStore instance
        new_store = VectorStore(storage_dir=self.storage_path)
        new_store.load()

        self.assertEqual(new_store.count, 3)
        self.assertEqual(new_store.dimension, 3)

        # Search the reloaded store to verify data integrity
        results = new_store.search([1.0, 0.0, 0.0], top_k=1)
        self.assertEqual(results[0].text, "RAG is an AI framework.")
        self.assertAlmostEqual(results[0].similarity_score, 1.0, places=4)

    def test_dimension_mismatch_error(self):
        """Test that searching with an invalid query vector dimension raises ValueError."""
        self.store.add("Sample", [1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            self.store.search([1.0, 0.0])  # 2D query against 3D store

    def test_zero_vector_query(self):
        """Test that all-zeros query vector returns empty results without crashing."""
        self.store.add("Sample", [1.0, 0.0, 0.0])
        results = self.store.search([0.0, 0.0, 0.0])
        self.assertEqual(results, [])

    def test_save_empty_store_raises_error(self):
        """Test that attempting to save an empty vector store raises ValueError."""
        with self.assertRaises(ValueError):
            self.store.save()


if __name__ == "__main__":
    unittest.main()
