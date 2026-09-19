"""Unit tests for UI helpers and status validation (Phase 5).

Runs completely offline without requiring a live browser.
"""

import unittest
from src.ui_helpers import format_source_display, get_kb_status, validate_question
from src.vector_store import VectorStore


class TestUIHelpers(unittest.TestCase):
    def test_validate_question_empty(self):
        """Verify that empty, None, and whitespace strings fail validation with user warning."""
        valid, msg = validate_question("")
        self.assertFalse(valid)
        self.assertEqual(msg, "Please enter a question.")

        valid, msg = validate_question(None)
        self.assertFalse(valid)
        self.assertEqual(msg, "Please enter a question.")

        valid, msg = validate_question("   \n\t   ")
        self.assertFalse(valid)
        self.assertEqual(msg, "Please enter a question.")

    def test_validate_question_valid(self):
        """Verify that valid question text passes validation and is cleanly stripped."""
        valid, clean_q = validate_question("  What is RAG?  ")
        self.assertTrue(valid)
        self.assertEqual(clean_q, "What is RAG?")

    def test_get_kb_status_not_ready(self):
        """Verify status dictionary when knowledge base is not loaded."""
        status = get_kb_status(None, is_loaded=False)
        self.assertFalse(status["is_ready"])
        self.assertEqual(status["chunk_count"], 0)
        self.assertEqual(status["document_count"], 0)

    def test_get_kb_status_ready(self):
        """Verify status dictionary when vector store is populated."""
        store = VectorStore()
        store.add("Sample text 1", [1.0, 0.0], {"source": "doc1.pdf", "page": 1})
        store.add("Sample text 2", [0.0, 1.0], {"source": "doc2.pdf", "page": 1})
        store.add("Sample text 3", [0.5, 0.5], {"source": "doc1.pdf", "page": 2})

        status = get_kb_status(store, is_loaded=True)
        self.assertTrue(status["is_ready"])
        self.assertEqual(status["chunk_count"], 3)
        self.assertEqual(status["document_count"], 2)
        self.assertIn("doc1.pdf", status["documents"])
        self.assertIn("doc2.pdf", status["documents"])

    def test_format_source_display(self):
        """Verify formatted string for expander source title."""
        src = {
            "source": "manual.pdf",
            "page": 4,
            "similarity_score": 0.876543,
        }
        title = format_source_display(src)
        self.assertEqual(title, "📄 manual.pdf — Page 4 (Similarity: 0.8765)")


if __name__ == "__main__":
    unittest.main()
