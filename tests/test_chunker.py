"""Unit tests for text chunking and metadata retention (Phase 2)."""

import unittest
from src.loader import DocumentPage
from src.chunker import chunk_document_pages, split_text_with_overlap, TextChunk


class TestTextChunker(unittest.TestCase):
    def setUp(self):
        self.long_sample_text = (
            "Retrieval-Augmented Generation, commonly known as RAG, is an artificial intelligence framework "
            "designed to improve the quality of large language model responses by grounding the model on "
            "external knowledge sources. While standard language models are trained on large corpuses of data, "
            "their knowledge is static and cut off at their training date. Furthermore, models can generate "
            "factually incorrect information, a phenomenon known as hallucination."
        )
        self.sample_page = DocumentPage(
            text=self.long_sample_text,
            metadata={"source": "ai_primer.pdf", "page": 2, "total_pages": 5},
        )

    def test_chunking_divides_long_text(self):
        """Test 4: Verify that long text is divided into multiple manageable chunks."""
        chunks = chunk_document_pages([self.sample_page], chunk_size=120, chunk_overlap=25)
        self.assertGreater(len(chunks), 1)

        # Ensure no chunk exceeds the maximum chunk size significantly
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 130)

    def test_chunk_overlap(self):
        """Test 5: Verify that consecutive chunks maintain the configured overlap."""
        chunks = chunk_document_pages([self.sample_page], chunk_size=150, chunk_overlap=40)
        self.assertGreaterEqual(len(chunks), 2)

        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            c1_text = chunks[i].text
            c2_text = chunks[i + 1].text

            # Suffix of first chunk should appear in the beginning of second chunk
            has_overlap = False
            for length in range(30, 5, -1):
                suffix = c1_text[-length:]
                if suffix in c2_text:
                    has_overlap = True
                    break
            self.assertTrue(has_overlap, f"No overlap detected between chunk {i} and {i+1}")

    def test_metadata_preservation(self):
        """Test 6: Verify source filename, page information, and chunk identifiers are retained."""
        chunks = chunk_document_pages([self.sample_page], chunk_size=150, chunk_overlap=30)
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.metadata["source"], "ai_primer.pdf")
            self.assertEqual(chunk.metadata["page"], 2)
            self.assertEqual(chunk.metadata["total_pages"], 5)
            self.assertEqual(chunk.metadata["chunk_id"], i)
            self.assertIn("char_count", chunk.metadata)
            self.assertEqual(chunk.metadata["char_count"], len(chunk.text))

    def test_configuration_overrides(self):
        """Test 7: Verify that changing configured chunk size/overlap affects the chunker."""
        chunks_large = chunk_document_pages([self.sample_page], chunk_size=300, chunk_overlap=30)
        chunks_small = chunk_document_pages([self.sample_page], chunk_size=100, chunk_overlap=20)

        self.assertLess(len(chunks_large), len(chunks_small))

    def test_empty_page_handling(self):
        """Verify that pages without extractable text produce zero chunks without error."""
        empty_page = DocumentPage(text="", metadata={"source": "empty.pdf", "page": 1})
        chunks = chunk_document_pages([empty_page])
        self.assertEqual(len(chunks), 0)

    def test_invalid_parameters(self):
        """Verify that invalid chunk_size or overlap parameters raise ValueError."""
        with self.assertRaises(ValueError):
            chunk_document_pages([self.sample_page], chunk_size=0)

        with self.assertRaises(ValueError):
            chunk_document_pages([self.sample_page], chunk_size=100, chunk_overlap=100)

        with self.assertRaises(ValueError):
            chunk_document_pages([self.sample_page], chunk_size=100, chunk_overlap=120)


if __name__ == "__main__":
    unittest.main()
