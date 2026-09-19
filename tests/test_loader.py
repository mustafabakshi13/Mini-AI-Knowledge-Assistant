"""Unit tests for document loading and text cleaning (Phase 2)."""

import unittest
from pathlib import Path
from src.loader import DocumentPage, clean_text, load_pdf
from src.config import DATA_DIR


class TestDocumentLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_pdf = DATA_DIR / "sample.pdf"
        if not cls.sample_pdf.exists():
            raise FileNotFoundError(f"Test sample PDF missing at {cls.sample_pdf}")

    def test_pdf_loading(self):
        """Test 1: Verify that a valid PDF can be loaded and its text extracted."""
        pages = load_pdf(self.sample_pdf)
        self.assertIsInstance(pages, list)
        self.assertGreater(len(pages), 0)
        self.assertIn("RAG", pages[0].text)

    def test_multipage_handling(self):
        """Test 2: Verify that text from multiple pages is processed and page metadata is preserved."""
        pages = load_pdf(self.sample_pdf)
        self.assertEqual(len(pages), 4)

        # Verify page numbers are 1-indexed and incrementing
        page_numbers = [p.metadata["page"] for p in pages]
        self.assertEqual(page_numbers, [1, 2, 3, 4])

        # Verify total_pages matches
        for p in pages:
            self.assertEqual(p.metadata["total_pages"], 4)
            self.assertEqual(p.metadata["source"], "sample.pdf")

    def test_empty_page_handling(self):
        """Test 3: Verify that an empty/non-text page does not crash the pipeline."""
        pages = load_pdf(self.sample_pdf, skip_empty_pages=False)
        page_4 = pages[3]
        self.assertEqual(page_4.text, "")
        self.assertEqual(page_4.metadata["char_count"], 0)

        # Also test with skip_empty_pages=True
        non_empty = load_pdf(self.sample_pdf, skip_empty_pages=True)
        self.assertEqual(len(non_empty), 3)

    def test_clean_text_utility(self):
        """Verify that clean_text removes excessive whitespace without mangling words."""
        raw = "  Hello   world!  \r\n\r\n\nThis is    a test.   \n\n\n\nEnd.  "
        cleaned = clean_text(raw)
        expected = "Hello world!\n\nThis is a test.\n\nEnd."
        self.assertEqual(cleaned, expected)

    def test_nonexistent_file_raises_error(self):
        """Verify that attempting to load a missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_pdf("nonexistent_document_12345.pdf")

    def test_corrupt_pdf_raises_value_error(self):
        """Verify that attempting to load a non-PDF/corrupt file raises ValueError."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"NOT A REAL PDF FILE CONTENT AT ALL")
            temp_path = f.name
        try:
            with self.assertRaises(ValueError):
                load_pdf(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_empty_file_raises_value_error(self):
        """Verify that attempting to load an empty 0-byte file raises ValueError."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = f.name
        try:
            with self.assertRaises(ValueError):
                load_pdf(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

