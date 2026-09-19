"""Unit tests for GeminiEmbedder input validation and configuration (Phase 3).

These tests run completely offline and verify parameter handling without calling
external Gemini endpoints.
"""

import unittest
from src.embeddings import GeminiEmbedder


class TestGeminiEmbedderOffline(unittest.TestCase):
    def test_missing_or_placeholder_key_raises_error(self):
        """Verify that a missing or placeholder API key raises a clear ValueError."""
        with self.assertRaises(ValueError) as ctx:
            GeminiEmbedder(api_key="")
        self.assertIn("Gemini API key is not configured", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            GeminiEmbedder(api_key="your_gemini_api_key_here")
        self.assertIn("Gemini API key is not configured", str(ctx.exception))

    def test_empty_text_embedding_raises_error(self):
        """Verify that attempting to embed empty or whitespace-only text raises ValueError."""
        # Initialize with a dummy key format (does not call API during init)
        embedder = GeminiEmbedder(api_key="AIzaSyDummyKeyForOfflineValidation12345")

        with self.assertRaises(ValueError) as ctx:
            embedder.embed_text("")
        self.assertIn("Cannot generate embedding for empty", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            embedder.embed_text("   \n\t  ")
        self.assertIn("Cannot generate embedding for empty", str(ctx.exception))

    def test_empty_chunks_list_returns_empty(self):
        """Verify that passing an empty list to embed_chunks returns an empty list immediately."""
        embedder = GeminiEmbedder(api_key="AIzaSyDummyKeyForOfflineValidation12345")
        res = embedder.embed_chunks([])
        self.assertEqual(res, [])

    def test_api_failure_raises_runtime_error(self):
        """Verify that API failure during embed_text raises RuntimeError."""
        from unittest.mock import MagicMock
        embedder = GeminiEmbedder(api_key="AIzaSyDummyKeyForOfflineValidation12345")
        embedder.client = MagicMock()
        embedder.client.models.embed_content.side_effect = Exception("Service unavailable")

        with self.assertRaises(RuntimeError) as ctx:
            embedder.embed_text("Valid text query")
        self.assertIn("Gemini embedding API error", str(ctx.exception))

    def test_batch_api_failure_raises_runtime_error(self):
        """Verify that API failure during embed_chunks raises RuntimeError."""
        from unittest.mock import MagicMock
        from src.chunker import TextChunk
        embedder = GeminiEmbedder(api_key="AIzaSyDummyKeyForOfflineValidation12345")
        embedder.client = MagicMock()
        embedder.client.models.embed_content.side_effect = Exception("Rate limit exceeded")

        chunks = [TextChunk(text="Chunk 1", metadata={})]
        with self.assertRaises(RuntimeError) as ctx:
            embedder.embed_chunks(chunks)
        self.assertIn("Failed to embed chunk batch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

