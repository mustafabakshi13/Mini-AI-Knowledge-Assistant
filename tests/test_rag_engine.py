"""Offline unit tests for RAGEngine (Phase 4).

These tests verify context formatting, prompt construction, threshold filtering,
input validation, and response structures completely offline using mock components.
No external Gemini API calls are made.
"""

from unittest.mock import MagicMock
import unittest
from src.chunker import TextChunk
from src.rag_engine import (
    FALLBACK_NO_CONTEXT_MESSAGE,
    RAGEngine,
    RAGResponse,
    SYSTEM_INSTRUCTION,
)
from src.vector_store import SearchResult, VectorStore


class TestRAGEngineOffline(unittest.TestCase):
    def setUp(self):
        # Create a mock vector store with deterministic items
        self.mock_store = VectorStore()
        self.mock_store.add(
            text="RAG combines retrieval with language models for factual answers.",
            embedding=[1.0, 0.0, 0.0],
            metadata={"source": "test_guide.pdf", "page": 1, "chunk_id": 0},
        )
        self.mock_store.add(
            text="Vector databases index embeddings using cosine similarity.",
            embedding=[0.0, 1.0, 0.0],
            metadata={"source": "test_guide.pdf", "page": 2, "chunk_id": 1},
        )

        # Mock embedder that returns [1.0, 0.0, 0.0]
        self.mock_embedder = MagicMock()
        self.mock_embedder.embed_text.return_value = [1.0, 0.0, 0.0]

        # Mock LLM client
        self.mock_llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "According to test_guide.pdf (Page 1), RAG combines retrieval with language models."
        self.mock_llm_client.models.generate_content.return_value = mock_response

        # Instantiate RAGEngine with injected mocks
        self.engine = RAGEngine(
            vector_store=self.mock_store,
            embedder=self.mock_embedder,
            llm_client=self.mock_llm_client,
            score_threshold=0.2,
        )

    def test_empty_or_whitespace_question(self):
        """Test that empty or whitespace questions are rejected gracefully without calling LLM."""
        res_empty = self.engine.answer_question("")
        self.assertFalse(res_empty.has_sufficient_context)
        self.assertIn("Please provide a valid", res_empty.answer)
        self.mock_embedder.embed_text.assert_not_called()

        res_spaces = self.engine.answer_question("   \t  ")
        self.assertFalse(res_spaces.has_sufficient_context)
        self.mock_embedder.embed_text.assert_not_called()

    def test_format_context(self):
        """Test that search results are formatted into a clean context block with metadata."""
        results = [
            SearchResult(
                text="Artificial Intelligence overview.",
                metadata={"source": "ai.pdf", "page": 1, "chunk_id": 0},
                similarity_score=0.9543,
            )
        ]
        context = self.engine.format_context(results)
        self.assertIn("[Document: ai.pdf | Page: 1 | Chunk ID: 0 | Similarity: 0.9543]", context)
        self.assertIn("Artificial Intelligence overview.", context)

    def test_format_prompt(self):
        """Test that the prompt incorporates context and enforces grounding rules."""
        context = "[Document: doc.pdf | Page: 1 | Chunk ID: 0]\nSome context text."
        prompt = self.engine.format_prompt("What is X?", context)
        self.assertIn("--- Retrieved Document Context ---", prompt)
        self.assertIn("User Question: What is X?", prompt)
        self.assertIn("grounded, factual answer", prompt)

    def test_end_to_end_answering_with_mocks(self):
        """Test the end-to-end question answering flow with mocked LLM."""
        response = self.engine.answer_question("What is RAG?")

        self.assertIsInstance(response, RAGResponse)
        self.assertTrue(response.has_sufficient_context)
        self.assertIn("According to test_guide.pdf", response.answer)
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0]["source"], "test_guide.pdf")
        self.assertEqual(response.sources[0]["page"], 1)
        self.assertEqual(response.sources[0]["chunk_id"], 0)

        # Verify embedder was called with query
        self.mock_embedder.embed_text.assert_called_once_with(
            "What is RAG?", task_type="RETRIEVAL_QUERY"
        )

        # Verify LLM generate_content was called
        self.mock_llm_client.models.generate_content.assert_called_once()

    def test_low_similarity_threshold_filtering(self):
        """Test that queries returning only chunks below the score threshold produce fallback."""
        # Query that produces orthogonal similarity (score = 0.0, below threshold 0.2)
        self.mock_embedder.embed_text.return_value = [0.0, 0.0, 1.0]

        response = self.engine.answer_question("Tell me about astronomy.")
        self.assertFalse(response.has_sufficient_context)
        self.assertEqual(response.answer, FALLBACK_NO_CONTEXT_MESSAGE)
        self.assertEqual(response.sources, [])

        # Ensure LLM was NOT called to prevent hallucinations
        self.mock_llm_client.models.generate_content.assert_not_called()

    def test_empty_vector_store_fallback(self):
        """Test that an empty vector store immediately produces fallback without calling LLM."""
        empty_store = VectorStore()
        engine_empty = RAGEngine(
            vector_store=empty_store,
            embedder=self.mock_embedder,
            llm_client=self.mock_llm_client,
        )

        response = engine_empty.answer_question("Any question?")
        self.assertFalse(response.has_sufficient_context)
        self.assertEqual(response.answer, FALLBACK_NO_CONTEXT_MESSAGE)
        self.mock_llm_client.models.generate_content.assert_not_called()

    def test_llm_api_failure_raises_runtime_error(self):
        """Test that Gemini LLM API failure raises RuntimeError with descriptive message."""
        self.mock_llm_client.models.generate_content.side_effect = Exception("Model overloaded")
        with self.assertRaises(RuntimeError) as ctx:
            self.engine.answer_question("What is RAG?")
        self.assertIn("Gemini LLM generation failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

