"""Embeddings module for Mini AI Knowledge Assistant.

Wraps the official Google GenAI SDK (google-genai) to convert text and TextChunks
into high-dimensional vector representations using Gemini embedding models.
"""

from typing import List, Optional, Tuple
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL
from src.chunker import TextChunk


class GeminiEmbedder:
    """Handles text and batch chunk embedding using the official Google GenAI SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initializes the GeminiEmbedder.

        Args:
            api_key: Optional Gemini API key. Defaults to GEMINI_API_KEY from config.
            model: Optional embedding model name. Defaults to GEMINI_EMBEDDING_MODEL from config.

        Raises:
            ValueError: If the API key is missing or set to the default placeholder.
        """
        raw_key = api_key if api_key is not None else GEMINI_API_KEY
        if not raw_key or raw_key == "your_gemini_api_key_here":
            raise ValueError(
                "Gemini API key is not configured. Please set a valid GEMINI_API_KEY "
                "in your .env file or pass it to GeminiEmbedder(api_key=...)."
            )

        self.api_key = raw_key
        self.model = model if model is not None else GEMINI_EMBEDDING_MODEL
        self.client = genai.Client(api_key=self.api_key)

    def embed_text(
        self,
        text: str,
        task_type: Optional[str] = None,
    ) -> List[float]:
        """Generates an embedding vector for a single string of text.

        Args:
            text: The text to embed.
            task_type: Optional task type hint (e.g. 'RETRIEVAL_QUERY').

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            ValueError: If text is empty or whitespace only.
            RuntimeError: If the Gemini API request fails.
        """
        clean = text.strip()
        if not clean:
            raise ValueError("Cannot generate embedding for empty or whitespace-only text.")

        config = types.EmbedContentConfig(task_type=task_type) if task_type else None

        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=clean,
                config=config,
            )
            if not response.embeddings or not response.embeddings[0].values:
                raise RuntimeError("Gemini API returned an empty embedding response.")
            return response.embeddings[0].values
        except Exception as exc:
            raise RuntimeError(f"Gemini embedding API error: {exc}") from exc

    def embed_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 32,
        task_type: Optional[str] = "RETRIEVAL_DOCUMENT",
    ) -> List[Tuple[TextChunk, List[float]]]:
        """Generates embedding vectors for a list of TextChunks in batches.

        Preserves the association between each original TextChunk and its vector.

        Args:
            chunks: List of TextChunk objects to embed.
            batch_size: Maximum number of chunks to send in a single API call.
            task_type: Optional task type hint for document chunks.

        Returns:
            A list of tuples: (original_TextChunk, embedding_vector).

        Raises:
            ValueError: If chunks list is empty.
            RuntimeError: If an API call fails.
        """
        if not chunks:
            return []

        results: List[Tuple[TextChunk, List[float]]] = []
        config = types.EmbedContentConfig(task_type=task_type) if task_type else None

        for start_idx in range(0, len(chunks), batch_size):
            batch = chunks[start_idx : start_idx + batch_size]
            contents = [
                types.Content(parts=[types.Part.from_text(text=chunk.text)])
                for chunk in batch
            ]

            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                if not response.embeddings or len(response.embeddings) != len(batch):
                    raise RuntimeError(
                        f"Expected {len(batch)} embeddings from API batch, "
                        f"but received {len(response.embeddings) if response.embeddings else 0}."
                    )

                for chunk, emb in zip(batch, response.embeddings):
                    results.append((chunk, emb.values))
            except Exception as exc:
                raise RuntimeError(f"Failed to embed chunk batch starting at index {start_idx}: {exc}") from exc

        return results
