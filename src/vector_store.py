"""Vector store module for Mini AI Knowledge Assistant.

Provides a lightweight, transparent vector storage and cosine similarity
search implementation using NumPy with local file persistence (.npz + .json).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
from src.chunker import TextChunk
from src.config import VECTOR_STORE_DIR


@dataclass
class SearchResult:
    """Represents a retrieved document chunk with its similarity score."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    similarity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary format."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "similarity_score": self.similarity_score,
        }


class VectorStore:
    """NumPy-based vector store providing cosine similarity search and persistence."""

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None) -> None:
        """Initializes the VectorStore.

        Args:
            storage_dir: Directory used to persist embeddings and metadata.
                         Defaults to VECTOR_STORE_DIR from config.
        """
        self.storage_dir = Path(storage_dir) if storage_dir is not None else VECTOR_STORE_DIR
        self.embeddings: Optional[np.ndarray] = None
        self.chunks_data: List[Dict[str, Any]] = []
        self.indexed_hashes: Dict[str, str] = {}  # file_hash -> filename

    def is_document_indexed(self, file_hash: str) -> bool:
        """Returns True if a document with the matching SHA-256 hash is already indexed."""
        return file_hash in self.indexed_hashes

    def register_document_hash(self, file_hash: str, filename: str) -> None:
        """Registers a document's SHA-256 hash to prevent duplicate indexing."""
        if file_hash:
            self.indexed_hashes[file_hash] = filename

    @property
    def count(self) -> int:
        """Returns the number of chunks currently stored in the vector store."""
        return len(self.chunks_data)

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimensionality, or 0 if empty."""
        if self.embeddings is not None and self.embeddings.ndim == 2:
            return self.embeddings.shape[1]
        return 0

    def add_chunks(
        self,
        chunks_with_embeddings: List[Tuple[TextChunk, List[float]]],
    ) -> int:
        """Adds a collection of TextChunks and their corresponding embedding vectors.

        Args:
            chunks_with_embeddings: List of (TextChunk, embedding_vector) tuples.

        Returns:
            The new total count of stored items.

        Raises:
            ValueError: If input is empty or vector dimensions are inconsistent.
        """
        if not chunks_with_embeddings:
            return self.count

        new_vectors = []
        new_chunks_data = []
        for chunk, emb in chunks_with_embeddings:
            if not emb:
                raise ValueError("Encountered empty embedding vector.")
            new_vectors.append(emb)
            new_chunks_data.append({
                "text": chunk.text,
                "metadata": chunk.metadata,
            })

        new_arr = np.array(new_vectors, dtype=np.float32)

        if self.embeddings is None:
            self.embeddings = new_arr
            self.chunks_data.extend(new_chunks_data)
        else:
            if new_arr.shape[1] != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch: store has {self.dimension}, "
                    f"incoming batch has {new_arr.shape[1]}."
                )
            self.embeddings = np.vstack([self.embeddings, new_arr])
            self.chunks_data.extend(new_chunks_data)

        return self.count

    def add(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Adds an individual text and embedding vector.

        Convenient for testing and programmatic ingestion.
        """
        chunk = TextChunk(text=text, metadata=metadata or {})
        return self.add_chunks([(chunk, embedding)])

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
    ) -> List[SearchResult]:
        """Performs cosine similarity search against stored embeddings.

        Cosine similarity formula:
            similarity(A, B) = dot(A, B) / (||A|| * ||B||)

        Args:
            query_embedding: The vector representation of the query.
            top_k: Number of most relevant results to return.

        Returns:
            List of SearchResult objects sorted by similarity score descending.

        Raises:
            ValueError: If top_k <= 0 or query embedding dimension does not match.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if self.count == 0 or self.embeddings is None:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        if query_vec.ndim != 1:
            query_vec = query_vec.flatten()

        if len(query_vec) != self.dimension:
            raise ValueError(
                f"Query embedding dimension ({len(query_vec)}) does not match "
                f"vector store dimension ({self.dimension}). "
                "The stored vectors were created with a different embedding model or simulation. "
                "Please rebuild the index using the 'Rebuild Index' button in the Streamlit sidebar "
                "or re-run indexing with the active embedding model."
            )

        query_norm = float(np.linalg.norm(query_vec))
        if query_norm == 0.0:
            # Query vector of all zeros has no direction
            return []

        matrix = self.embeddings
        matrix_norms = np.linalg.norm(matrix, axis=1)
        matrix_norms = np.where(matrix_norms == 0.0, 1e-10, matrix_norms)

        # Vectorized cosine similarity computation
        similarities = np.dot(matrix, query_vec) / (matrix_norms * query_norm)

        # Rank indices by descending similarity
        k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:k]

        results: List[SearchResult] = []
        for idx in top_indices:
            chunk_info = self.chunks_data[idx]
            score = float(similarities[idx])
            results.append(
                SearchResult(
                    text=chunk_info["text"],
                    metadata=chunk_info["metadata"],
                    similarity_score=round(score, 6),
                )
            )

        return results

    def save(self, directory: Optional[Union[str, Path]] = None) -> Path:
        """Persists the vector store to disk as embeddings.npz and chunks.json.

        Args:
            directory: Directory to save files into. Defaults to self.storage_dir.

        Returns:
            The Path where files were saved.

        Raises:
            ValueError: If the vector store is empty.
        """
        target_dir = Path(directory) if directory is not None else self.storage_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        if self.count == 0 or self.embeddings is None:
            raise ValueError("Cannot save an empty vector store.")

        # 1. Save embeddings as compressed NumPy array
        np.savez_compressed(target_dir / "embeddings.npz", embeddings=self.embeddings)

        # 2. Save chunk texts and metadata as JSON
        with open(target_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks_data, f, ensure_ascii=False, indent=2)

        # 3. Save indexed document hashes to prevent duplicate ingestion
        with open(target_dir / "hashes.json", "w", encoding="utf-8") as f:
            json.dump(self.indexed_hashes, f, ensure_ascii=False, indent=2)

        return target_dir

    def load(self, directory: Optional[Union[str, Path]] = None) -> bool:
        """Loads a previously persisted vector store from disk.

        Args:
            directory: Directory containing embeddings.npz, chunks.json, and hashes.json.

        Returns:
            True if loaded successfully.

        Raises:
            FileNotFoundError: If the stored files do not exist.
        """
        source_dir = Path(directory) if directory is not None else self.storage_dir
        emb_file = source_dir / "embeddings.npz"
        meta_file = source_dir / "chunks.json"
        hash_file = source_dir / "hashes.json"

        if not emb_file.is_file() or not meta_file.is_file():
            raise FileNotFoundError(
                f"Vector store files not found in {source_dir}. "
                f"Expected 'embeddings.npz' and 'chunks.json'."
            )

        loaded_npz = np.load(emb_file)
        self.embeddings = loaded_npz["embeddings"]

        with open(meta_file, "r", encoding="utf-8") as f:
            self.chunks_data = json.load(f)

        if hash_file.is_file():
            with open(hash_file, "r", encoding="utf-8") as f:
                self.indexed_hashes = json.load(f)
        else:
            self.indexed_hashes = {}

        if len(self.chunks_data) != len(self.embeddings):
            raise ValueError(
                f"Integrity error: {len(self.chunks_data)} metadata records "
                f"found for {len(self.embeddings)} vectors."
            )

        return True

    def clear(self) -> None:
        """Clears all stored vectors and metadata from memory."""
        self.embeddings = None
        self.chunks_data = []
        self.indexed_hashes = {}

