"""Text chunking module for Mini AI Knowledge Assistant.

Splits extracted document pages into manageable, overlapping text chunks
with boundary awareness (paragraphs, sentences, words) while preserving
source and page metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.loader import DocumentPage


@dataclass
class TextChunk:
    """Represents an indexed chunk of text with traceable metadata."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk object to dictionary format."""
        return {
            "text": self.text,
            "metadata": self.metadata,
        }


def split_text_with_overlap(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[Tuple[str, int, int]]:
    """Splits a single string of text into overlapping segments with boundary snapping.

    Prefers splitting on natural text boundaries (paragraph breaks, line breaks,
    sentence punctuation, and word spaces) to avoid cutting words in half.

    Args:
        text: The input text to divide.
        chunk_size: Maximum character length for each chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        A list of tuples: (chunk_text, start_char_index, end_char_index).
    """
    text = text.strip()
    if not text:
        return []

    text_len = len(text)
    if text_len <= chunk_size:
        return [(text, 0, text_len)]

    chunks: List[Tuple[str, int, int]] = []
    start = 0
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    while start < text_len:
        # If the remaining portion fits inside one chunk, finish up
        if text_len - start <= chunk_size:
            final_text = text[start:].strip()
            if final_text:
                chunks.append((final_text, start, text_len))
            break

        candidate_end = start + chunk_size
        window = text[start:candidate_end]

        # Search for a natural boundary in the latter portion of the window
        search_start_offset = max(chunk_overlap, int(chunk_size * 0.65))
        search_sub = window[search_start_offset:]

        boundary_pos = -1
        for sep in separators:
            rel_pos = search_sub.rfind(sep)
            if rel_pos != -1:
                # Found a boundary marker
                boundary_pos = (
                    start
                    + search_start_offset
                    + rel_pos
                    + (len(sep) if sep in [". ", "! ", "? "] else 0)
                )
                break

        # Fallback: search for any space after start + chunk_overlap
        if boundary_pos == -1:
            rel_space = window[chunk_overlap:].rfind(" ")
            if rel_space != -1:
                boundary_pos = start + chunk_overlap + rel_space
            else:
                boundary_pos = candidate_end

        chunk_str = text[start:boundary_pos].strip()
        if chunk_str:
            chunks.append((chunk_str, start, boundary_pos))

        # Calculate target start for next chunk based on overlap
        next_start = boundary_pos - chunk_overlap
        if next_start <= start:
            next_start = start + 1

        # Snap next_start forward to avoid beginning inside a sliced word
        if next_start < text_len:
            if next_start > 0 and not text[next_start - 1].isspace() and not text[next_start].isspace():
                next_space = text.find(" ", next_start, min(text_len, boundary_pos))
                if next_space != -1:
                    next_start = next_space + 1

        start = next_start

    return chunks


def chunk_document_pages(
    pages: List[DocumentPage],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[TextChunk]:
    """Chunks a collection of document pages into structured, overlapping chunks.

    Args:
        pages: List of DocumentPage objects from the document loader.
        chunk_size: Maximum chunk length in characters. Defaults to config value.
        chunk_overlap: Overlap between consecutive chunks in characters. Defaults to config value.

    Returns:
        List of TextChunk objects with preserved page and source metadata.

    Raises:
        ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size.
    """
    effective_chunk_size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
    effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP

    if effective_chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {effective_chunk_size}")

    if effective_chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {effective_chunk_overlap}")

    if effective_chunk_overlap >= effective_chunk_size:
        raise ValueError(
            f"chunk_overlap ({effective_chunk_overlap}) must be strictly less than "
            f"chunk_size ({effective_chunk_size})"
        )

    all_chunks: List[TextChunk] = []
    chunk_counter = 0

    for page in pages:
        page_text = page.text.strip()
        if not page_text:
            # Gracefully ignore empty pages without throwing an error
            continue

        page_meta = page.metadata
        split_results = split_text_with_overlap(
            page_text,
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap,
        )

        for chunk_text, start_char, end_char in split_results:
            chunk_metadata = {
                "source": page_meta.get("source", "unknown"),
                "source_path": page_meta.get("source_path", ""),
                "page": page_meta.get("page", 1),
                "total_pages": page_meta.get("total_pages", 1),
                "chunk_id": chunk_counter,
                "start_char": start_char,
                "end_char": end_char,
                "char_count": len(chunk_text),
            }

            all_chunks.append(TextChunk(text=chunk_text, metadata=chunk_metadata))
            chunk_counter += 1

    return all_chunks
