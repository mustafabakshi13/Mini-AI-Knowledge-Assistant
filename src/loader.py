"""Document loader module for Mini AI Knowledge Assistant.

Extracts text from PDF documents using pypdf, cleans common extraction artifacts,
and preserves page-level metadata.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union
import re
import pypdf


@dataclass
class DocumentPage:
    """Represents a single page extracted from a document."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert page object to dictionary format."""
        return {
            "text": self.text,
            "metadata": self.metadata,
        }


def clean_text(raw_text: str) -> str:
    """Cleans extracted text without altering semantic meaning.

    - Normalizes line endings to \n
    - Replaces excessive horizontal spaces and tabs with a single space
    - Trims leading and trailing whitespace from each line
    - Collapses three or more consecutive line breaks into a double line break
    - Strips leading and trailing whitespace from the document
    """
    if not raw_text:
        return ""

    # Normalize line endings
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace horizontal whitespace (tabs, multiple spaces) with a single space
    text = re.sub(r"[^\S\n]+", " ", text)

    # Strip whitespace from each individual line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse excessive consecutive newlines to a standard paragraph break (\n\n)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_pdf(
    file_path: Union[str, Path],
    skip_empty_pages: bool = False
) -> List[DocumentPage]:
    """Loads a PDF file and extracts text page-by-page.

    Args:
        file_path: Path to the target PDF file.
        skip_empty_pages: If True, pages containing no meaningful text are omitted.
                          Defaults to False to preserve exact page accounting.

    Returns:
        List of DocumentPage objects with cleaned text and associated metadata.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file cannot be opened as a PDF.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF document not found at: {path}")

    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to read PDF file '{path.name}': {exc}") from exc

    total_pages = len(reader.pages)
    extracted_pages: List[DocumentPage] = []

    for index, page in enumerate(reader.pages):
        page_number = index + 1
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            # Handle potential extraction errors gracefully without terminating
            raw_text = ""

        cleaned = clean_text(raw_text)

        if skip_empty_pages and not cleaned:
            continue

        metadata = {
            "source": path.name,
            "source_path": str(path),
            "page": page_number,
            "total_pages": total_pages,
            "char_count": len(cleaned),
        }

        extracted_pages.append(DocumentPage(text=cleaned, metadata=metadata))

    return extracted_pages


def compute_file_hash(data_or_path: Union[str, Path, bytes]) -> str:
    """Computes the SHA-256 hash of a file or byte buffer for duplicate detection.

    Args:
        data_or_path: File path or raw bytes.

    Returns:
        Hexadecimal SHA-256 string.
    """
    import hashlib

    sha256 = hashlib.sha256()
    if isinstance(data_or_path, bytes):
        sha256.update(data_or_path)
    else:
        path = Path(data_or_path).resolve()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
    return sha256.hexdigest()


def load_multiple_pdfs(
    file_paths: List[Union[str, Path]],
    skip_empty_pages: bool = False,
) -> List[DocumentPage]:
    """Loads multiple PDF files and aggregates their extracted pages.

    Preserves originating document filename and page numbers for every page.

    Args:
        file_paths: List of paths to PDF documents.
        skip_empty_pages: If True, pages without text are omitted.

    Returns:
        Aggregated list of DocumentPage objects from all provided PDFs.
    """
    all_pages: List[DocumentPage] = []
    for fp in file_paths:
        pages = load_pdf(fp, skip_empty_pages=skip_empty_pages)
        all_pages.extend(pages)
    return all_pages

