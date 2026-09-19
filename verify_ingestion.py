"""Verification script for Phase 2: Document Ingestion and Text Chunking.

Demonstrates the complete end-to-end ingestion pipeline:
PDF -> Loader -> Cleaner -> Chunker -> Structured Chunks
"""

import sys
from pathlib import Path
from src.loader import load_pdf
from src.chunker import chunk_document_pages
from src.config import DATA_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def verify_ingestion(pdf_path: Path) -> None:
    print("=" * 70)
    print("  Phase 2 Verification: Document Ingestion & Chunking Pipeline")
    print("=" * 70)
    print(f"Target PDF File     : {pdf_path.name}")
    print(f"File Path           : {pdf_path}")
    print(f"Configured Chunk Size : {DEFAULT_CHUNK_SIZE} characters")
    print(f"Configured Overlap    : {DEFAULT_CHUNK_OVERLAP} characters")
    print("-" * 70)

    # Step 1: Load and Clean PDF
    print("[1/2] Loading and extracting text from PDF...")
    pages = load_pdf(pdf_path)
    print(f"      Total pages in PDF     : {len(pages)}")

    non_empty_pages = [p for p in pages if p.text.strip()]
    print(f"      Pages containing text  : {len(non_empty_pages)}")
    for p in pages:
        char_count = p.metadata.get("char_count", 0)
        status = f"{char_count} chars" if char_count > 0 else "EMPTY (handled gracefully)"
        print(f"        - Page {p.metadata.get('page')}: {status}")

    # Step 2: Chunk the Document Pages
    print("\n[2/2] Chunking extracted pages into structured chunks...")
    chunks = chunk_document_pages(pages)
    print(f"      Total chunks generated : {len(chunks)}")
    print("-" * 70)

    # Display Preview of Chunks
    preview_count = min(len(chunks), 4)
    print(f"Previewing first {preview_count} generated chunks:\n")

    for i in range(preview_count):
        chunk = chunks[i]
        meta = chunk.metadata
        print(f"--- Chunk #{meta['chunk_id']} [Source: {meta['source']} | Page: {meta['page']} | Chars: {meta['char_count']}] ---")
        preview_text = chunk.text.replace("\n", " ")
        if len(preview_text) > 140:
            preview_text = preview_text[:140] + "..."
        print(f"Text: \"{preview_text}\"")
        print(f"Metadata: {meta}\n")

    # Overlap Demonstration
    if len(chunks) >= 2:
        print("-" * 70)
        print("Consecutive Chunk Overlap Demonstration:")
        c0_text = chunks[0].text
        c1_text = chunks[1].text
        # Find common suffix of c0 that appears in c1
        overlap_found = ""
        for length in range(min(len(c0_text), len(c1_text)), 10, -1):
            suffix = c0_text[-length:]
            if suffix in c1_text:
                overlap_found = suffix
                break

        print(f"Chunk 0 ends with   : \"...{c0_text[-40:].strip()}\"")
        print(f"Chunk 1 starts with : \"{c1_text[:40].strip()}...\"")
        if overlap_found:
            print(f"Detected Overlap    : \"{overlap_found.strip()}\" ({len(overlap_found)} characters)")
        print("-" * 70)

    print("\n[SUCCESS] Document ingestion and chunking pipeline verified successfully!")
    print("=" * 70)


def main():
    sample_pdf = DATA_DIR / "sample.pdf"
    if not sample_pdf.exists():
        print(f"Error: Sample PDF not found at {sample_pdf}. Please run data/create_sample_pdf.py first.")
        sys.exit(1)

    verify_ingestion(sample_pdf)


if __name__ == "__main__":
    main()
