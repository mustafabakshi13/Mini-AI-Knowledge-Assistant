"""Utility to generate a deterministic sample PDF for testing the RAG pipeline."""

from pathlib import Path


def generate_pdf(pages_text: list[str], output_path: Path) -> None:
    """Generates a valid standard PDF 1.4 file with text across multiple pages.
    
    Includes exact cross-reference tables and standard Helvetica font definitions.
    """
    objects = []

    # Object 1: Catalog
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")

    # Object 2: Pages container
    kids_refs = " ".join(f"{3 + i * 2} 0 R" for i in range(len(pages_text)))
    objects.append(f"<< /Type /Pages /Kids [{kids_refs}] /Count {len(pages_text)} >>".encode("ascii"))

    font_obj_idx = 3 + len(pages_text) * 2

    # Page objects and Content stream objects
    for i, text in enumerate(pages_text):
        content_idx = 4 + i * 2
        # Page dictionary
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj_idx} 0 R >> >> "
            f"/Contents {content_idx} 0 R >>".encode("ascii")
        )

        # Content stream (layout text line by line)
        lines = text.strip().split("\n")
        stream_cmds = ["BT", "/F1 11 Tf", "50 720 Td", "15 TL"]
        for line in lines:
            safe_line = (
                line.replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
            )
            if safe_line.strip():
                stream_cmds.append(f"({safe_line}) Tj")
            stream_cmds.append("T*")
        stream_cmds.append("ET")

        stream_bytes = "\n".join(stream_cmds).encode("utf-8")
        objects.append(
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream"
        )

    # Font object
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    # Assemble complete PDF bytearray
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj_data in enumerate(objects):
        obj_num = i + 1
        offsets.append(len(output))
        output.extend(f"{obj_num} 0 obj\n".encode("ascii"))
        output.extend(obj_data)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        output.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    output.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(output)


def main():
    pages = [
        # Page 1
        (
            "Mini AI Knowledge Assistant - Sample Document\n"
            "\n"
            "Section 1: What is Retrieval-Augmented Generation (RAG)?\n"
            "Retrieval-Augmented Generation, commonly known as RAG, is an artificial intelligence framework\n"
            "designed to improve the quality of large language model responses by grounding the model on\n"
            "external knowledge sources. While standard language models are trained on large corpuses of data,\n"
            "their knowledge is static and cut off at their training date. Furthermore, models can generate\n"
            "factually incorrect information, a phenomenon known as hallucination.\n"
            "\n"
            "RAG addresses these challenges by retrieving authoritative facts from an external knowledge base\n"
            "before generating a response. This allows the model to reference timely and private data,\n"
            "substantially reducing the likelihood of hallucinations."
        ),
        # Page 2
        (
            "Section 2: The Core RAG Pipeline\n"
            "The architecture of a typical RAG system consists of three fundamental stages:\n"
            "1. Ingestion: Documents such as PDFs or text files are parsed, cleaned, and split into smaller\n"
            "manageable segments called text chunks.\n"
            "2. Indexing: Each chunk is passed through an embedding model to produce a high-dimensional vector\n"
            "representation. These vectors are stored in a vector database such as FAISS or Chroma.\n"
            "3. Retrieval and Generation: When a user poses a question, the query is converted into an embedding.\n"
            "The vector database performs a cosine similarity search to find the most relevant chunks.\n"
            "Finally, the retrieved chunks and the user query are assembled into a prompt and submitted to an LLM."
        ),
        # Page 3
        (
            "Section 3: Chunking Strategies and Grounded Answering\n"
            "Chunking is the process of breaking long documents into cohesive, smaller units. Two primary\n"
            "parameters govern chunking: chunk size and chunk overlap. Chunk size determines the maximum length\n"
            "of each fragment, while chunk overlap ensures that context spanning chunk boundaries is not lost.\n"
            "\n"
            "To guarantee grounded answers, the system prompt must explicitly instruct the LLM to answer only\n"
            "using the retrieved facts. If the information is missing from the context, the assistant must\n"
            "state that it cannot find the answer rather than speculating."
        ),
        # Page 4 (Empty page for edge case testing)
        ""
    ]

    output_file = Path(__file__).resolve().parent / "sample.pdf"
    generate_pdf(pages, output_file)
    print(f"Generated sample PDF at: {output_file} ({len(pages)} pages)")

    # Document 2: Machine Learning Primer for multi-document testing
    ml_pages = [
        # Page 1
        (
            "Machine Learning Primer - Document 2\n"
            "\n"
            "Section 1: Supervised vs Unsupervised Learning\n"
            "Machine learning is categorized into three primary paradigms:\n"
            "1. Supervised Learning: Algorithms are trained on labeled data where inputs are paired with target outputs.\n"
            "Common supervised tasks include classification and linear regression.\n"
            "2. Unsupervised Learning: Algorithms discover hidden patterns, groupings, or clusters in unlabeled data.\n"
            "Prominent unsupervised techniques include k-means clustering and principal component analysis (PCA).\n"
            "3. Reinforcement Learning: Agents learn through trial-and-error by interacting with an environment to maximize rewards."
        ),
        # Page 2
        (
            "Section 2: Gradient Descent and Optimization\n"
            "Optimization is the engine of modern neural network training. Gradient descent is an iterative\n"
            "first-order optimization algorithm used to find a local minimum of a differentiable objective function.\n"
            "The algorithm calculates the gradient of the loss function with respect to model weights and updates the\n"
            "weights in the opposite direction of the gradient scaled by a hyperparameter called the learning rate.\n"
            "\n"
            "Variants such as Stochastic Gradient Descent (SGD) and Adam are widely used in practice."
        )
    ]
    ml_output_file = Path(__file__).resolve().parent / "sample_ml_primer.pdf"
    generate_pdf(ml_pages, ml_output_file)
    print(f"Generated ML Primer PDF at: {ml_output_file} ({len(ml_pages)} pages)")


if __name__ == "__main__":
    main()

