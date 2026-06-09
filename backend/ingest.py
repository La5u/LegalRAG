from pathlib import Path

from haystack import Document


def load_text(path: Path) -> str:
    """Load a UTF-8 text document and normalize whitespace."""
    text = path.read_text(encoding="utf-8")
    return " ".join(text.split())


def chunk_text(
    text: str,
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[Document]:
    """Split text into overlapping word-based Haystack documents."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    words = text.split()
    step = chunk_size - overlap
    chunks = []

    for chunk_number, start in enumerate(range(0, len(words), step), start=1):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break

        chunks.append(
            Document(
                content=" ".join(chunk_words),
                meta={
                    "chunk_number": chunk_number,
                    "start_word": start,
                    "end_word": start + len(chunk_words),
                },
            )
        )

        if start + chunk_size >= len(words):
            break

    return chunks
