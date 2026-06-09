from pathlib import Path

from backend.ingest import chunk_text, load_text
from backend.rag import LegalRAG

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENT_PATH = PROJECT_ROOT / "documents" / "us_constitution.txt"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20


def main() -> None:
    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"Missing document: {DOCUMENT_PATH}. See README.md for setup."
        )

    print("Loading and embedding the U.S. Constitution...")
    text = load_text(DOCUMENT_PATH)
    chunks = chunk_text(
        text,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    rag = LegalRAG()
    rag.index(chunks)
    print(f"Indexed {len(chunks)} chunks.")

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        answer, sources = rag.answer(query)
        print(f"\nAnswer:\n{answer}")
        print("\nRetrieved chunks:")
        for source in sources:
            print(
                f"- Chunk {source.meta['chunk_number']} "
                f"(cosine score: {source.score:.3f})"
            )


if __name__ == "__main__":
    main()
