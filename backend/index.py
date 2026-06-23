from pathlib import Path

from backend.ingest import chunk_text, load_document
from backend.rag import LegalRAG

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = PROJECT_ROOT / "documents"
SUPPORTED_SUFFIXES = {".hwpx", ".txt"}


def get_document_paths() -> list[Path]:
    return sorted(
        path
        for path in DOCUMENTS_PATH.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def document_id(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def main() -> None:
    rag = LegalRAG()
    document_paths = get_document_paths()
    rag.prune_missing_documents(
        [document_id(document_path) for document_path in document_paths]
    )

    for document_path in document_paths:
        source_path = document_id(document_path)
        source_mtime_ns = document_path.stat().st_mtime_ns
        if rag.document_is_current(source_path, source_mtime_ns):
            print(f"skip {document_path.name}")
            continue

        chunks = chunk_text(
            load_document(document_path),
            document_name=document_path.name,
        )
        rag.index(
            chunks,
            source_path=source_path,
            source_mtime_ns=source_mtime_ns,
        )
        print(f"indexed {document_path.name}: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
