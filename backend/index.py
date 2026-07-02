import re
from pathlib import Path

from backend.ingest import chunk_text, load_document
from backend.rag import LegalRAG

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = PROJECT_ROOT / "documents"
RAW_DOCUMENTS_PATH = DOCUMENTS_PATH / "raw"
SUPPORTED_SUFFIXES = {".hwpx", ".pdf", ".txt"}


def _normalized_stem(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"\([^)]*\)", "", stem).strip()
    return stem.casefold()


def get_document_paths() -> list[Path]:
    paths = [
        path
        for path in RAW_DOCUMENTS_PATH.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    hwpx_stems = {
        _normalized_stem(path)
        for path in paths
        if path.suffix.lower() == ".hwpx"
    }
    return sorted(
        path
        for path in paths
        if path.suffix.lower() != ".pdf" or _normalized_stem(path) not in hwpx_stems
    )


def document_id(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def main() -> None:
    failures = []
    document_paths = get_document_paths()

    with LegalRAG() as rag:
        rag.prune_missing_documents(
            [document_id(document_path) for document_path in document_paths]
        )

        for document_path in document_paths:
            source_path = document_id(document_path)
            source_mtime_ns = document_path.stat().st_mtime_ns
            if rag.document_is_current(source_path, source_mtime_ns):
                print(f"skip {document_path.name}")
                continue

            try:
                chunks = chunk_text(
                    load_document(document_path),
                    document_name=document_path.name,
                )
                rag.index(
                    chunks,
                    source_path=source_path,
                    source_mtime_ns=source_mtime_ns,
                )
            except Exception as error:
                print(f"failed {document_path.name}: {error}")
                failures.append((document_path, error))
                continue

            print(f"indexed {document_path.name}: {len(chunks)} chunks")

    if failures:
        print("index completed with failures:")
        for failed_path, error in failures:
            print(f"- {document_id(failed_path)}: {error}")


if __name__ == "__main__":
    main()
