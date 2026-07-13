import argparse
import re
from pathlib import Path

from backend.catalog import PROJECT_ROOT, manifest_errors, selected_index_assets
from backend.chunker import chunk_text
from backend.loaders import load_document
from backend.rag import LegalRAG

DOCUMENTS_PATH = PROJECT_ROOT / "documents"
RAW_DOCUMENTS_PATH = DOCUMENTS_PATH / "raw"
SUPPORTED_SUFFIXES = {".hwpx", ".pdf", ".txt"}


def _normalized_stem(path: Path) -> str:
    """Compatibility helper retained for callers; selection uses the manifest."""
    return re.sub(r"\([^)]*\)", "", path.stem).strip().casefold()


def get_document_paths() -> list[Path]:
    return [asset.path for asset in selected_index_assets()]


def document_id(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index the configured legal corpus.")
    parser.add_argument(
        "--types",
        help="Comma-separated document types to rebuild; all types by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_types = {
        value.strip() for value in (args.types or "").split(",") if value.strip()
    }
    failures = []
    catalog_errors = manifest_errors()
    if catalog_errors:
        raise RuntimeError("Invalid document manifest:\n- " + "\n- ".join(catalog_errors))
    assets = selected_index_assets()
    document_paths = [asset.path for asset in assets]
    metadata_by_path = {asset.path: asset.index_metadata() for asset in assets}
    indexed_count = 0
    skipped_count = 0

    with LegalRAG() as rag:
        for document_path in document_paths:
            document_type = metadata_by_path[document_path].get("document_type")
            if selected_types and document_type not in selected_types:
                continue
            source_path = document_id(document_path)
            source_mtime_ns = document_path.stat().st_mtime_ns
            if rag.document_is_current(source_path, source_mtime_ns):
                skipped_count += 1
                continue

            try:
                chunks = chunk_text(
                    load_document(document_path),
                    document_name=document_path.name,
                    document_type=document_type,
                )
                for chunk in chunks:
                    chunk.meta.update(metadata_by_path[document_path])
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
            indexed_count += 1

        if not failures:
            removed_documents = rag.prune_missing_documents(
                [document_id(document_path) for document_path in document_paths],
                cleanup_orphans=bool(indexed_count),
            )
            if not selected_types and (indexed_count or removed_documents):
                rag.relink_citations()

    if skipped_count:
        print(f"current documents skipped: {skipped_count}")

    if failures:
        print("index completed with failures:")
        for failed_path, error in failures:
            print(f"- {document_id(failed_path)}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
