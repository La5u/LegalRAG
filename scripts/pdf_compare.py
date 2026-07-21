from pathlib import Path

from backend.chunker import chunk_text
from backend.loaders import (
    extract_pdf_with_pdfium,
    extract_pdfplumber_parts,
    load_hwpx,
    load_pdf,
    pdf_text_quality,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def matching_hwpx_path(pdf_path: Path) -> Path | None:
    candidate = pdf_path.with_suffix(".hwpx")
    if candidate.exists():
        return candidate
    return None


def compact_sample(text: str, limit: int = 500) -> str:
    return " ".join(text.split())[:limit]


def paragraph_count(text: str) -> int:
    return sum(1 for block in text.split("\n\n") if block.strip())


def chunk_count(text: str, document_name: str) -> int:
    return len(chunk_text(text, document_name=document_name))


def compare_pair(pdf_path: Path) -> None:
    pdf_path = pdf_path.resolve()
    hwpx_path = matching_hwpx_path(pdf_path)
    if hwpx_path is None:
        print(f"no matching HWPX for {pdf_path}")
        return

    try:
        hwpx_text = load_hwpx(hwpx_path)
    except Exception as error:
        print(f"failed to load HWPX {hwpx_path}: {error}")
        return
    pdfium_text = extract_pdf_with_pdfium(pdf_path)
    pdfplumber_text, pdfplumber_table_count = extract_pdfplumber_parts(pdf_path)
    loaded_pdf_text = load_pdf(pdf_path)

    print(f"pdf: {pdf_path.relative_to(PROJECT_ROOT)}")
    print(f"hwpx: {hwpx_path.relative_to(PROJECT_ROOT)}")
    print(f"hwpx quality: {pdf_text_quality(hwpx_text)}")
    print(f"pdfium: {pdf_text_quality(pdfium_text)}")
    print(f"pdfplumber: {pdf_text_quality(pdfplumber_text)}")
    print(f"pdfplumber tables: {pdfplumber_table_count}")
    print(f"selected: {'pdfplumber' if loaded_pdf_text == pdfplumber_text else 'pdfium'}")
    print(
        "paragraphs:",
        {
            "hwpx": paragraph_count(hwpx_text),
            "pdfium": paragraph_count(pdfium_text),
            "pdfplumber": paragraph_count(pdfplumber_text),
        },
    )
    print(
        "chunks:",
        {
            "hwpx": chunk_count(hwpx_text, hwpx_path.name),
            "pdfium": chunk_count(pdfium_text, pdf_path.name),
            "pdfplumber": chunk_count(pdfplumber_text, pdf_path.name),
        },
    )
    print()
    print("hwpx sample:")
    print(compact_sample(hwpx_text))
    print()
    print("pdfium sample:")
    print(compact_sample(pdfium_text))
    print()
    print("pdfplumber sample:")
    print(compact_sample(pdfplumber_text))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", type=Path)
    args = parser.parse_args()
    compare_pair(args.pdf_path)


if __name__ == "__main__":
    main()
