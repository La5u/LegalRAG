import re
import statistics
from pathlib import Path

import pdfplumber
from haystack import Document

DEFAULT_MAX_WORDS = 150
MAX_SHORT_BLOCK_WORDS = 50
LINE_POSITION_TOLERANCE = 3
PARAGRAPH_GAP_TOLERANCE = 2


def extract_page_text(page) -> str:
    words = page.extract_words(use_text_flow=True)
    lines = []

    for word in words:
        if not lines or abs(word["top"] - lines[-1][0]["top"]) > LINE_POSITION_TOLERANCE:
            lines.append([word])
        else:
            lines[-1].append(word)

    lines = [
        line
        for line in lines
        if not " ".join(word["text"] for word in line).strip().isdigit()
    ]
    gaps = [
        lines[index][0]["top"] - lines[index - 1][0]["bottom"]
        for index in range(1, len(lines))
    ]
    normal_gap = statistics.median(gaps) if gaps else 0
    text = []

    for index, line in enumerate(lines):
        if index:
            gap = line[0]["top"] - lines[index - 1][0]["bottom"]
            text.append("\n\n" if gap > normal_gap + PARAGRAPH_GAP_TOLERANCE else "\n")
        text.append(" ".join(word["text"] for word in line))

    return "".join(text)


def load_pdf(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        pages = [extract_page_text(page).strip() for page in pdf.pages]
    return "".join(f"\f{page}" for page in pages)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8").strip()
    if suffix == ".pdf":
        return load_pdf(path)
    raise ValueError("Only TXT and PDF documents are supported.")


def split_long_block(block: str, max_words: int) -> list[str]:
    parts = []
    current_words = []

    for sentence in re.split(r"(?<=[.!?])\s+", block):
        sentence_words = sentence.split()
        if len(current_words) + len(sentence_words) > max_words and current_words:
            parts.append(" ".join(current_words))
            current_words = []

        while len(sentence_words) > max_words:
            parts.append(" ".join(sentence_words[:max_words]))
            sentence_words = sentence_words[max_words:]

        current_words.extend(sentence_words)

    if current_words:
        parts.append(" ".join(current_words))

    return parts


def chunk_text(
    text: str,
    max_words: int = DEFAULT_MAX_WORDS,
    document_name: str = "Unknown document",
) -> list[Document]:
    page_texts = text.split("\f")
    has_pages = len(page_texts) > 1 and not page_texts[0]
    blocks = [
        (" ".join(block.split()), page_number if has_pages else None)
        for page_number, page_text in enumerate(page_texts)
        for block in re.split(r"(?:\n\s*\n+|[ \t]{4,})", page_text)
        if block.strip()
    ]
    chunks = []
    block_index = 0

    def add_chunk(
        content: str,
        group_id: int,
        part: int,
        page_number: int | None,
    ) -> None:
        chunks.append(
            Document(
                content=content,
                meta={
                    "chunk_number": len(chunks) + 1,
                    "document_name": document_name,
                    "group_id": group_id,
                    "part": part,
                    "page_number": page_number,
                },
            )
        )

    while block_index < len(blocks):
        block, page_number = blocks[block_index]
        block_words = block.split()

        if len(block_words) > max_words:
            parts = split_long_block(block, max_words)
            for part, content in enumerate(parts):
                add_chunk(content, block_index, part, page_number)
            block_index += 1
            continue

        if len(block_words) <= MAX_SHORT_BLOCK_WORDS and block_index + 1 < len(blocks):
            next_block, next_page = blocks[block_index + 1]
            if (
                page_number == next_page
                and len(block_words) + len(next_block.split()) <= max_words
            ):
                block = f"{block}\n\n{next_block}"
                block_index += 1

        add_chunk(block, block_index, 0, page_number)
        block_index += 1

    return chunks
