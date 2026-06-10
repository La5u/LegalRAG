import re
from pathlib import Path

from haystack import Document


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def split_long_block(block: str, max_words: int) -> list[str]:
    parts = []
    current_words = []

    for sentence in split_sentences(block):
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
    max_words: int = 150,
    document_name: str = "Unknown document",
) -> list[Document]:
    blocks = [
        " ".join(block.split())
        for block in re.split(r"(?:\n\s*\n+|[ \t]{4,})", text)
        if block.strip()
    ]
    chunks = []
    block_index = 0

    def add_chunk(content: str, group_id: int, part: int) -> None:
        chunks.append(
            Document(
                content=content,
                meta={
                    "chunk_number": len(chunks) + 1,
                    "document_name": document_name,
                    "group_id": group_id,
                    "part": part,
                },
            )
        )

    while block_index < len(blocks):
        block = blocks[block_index]
        block_words = block.split()

        if len(block_words) > max_words:
            parts = split_long_block(block, max_words)
            for part, content in enumerate(parts):
                add_chunk(content, block_index, part)
            block_index += 1
            continue

        if len(block_words) <= 20 and block_index + 1 < len(blocks):
            next_block = blocks[block_index + 1]
            if len(block_words) + len(next_block.split()) <= max_words:
                block = f"{block}\n\n{next_block}"
                block_index += 1

        add_chunk(block, block_index, 0)
        block_index += 1

    return chunks
