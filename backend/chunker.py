from dataclasses import dataclass, field
import re

from backend.config import DEFAULT_MAX_WORDS
from backend.constants import STRUCTURE_KINDS, STRUCTURE_PATTERNS
from backend.document import Document
from backend.text import split_sentences

MAX_SHORT_BLOCK_WORDS = 50


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


def structure_metadata(block: str) -> tuple[str | None, str | None, str | None]:
    """Return the structure kind, number, and title when a block starts a legal unit."""
    for kind, pattern in STRUCTURE_PATTERNS:
        match = pattern.match(block)
        if match:
            return kind, match.group("number"), match.group("title") or None

    return None, None, None


@dataclass
class ChunkingState:
    chunks: list[Document] = field(default_factory=list)
    structure_counts: dict = field(default_factory=dict)
    structure_orders: dict = field(default_factory=dict)
    current_structures: dict = field(
        default_factory=lambda: {
            kind: {"number": None, "title": None, "order": None}
            for kind in STRUCTURE_KINDS
        }
    )


def update_chunking_structure(state: ChunkingState, block: str) -> None:
    kind, number, title = structure_metadata(block)
    if not kind:
        return

    kind_index = STRUCTURE_KINDS.index(kind)
    parent_key = tuple(
        state.current_structures[parent_kind]["number"]
        for parent_kind in STRUCTURE_KINDS[:kind_index]
    )
    count_key = (kind, parent_key)
    order_key = (kind, parent_key, number)
    if order_key not in state.structure_orders:
        state.structure_counts[count_key] = state.structure_counts.get(count_key, 0) + 1
        state.structure_orders[order_key] = state.structure_counts[count_key]

    state.current_structures[kind] = {
        "number": number,
        "title": title,
        "order": state.structure_orders[order_key],
    }
    for child_kind in STRUCTURE_KINDS[kind_index + 1 :]:
        state.current_structures[child_kind] = {
            "number": None,
            "title": None,
            "order": None,
        }


def add_chunk(
    state: ChunkingState,
    content: str,
    group_id: int,
    part: int,
    document_name: str,
) -> None:
    meta = {
        "chunk_number": len(state.chunks) + 1,
        "document_name": document_name,
        "group_id": group_id,
        "part": part,
    }
    for kind in STRUCTURE_KINDS:
        structure = state.current_structures[kind]
        meta[f"{kind}_number"] = structure["number"]
        meta[f"{kind}_title"] = structure["title"]
        meta[f"{kind}_order"] = structure["order"]

    state.chunks.append(Document(content=content, meta=meta))


def chunk_text(
    text: str,
    max_words: int = DEFAULT_MAX_WORDS,
    document_name: str = "Unknown document",
) -> list[Document]:
    blocks = [
        normalized_block
        for block in re.split(r"(?:\n\s*\n+|[ \t]{4,})", text)
        if (normalized_block := " ".join(block.split()))
    ]
    state = ChunkingState()
    block_index = 0

    while block_index < len(blocks):
        group_id = block_index
        block = blocks[block_index]
        current_kind = structure_metadata(block)[0]
        update_chunking_structure(state, block)
        block_words = block.split()

        if len(block_words) > max_words:
            parts = split_long_block(block, max_words)
            for part, content in enumerate(parts):
                add_chunk(state, content, group_id, part, document_name)
            block_index += 1
            continue

        if (
            not current_kind
            and len(block_words) <= MAX_SHORT_BLOCK_WORDS
            and block_index + 1 < len(blocks)
        ):
            next_block = blocks[block_index + 1]
            next_kind = structure_metadata(next_block)[0]
            if (
                not next_kind
                and not any(structure_metadata(part.strip())[0] for part in block.splitlines())
                and len(block_words) + len(next_block.split()) <= max_words
            ):
                block = f"{block}\n\n{next_block}"
                block_index += 1

        add_chunk(state, block, group_id, 0, document_name)
        block_index += 1

    return state.chunks
