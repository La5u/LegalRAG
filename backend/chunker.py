from dataclasses import dataclass, field
from collections import Counter
import re

from backend.config import DEFAULT_MAX_WORDS
from backend.constants import STRUCTURE_KINDS, STRUCTURE_PATTERNS
from backend.document import Document
from backend.text import split_sentences

MAX_SHORT_BLOCK_WORDS = 50
PAGE_MARKER_PATTERN = re.compile(r"^\[\[PAGE:(\d+)\]\]$")
AMENDMENT_NOTE_PATTERN = re.compile(
    r"^\[(?:전문개정|본조신설|제목개정|시행일|본조삭제|종전의).*\]$"
)
FOOTER_BLOCK_PATTERN = re.compile(
    r"법제처.*국가법령정보센터|국가법령정보센터.*법제처|"
    r"^-?\s*/\s*-?\s*국가법령정보센터$"
)
DELETED_PROVISION_PATTERN = re.compile(r"^(?:\d+\.\s*)?삭제\s*<[^>]+>$")


def is_boilerplate_block(block: str) -> bool:
    """Identify standalone publication metadata that should not be searched."""
    compact = " ".join(block.split())
    return bool(
        AMENDMENT_NOTE_PATTERN.fullmatch(compact)
        or FOOTER_BLOCK_PATTERN.search(compact)
        or DELETED_PROVISION_PATTERN.fullmatch(compact)
        or compact == "다."
    )


def searchable_blocks(text: str) -> list[str]:
    blocks = [
        normalized
        for raw in re.split(r"(?:\n\s*\n+|[ \t]{4,})", text)
        if (normalized := " ".join(raw.split()))
        and not is_boilerplate_block(normalized)
    ]
    counts = Counter(blocks)
    return [
        block for block in blocks
        if not (
            counts[block] >= 3
            and len(block.split()) <= 20
            and structure_metadata(block)[0] is None
        )
    ]


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
    current_page: int | None = None
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


def current_structure_metadata(state: ChunkingState) -> dict:
    """Copy the current legal hierarchy into chunk metadata."""
    metadata = {}
    for kind in STRUCTURE_KINDS:
        structure = state.current_structures[kind]
        metadata[f"{kind}_number"] = structure["number"]
        metadata[f"{kind}_title"] = structure["title"]
        metadata[f"{kind}_order"] = structure["order"]
    return metadata


def add_chunk(
    state: ChunkingState,
    content: str,
    group_id: int,
    part: int,
    document_name: str,
    structure_meta: dict | None = None,
    legal_unit_paths: list[dict] | None = None,
) -> None:
    meta = {
        "chunk_number": len(state.chunks) + 1,
        "document_name": document_name,
        "group_id": group_id,
        "part": part,
        "page_number": state.current_page,
    }
    meta.update(structure_meta or current_structure_metadata(state))
    if legal_unit_paths:
        meta["legal_unit_paths"] = legal_unit_paths

    state.chunks.append(Document(content=content, meta=meta))


def chunk_text(
    text: str,
    max_words: int = DEFAULT_MAX_WORDS,
    document_name: str = "Unknown document",
    document_type: str | None = None,
) -> list[Document]:
    uses_legal_structure = document_type in {
        "statute", "regulation", "contract", "constitutional_decision",
    } if document_type else True
    blocks = searchable_blocks(text)
    state = ChunkingState()
    block_index = 0

    while block_index < len(blocks):
        group_id = block_index
        block = blocks[block_index]
        page_match = PAGE_MARKER_PATTERN.fullmatch(block)
        if page_match:
            state.current_page = int(page_match.group(1))
            block_index += 1
            continue
        current_kind = structure_metadata(block)[0] if uses_legal_structure else None
        if uses_legal_structure:
            update_chunking_structure(state, block)
        primary_structure = current_structure_metadata(state)
        legal_unit_paths = [primary_structure] if uses_legal_structure else []
        block_words = block.split()

        if len(block_words) > max_words:
            parts = split_long_block(block, max_words)
            for part, content in enumerate(parts):
                add_chunk(
                    state,
                    content,
                    group_id,
                    part,
                    document_name,
                    primary_structure,
                    legal_unit_paths,
                )
            block_index += 1
            continue

        while block_index + 1 < len(blocks):
            next_block = blocks[block_index + 1]
            next_kind = structure_metadata(next_block)[0] if uses_legal_structure else None
            if PAGE_MARKER_PATTERN.fullmatch(next_block):
                break

            can_merge = not current_kind and not next_kind
            if current_kind in {"article", "clause", "item", "subitem"}:
                current_level = STRUCTURE_KINDS.index(current_kind)
                next_level = (
                    STRUCTURE_KINDS.index(next_kind) if next_kind else current_level
                )
                can_merge = next_level > current_level or (
                    current_kind in {"clause", "item", "subitem"}
                    and next_level == current_level
                )

            if (
                not can_merge
                or len(block_words) + len(next_block.split()) > max_words
            ):
                break

            block = f"{block}\n\n{next_block}"
            block_words = block.split()
            block_index += 1
            if uses_legal_structure and next_kind:
                update_chunking_structure(state, next_block)
                legal_unit_paths.append(current_structure_metadata(state))

        add_chunk(
            state,
            block,
            group_id,
            0,
            document_name,
            primary_structure,
            legal_unit_paths,
        )
        block_index += 1

    return state.chunks
