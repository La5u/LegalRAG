from dataclasses import dataclass, field
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from backend.config import DEFAULT_MAX_WORDS

MAX_SHORT_BLOCK_WORDS = 50
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？다요죠])\s+")
# Korean law structure used here:
# 법령 = law/document, 장 = chapter, 절 = section, 조문 = article,
# 항 = clause, 호 = item, 목 = sub-item.
# Example: "제1조(목적)" means "Article 1 (Purpose)".
STRUCTURE_PATTERNS = [
    ("chapter", re.compile(r"^(?P<number>제\d+장)\s*(?P<title>.*)")),
    (
        "chapter",
        re.compile(
            r"^(?P<number>Article\s+[IVXLCDM\d]+)\.?\s*(?P<title>.*)",
            re.IGNORECASE,
        ),
    ),
    ("chapter", re.compile(r"^(?P<number>ARTICLE\s+[A-Z]+)\.?\s*(?P<title>.*)")),
    ("section", re.compile(r"^(?P<number>제\d+절)\s*(?P<title>.*)")),
    ("article", re.compile(r"^(?P<number>제\d+조(?:의\d+)?)(?:\((?P<title>[^)]*)\))?")),
    (
        "article",
        re.compile(
            r"^(?P<number>Section\s+\d+)\.?\s*(?P<title>.*)",
            re.IGNORECASE,
        ),
    ),
    ("clause", re.compile(r"^(?P<number>[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*(?P<title>.*)")),
    ("item", re.compile(r"^(?P<number>\d+\.)\s*(?P<title>.*)")),
    ("subitem", re.compile(r"^(?P<number>[가-하]\.)\s*(?P<title>.*)")),
]
STRUCTURE_KINDS = list(dict.fromkeys(kind for kind, _pattern in STRUCTURE_PATTERNS))


@dataclass(frozen=True)
class Document:
    content: str
    meta: dict = field(default_factory=dict)
    score: float | None = None


def local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def extract_hwpx_paragraphs(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    paragraphs = []

    for paragraph in root.iter():
        if local_name(paragraph.tag) != "p":
            continue

        text_parts = [
            node.text
            for node in paragraph.iter()
            if local_name(node.tag) == "t" and node.text
        ]
        text = "".join(text_parts).strip()
        if text:
            paragraphs.append(text)

    return paragraphs


def load_hwpx(path: Path) -> str:
    paragraphs = []

    with zipfile.ZipFile(path) as hwpx:
        section_names = sorted(
            name
            for name in hwpx.namelist()
            if name.startswith("Contents/section") and name.endswith(".xml")
        )

        for section_name in section_names:
            xml_text = hwpx.read(section_name).decode("utf-8")
            paragraphs.extend(extract_hwpx_paragraphs(xml_text))

    # 조문 = Article.
    # Korean laws usually use "제1조", "제2조" and similar article markers.
    # Keeping paragraph breaks here helps the later graph step detect articles.
    return "\n\n".join(paragraphs)


def load_txt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start_marker = "THE CONSTITUTION OF THE UNITED STATES OF AMERICA"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start != -1 and end != -1 and start < end:
        return text[start:end].strip()
    return text


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".hwpx":
        return load_hwpx(path)
    if suffix == ".txt":
        return load_txt(path)
    raise ValueError("Only HWPX and TXT documents are supported.")


def split_long_block(block: str, max_words: int) -> list[str]:
    parts = []
    current_words = []

    for sentence in SENTENCE_SPLIT_PATTERN.split(block):
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


def chunk_text(
    text: str,
    max_words: int = DEFAULT_MAX_WORDS,
    document_name: str = "Unknown document",
) -> list[Document]:
    blocks = [
        " ".join(block.split())
        for block in re.split(r"(?:\n\s*\n+|[ \t]{4,})", text)
        if block.strip()
    ]
    chunks = []
    block_index = 0
    structure_counts = {}
    structure_orders = {}
    current_structures = {
        kind: {"number": None, "title": None, "order": None}
        for kind in STRUCTURE_KINDS
    }

    def update_structure(block: str) -> None:
        # When a new legal structure heading begins, remember it.
        # Later chunks inherit this metadata until a newer heading appears.
        kind, number, title = structure_metadata(block)
        if not kind:
            return

        kind_index = STRUCTURE_KINDS.index(kind)
        parent_key = tuple(
            current_structures[parent_kind]["number"]
            for parent_kind in STRUCTURE_KINDS[:kind_index]
        )
        count_key = (kind, parent_key)
        order_key = (kind, parent_key, number)
        if order_key not in structure_orders:
            structure_counts[count_key] = structure_counts.get(count_key, 0) + 1
            structure_orders[order_key] = structure_counts[count_key]

        current_structures[kind] = {
            "number": number,
            "title": title,
            "order": structure_orders[order_key],
        }
        for child_kind in STRUCTURE_KINDS[kind_index + 1 :]:
            current_structures[child_kind] = {
                "number": None,
                "title": None,
                "order": None,
            }

    def add_chunk(
        content: str,
        group_id: int,
        part: int,
    ) -> None:
        meta = {
            "chunk_number": len(chunks) + 1,
            "document_name": document_name,
            "group_id": group_id,
            "part": part,
        }
        for kind in STRUCTURE_KINDS:
            structure = current_structures[kind]
            meta[f"{kind}_number"] = structure["number"]
            meta[f"{kind}_title"] = structure["title"]
            meta[f"{kind}_order"] = structure["order"]

        chunks.append(
            Document(
                content=content,
                # These fields are the bridge from plain chunks to the graph:
                # Structure nodes are created from them during Neo4j indexing.
                meta=meta,
            )
        )

    while block_index < len(blocks):
        group_id = block_index
        block = blocks[block_index]
        update_structure(block)
        block_words = block.split()

        if len(block_words) > max_words:
            parts = split_long_block(block, max_words)
            for part, content in enumerate(parts):
                add_chunk(content, group_id, part)
            block_index += 1
            continue

        if len(block_words) <= MAX_SHORT_BLOCK_WORDS and block_index + 1 < len(blocks):
            next_block = blocks[block_index + 1]
            next_kind = structure_metadata(next_block)[0]
            if (
                not next_kind
                and len(block_words) + len(next_block.split()) <= max_words
            ):
                block = f"{block}\n\n{next_block}"
                block_index += 1

        add_chunk(block, group_id, 0)
        block_index += 1

    return chunks
