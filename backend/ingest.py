from dataclasses import dataclass, field
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium

from backend.config import DEFAULT_MAX_WORDS
from backend.constants import ARTICLE_PATTERN
from backend.text import split_sentences

MAX_SHORT_BLOCK_WORDS = 50
KOREAN_PATTERN = re.compile(r"[가-힣]")
HANJA_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
PDF_PARAGRAPH_START_PATTERN = re.compile(
    r"^(제\d+장|제\d+절|제\d+조(?:의\d+)?|【[^】]+】|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|\d+\.|[가-하]\.)"
)
PDF_NOISE_LINE_PATTERN = re.compile(
    r"^(법제처\s*\d*\s*국가법령정보센터|법제처\s*-?\s*/\s*-\s*국가법령정보센터|-?\s*\d+\s*/\s*\d+\s*-?)$"
)
PDF_ARTIFACT_LINE_PATTERN = re.compile(r"^[\d. ]+>$")
PDF_LEADER_LINE_PATTERN = re.compile(r"(?:[.·]\s*){8,}")
SPACED_DIGITS_PATTERN = re.compile(r"(?<!\d)(\d(?:\s+\d)+)(?!\d)")
REPLACEMENT_CHARACTER = "\ufffd"
REPLACEMENT_RUN_PATTERN = re.compile(f"{REPLACEMENT_CHARACTER}{{3,}}")
MIN_PDF_TEXT_LENGTH = 200
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
            try:
                paragraphs.extend(extract_hwpx_paragraphs(xml_text))
            except ET.ParseError as error:
                raise ValueError(
                    f"Malformed XML in {path.name}:{section_name}: {error}"
                ) from error

    # 조문 = Article.
    # Korean laws usually use "제1조", "제2조" and similar article markers.
    # Keeping paragraph breaks here helps the later graph step detect articles.
    return "\n\n".join(paragraphs)


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def compact_spaced_digits(text: str) -> str:
    return SPACED_DIGITS_PATTERN.sub(
        lambda match: match.group(1).replace(" ", ""),
        text,
    )


def clean_pdf_line(line: str) -> str:
    line = " ".join(line.split())
    if (
        PDF_ARTIFACT_LINE_PATTERN.match(line)
        or PDF_LEADER_LINE_PATTERN.search(line)
        or REPLACEMENT_RUN_PATTERN.search(line)
    ):
        return ""
    line = compact_spaced_digits(line)
    return line.strip()


def is_pdf_noise_line(line: str) -> bool:
    return bool(PDF_NOISE_LINE_PATTERN.match(line))


def pdf_line_starts_paragraph(line: str) -> bool:
    return bool(PDF_PARAGRAPH_START_PATTERN.match(line))


def normalize_pdf_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<!\n)(?=【[^】]+】)", "\n", text)
    paragraphs = []
    current_lines = []

    for raw_line in text.splitlines():
        line = clean_pdf_line(raw_line)
        if not line:
            if current_lines:
                paragraphs.append(" ".join(current_lines))
                current_lines = []
            continue
        if is_pdf_noise_line(line):
            continue
        if pdf_line_starts_paragraph(line) and current_lines:
            paragraphs.append(" ".join(current_lines))
            current_lines = []
        current_lines.append(line)

    if current_lines:
        paragraphs.append(" ".join(current_lines))

    return "\n\n".join(paragraphs)


def pdf_text_quality(text: str) -> dict[str, int]:
    return {
        "length": len(text),
        "korean": len(KOREAN_PATTERN.findall(text)),
        "hanja": len(HANJA_PATTERN.findall(text)),
        "articles": len(ARTICLE_PATTERN.findall(text)),
        "replacement": text.count(REPLACEMENT_CHARACTER),
    }


def pdf_text_score(text: str) -> int:
    quality = pdf_text_quality(text)
    return (
        quality["length"]
        + quality["korean"] * 2
        + quality["hanja"] * 2
        + quality["articles"] * 25
        - quality["replacement"] * 50
    )


def extract_pdf_with_pdfium(path: Path) -> str:
    pages = []

    pdf = pdfium.PdfDocument(path)
    for page in pdf:
        text_page = page.get_textpage()
        text = text_page.get_text_range().strip()
        if text:
            pages.append(text)

    return normalize_pdf_text("\n\n".join(pages))


def extract_pdfplumber_parts(path: Path) -> tuple[str, int]:
    pages = []
    table_count = 0

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True) or page.extract_text() or ""
            tables = page.extract_tables()
            table_count += len(tables)
            table_text = []
            for table in tables:
                for row in table:
                    cells = [cell.strip() for cell in row if cell and cell.strip()]
                    if cells:
                        table_text.append(" | ".join(cells))
            combined = "\n".join(part for part in [text.strip(), "\n".join(table_text)] if part)
            if combined:
                pages.append(combined)

    return normalize_pdf_text("\n\n".join(pages)), table_count


def _prefer_pdfplumber(
    pdfium_text: str,
    pdfplumber_text: str | None = None,
    pdfplumber_table_count: int = 0,
) -> bool:
    pdfium_quality = pdf_text_quality(pdfium_text)
    if pdfium_quality["length"] < MIN_PDF_TEXT_LENGTH:
        return True
    if pdfium_quality["replacement"] > 0:
        return True
    if pdfium_quality["korean"] == 0:
        return True
    if pdfplumber_text is None:
        return False
    pdfplumber_quality = pdf_text_quality(pdfplumber_text)
    if pdfplumber_quality["hanja"] > pdfium_quality["hanja"]:
        return True
    if pdfplumber_quality["articles"] > pdfium_quality["articles"]:
        return True
    if pdfplumber_table_count > 0 and pdf_text_score(pdfplumber_text) > pdf_text_score(pdfium_text) * 1.2:
        return True
    return False


def load_pdf(path: Path) -> str:
    pdfium_text = extract_pdf_with_pdfium(path)
    if not _prefer_pdfplumber(pdfium_text):
        return pdfium_text
    pdfplumber_text, pdfplumber_table_count_ = extract_pdfplumber_parts(path)
    if _prefer_pdfplumber(pdfium_text, pdfplumber_text, pdfplumber_table_count_):
        return pdfplumber_text
    return pdfium_text


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".hwpx":
        return load_hwpx(path)
    if suffix == ".txt":
        return load_txt(path)
    if suffix == ".pdf":
        return load_pdf(path)
    raise ValueError("Only HWPX, PDF, and TXT documents are supported.")


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
