import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium

from backend.constants import ARTICLE_PATTERN

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
