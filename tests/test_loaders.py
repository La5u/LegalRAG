import pytest

from backend.loaders import (
    clean_pdf_line,
    compact_spaced_digits,
    extract_hwpx_paragraphs,
    is_pdf_noise_line,
    local_name,
    normalize_pdf_text,
    pdf_line_starts_paragraph,
    pdf_text_quality,
    pdf_text_score,
    REPLACEMENT_CHARACTER,
)


# ---------------------------------------------------------------------------
# local_name — strip XML namespace
# ---------------------------------------------------------------------------
class TestLocalName:
    @pytest.mark.parametrize(
        "tag, expected",
        [
            ("{http://www.hwpf.org/2010/hwp}p", "p"),
            ("{http://www.hwpf.org/2010/hwp}t", "t"),
            ("p", "p"),
            ("t", "t"),
            ("{http://example.com/ns}element", "element"),
        ],
    )
    def test_strips_namespace(self, tag, expected):
        assert local_name(tag) == expected


# ---------------------------------------------------------------------------
# compact_spaced_digits — "1 2 3" → "123"
# ---------------------------------------------------------------------------
class TestCompactSpacedDigits:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("제1 2 3조", "제123조"),
            ("123", "123"),
            ("abc", "abc"),
            ("1 2", "12"),
        ],
    )
    def test_compacts(self, text, expected):
        assert compact_spaced_digits(text) == expected


# ---------------------------------------------------------------------------
# clean_pdf_line — normalize, remove artifacts
# ---------------------------------------------------------------------------
class TestCleanPdfLine:
    def test_normalizes_whitespace(self):
        assert clean_pdf_line("  hello   world  ") == "hello world"

    def test_removes_artifact_line(self):
        assert clean_pdf_line("123.456.789 >") == ""

    def test_removes_leader_dots(self):
        assert clean_pdf_line("hello ·········· world") == ""

    def test_removes_replacement_chars(self):
        assert clean_pdf_line(f"abc {REPLACEMENT_CHARACTER * 5} def") == ""

    def test_compacts_spaced_digits(self):
        assert clean_pdf_line("제1 2 3조") == "제123조"

    def test_returns_empty_for_only_whitespace(self):
        assert clean_pdf_line("   ") == ""


# ---------------------------------------------------------------------------
# is_pdf_noise_line —法처처 footer patterns
# ---------------------------------------------------------------------------
class TestIsPdfNoiseLine:
    @pytest.mark.parametrize(
        "line, expected",
        [
            ("법제처 1234 국가법령정보센터", True),
            ("법제처 - / - 국가법령정보센터", True),
            ("법제처 국가법령정보센터", True),
            ("1 / 2 -", True),
            ("일반 텍스트", False),
            ("", False),
        ],
    )
    def test_noise_lines(self, line, expected):
        assert is_pdf_noise_line(line) == expected


# ---------------------------------------------------------------------------
# pdf_line_starts_paragraph — detect paragraph boundaries
# ---------------------------------------------------------------------------
class TestPdfLineStartsParagraph:
    @pytest.mark.parametrize(
        "line, expected",
        [
            ("제1장 총칙", True),
            ("제15조의2 목적", True),
            ("제1절 정의", True),
            ("【 판시사항 】", True),
            ("① 개인정보", True),
            ("1. 정의", True),
            ("가. 개인정보", True),
            ("이 법은 개인정보", False),
            ("", False),
        ],
    )
    def test_paragraph_starts(self, line, expected):
        assert pdf_line_starts_paragraph(line) == expected


# ---------------------------------------------------------------------------
# extract_hwpx_paragraphs — XML text extraction
# ---------------------------------------------------------------------------
class TestExtractHwpxParagraphs:
    def test_extracts_text_from_t_nodes(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <hwp:document xmlns:hwp="http://www.hwpf.org/2010/hwp">
          <hwp:p>
            <hwp:t>Hello</hwp:t>
            <hwp:t>World</hwp:t>
          </hwp:p>
          <hwp:p>
            <hwp:t>Second</hwp:t>
          </hwp:p>
        </hwp:document>"""
        paragraphs = extract_hwpx_paragraphs(xml)
        assert paragraphs == ["HelloWorld", "Second"]

    def test_skips_empty_paragraphs(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <hwp:document xmlns:hwp="http://www.hwpf.org/2010/hwp">
          <hwp:p><hwp:t>Text</hwp:t></hwp:p>
          <hwp:p></hwp:p>
        </hwp:document>"""
        paragraphs = extract_hwpx_paragraphs(xml)
        assert paragraphs == ["Text"]


# ---------------------------------------------------------------------------
# normalize_pdf_text — full normalization pipeline
# ---------------------------------------------------------------------------
class TestNormalizePdfText:
    def test_joins_paragraphs(self):
        text = "제1조 목적\n이 법은\n\n제2조 정의"
        result = normalize_pdf_text(text)
        assert "제1조 목적 이 법은" in result

    def test_removes_noise_lines(self):
        text = "제1조 목적\n법제처 국가법령정보센터\n\n제2조 정의"
        result = normalize_pdf_text(text)
        assert "법제처" not in result

    def test_removes_replacement_runs(self):
        text = f"제1조 목적\n{REPLACEMENT_CHARACTER * 5}\n\n제2조 정의"
        result = normalize_pdf_text(text)
        assert REPLACEMENT_CHARACTER not in result

    def test_removes_artifact_lines(self):
        text = "제1조 목적\n1234.56.789 >\n\n제2조 정의"
        result = normalize_pdf_text(text)
        assert "1234.56.789" not in result

    def test_preserves_paragraph_breaks(self):
        text = "제1조 목적\n\n제2조 정의"
        result = normalize_pdf_text(text)
        assert "제1조 목적" in result
        assert "제2조 정의" in result

    def test_handles_crlf(self):
        text = "제1조 목적\r\n이 법은\r\n\r\n제2조 정의"
        result = normalize_pdf_text(text)
        assert "\r" not in result


# ---------------------------------------------------------------------------
# pdf_text_quality — character-level quality metrics
# ---------------------------------------------------------------------------
class TestPdfTextQuality:
    def test_counts_korean(self):
        quality = pdf_text_quality("이 법은 개인정보를 보호한다")
        assert quality["korean"] == 12

    def test_counts_hanja(self):
        quality = pdf_text_quality("个人信息保護法")
        assert quality["hanja"] == 7

    def test_counts_articles(self):
        quality = pdf_text_quality("제1조 목적\n제15조의2 범위")
        assert quality["articles"] == 2

    def test_counts_replacement(self):
        quality = pdf_text_quality(f"abc {REPLACEMENT_CHARACTER * 3} def")
        assert quality["replacement"] == 3

    def test_length(self):
        quality = pdf_text_quality("hello")
        assert quality["length"] == 5


# ---------------------------------------------------------------------------
# pdf_text_score — quality scoring
# ---------------------------------------------------------------------------
class TestPdfTextScore:
    def test_good_score_for_korean_legal_text(self):
        text = "제1조 목적\n이 법은 개인정보를 보호한다\n제15조의2 범위"
        score = pdf_text_score(text)
        assert score > 50

    def test_low_score_for_replacement_heavy(self):
        text = f"abc {REPLACEMENT_CHARACTER * 20}"
        score = pdf_text_score(text)
        assert score < 0

    def test_empty_text(self):
        score = pdf_text_score("")
        assert score == 0
