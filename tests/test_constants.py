import re

import pytest

from backend.constants import (
    ARTICLE_PATTERN,
    CITATION_PATTERN,
    CIRCLED_CLAUSE_MARKS,
    CONCEPT_STOP_TERMS,
    FULL_CITATION_PATTERN,
    LOCAL_CLAUSE_PATTERN,
    STRUCTURE_KINDS,
    STRUCTURE_PATTERNS,
    TOKEN_PATTERN,
)


# ---------------------------------------------------------------------------
# ARTICLE_PATTERN — matches "제N조" and "제N조의N"
# ---------------------------------------------------------------------------
class TestArticlePattern:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("제15조", ["제15조"]),
            ("제37조의2", ["제37조의2"]),
            ("제1조와 제2조", ["제1조", "제2조"]),
            ("제100조의10", ["제100조의10"]),
            # Should NOT match
            ("제조", []),
            ("15조", []),
            ("조제15", []),
        ],
    )
    def test_article_pattern(self, text, expected):
        assert ARTICLE_PATTERN.findall(text) == expected


# ---------------------------------------------------------------------------
# CITATION_PATTERN — article + optional clause/item
# ---------------------------------------------------------------------------
class TestCitationPattern:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("제15조", ["제15조"]),
            ("제15조제1항", ["제15조제1항"]),
            ("제15조제1호", ["제15조제1호"]),
            ("제15조①", ["제15조①"]),
            ("제15조의2제1항", ["제15조의2제1항"]),
            ("제15조제1항제1호", ["제15조제1항제1호"]),
        ],
    )
    def test_citation_pattern(self, text, expected):
        assert CITATION_PATTERN.findall(text) == expected


# ---------------------------------------------------------------------------
# FULL_CITATION_PATTERN — optional law name + article + clause
# ---------------------------------------------------------------------------
class TestFullCitationPattern:
    def test_with_law_name(self):
        text = '"개인정보 보호법" 제15조제1항'
        match = FULL_CITATION_PATTERN.search(text)
        assert match is not None
        assert match.group("law_name") == "개인정보 보호법"
        assert match.group("article") == "제15조"
        assert match.group("clause") == "제1항"

    def test_without_law_name(self):
        text = "제15조제2호"
        match = FULL_CITATION_PATTERN.search(text)
        assert match is not None
        assert match.group("law_name") is None
        assert match.group("article") == "제15조"
        assert match.group("clause") == "제2호"

    def test_with_circled_clause(self):
        text = "제37조의2①"
        match = FULL_CITATION_PATTERN.search(text)
        assert match is not None
        assert match.group("article") == "제37조의2"
        assert match.group("clause") == "①"

    def test_findall_multiple(self):
        text = "제15조 및 제16조제1항"
        matches = FULL_CITATION_PATTERN.findall(text)
        assert len(matches) >= 2


# ---------------------------------------------------------------------------
# LOCAL_CLAUSE_PATTERN — clause references within same article
# ---------------------------------------------------------------------------
class TestLocalClausePattern:
    @pytest.mark.parametrize(
        "text, expected_clauses",
        [
            ("제1항", ["제1항"]),
            ("제1호", ["제1호"]),
            ("①", ["①"]),
            ("가목", ["가목"]),
            ("제1항제2호", ["제1항", "제2호"]),
        ],
    )
    def test_local_clause(self, text, expected_clauses):
        matches = LOCAL_CLAUSE_PATTERN.findall(text)
        assert matches == expected_clauses


# ---------------------------------------------------------------------------
# TOKEN_PATTERN — Korean/English word extraction
# ---------------------------------------------------------------------------
class TestTokenPattern:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("개인정보 처리", ["개인정보", "처리"]),
            ("Hello World", ["Hello", "World"]),
            ("테스트123", ["테스트123"]),
            # Single char excluded (< 2)
            ("a 테스트", ["테스트"]),
            # Special chars excluded
            ("개인정보!@#", ["개인정보"]),
            # Korean mixed with dots/dashes
            ("한글·테스트", ["한글·테스트"]),
        ],
    )
    def test_token_pattern(self, text, expected):
        assert TOKEN_PATTERN.findall(text) == expected


# ---------------------------------------------------------------------------
# CONCEPT_STOP_TERMS — noise words filtered from concept names
# ---------------------------------------------------------------------------
class TestConceptStopTerms:
    @pytest.mark.parametrize(
        "term",
        [
            "일반규칙",
            "개인",
            "법",
            "경우",
            "사항",
            "제1항",
            "제2호",
        ],
    )
    def test_stop_terms_present(self, term):
        assert term in CONCEPT_STOP_TERMS

    def test_not_in_stop_terms(self):
        assert "개인정보처리자" not in CONCEPT_STOP_TERMS
        assert "정보주체" not in CONCEPT_STOP_TERMS


# ---------------------------------------------------------------------------
# STRUCTURE_PATTERNS — legal document structure detection
# ---------------------------------------------------------------------------
class TestStructurePatterns:
    @pytest.mark.parametrize(
        "text, expected_kind, expected_number",
        [
            ("제1장 총칙", "chapter", "제1장"),
            ("제15조(목적)", "article", "제15조"),
            ("① 개인정보란", "clause", "①"),
            ("1. 수집 항목", "item", "1."),
            ("가. 성명", "subitem", "가."),
            ("제1절 목적", "section", "제1절"),
        ],
    )
    def test_structure_match(self, text, expected_kind, expected_number):
        for kind, pattern in STRUCTURE_PATTERNS:
            match = pattern.match(text)
            if match:
                assert kind == expected_kind, f"Expected {expected_kind}, got {kind} for '{text}'"
                assert match.group("number") == expected_number
                return
        pytest.fail(f"No structure pattern matched '{text}'")

    def test_no_match_on_plain_text(self):
        text = "개인정보 처리에 관한 법률"
        for kind, pattern in STRUCTURE_PATTERNS:
            assert not pattern.match(text), f"Unexpected match for '{text}'"


# ---------------------------------------------------------------------------
# STRUCTURE_KINDS — ordered unique kinds
# ---------------------------------------------------------------------------
class TestStructureKinds:
    def test_kinds_are_ordered(self):
        expected = ["chapter", "section", "article", "clause", "item", "subitem"]
        assert STRUCTURE_KINDS == expected

    def test_kinds_unique(self):
        assert len(STRUCTURE_KINDS) == len(set(STRUCTURE_KINDS))


# ---------------------------------------------------------------------------
# CIRCLED_CLAUSE_MARKS
# ---------------------------------------------------------------------------
class TestCircledClauseMarks:
    def test_marks_present(self):
        assert "①" in CIRCLED_CLAUSE_MARKS
        assert "⑳" in CIRCLED_CLAUSE_MARKS
        assert len(CIRCLED_CLAUSE_MARKS) == 20
