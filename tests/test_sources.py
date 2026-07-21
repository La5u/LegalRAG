import pytest

from backend.sources import (
    CONTRACT_TERMS,
    DOCUMENT_SUMMARY_TERMS,
    SOURCE_ALIASES,
    explicit_source_terms,
    is_case_query,
    is_contract_query,
    is_document_summary_query,
    normalize_source_text,
    source_name_terms,
    source_name_overlap,
)


# ---------------------------------------------------------------------------
# normalize_source_text — strip prefixes, suffixes, special chars, lowercase
# ---------------------------------------------------------------------------
class TestNormalizeSourceText:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("documents/raw/statutes/개인정보보호법.pdf", "statutes/개인정보보호법"),
            ("개인정보보호법(법률).hwpx", "개인정보보호법법률"),
            ("  hello   world  ", "helloworld"),
            ("documents/raw/test.pdf", "test"),
        ],
    )
    def test_normalizes(self, text, expected):
        assert normalize_source_text(text) == expected

    def test_removes_pdf_extension(self):
        assert normalize_source_text("test.pdf") == "test"

    def test_case_insensitive_extension(self):
        assert normalize_source_text("test.PDF") == "test"

    def test_strips_special_chars(self):
        result = normalize_source_text("hello[world]★test")
        assert result == "helloworldtest"


# ---------------------------------------------------------------------------
# source_name_terms — extract normalized terms from name/path
# ---------------------------------------------------------------------------
class TestSourceNameTerms:
    def test_extracts_from_name(self):
        terms = source_name_terms("개인정보보호법(법률)", "documents/raw/statutes/개인정보보호법(법률).pdf")
        assert any("개인정보보호법" in t for t in terms)

    def test_filters_short_terms(self):
        terms = source_name_terms("abc", "short")
        assert all(len(t) >= 8 for t in terms)

    def test_deduplicates(self):
        terms = source_name_terms("개인정보보호법", "documents/raw/statutes/개인정보보호법.pdf")
        assert len(terms) == len(set(terms))


# ---------------------------------------------------------------------------
# source_name_overlap — check if query matches document name
# ---------------------------------------------------------------------------
class TestSourceNameOverlap:
    def test_exact_match(self):
        assert source_name_overlap("개인정보보호법", "개인정보보호법(법률)", "documents/raw/statutes/개인정보보호법(법률).pdf")

    def test_partial_match(self):
        assert source_name_overlap("개인정보보호", "개인정보보호법(법률)", "documents/raw/statutes/개인정보보호법(법률).pdf")

    def test_no_match(self):
        assert not source_name_overlap("헌법", "개인정보보호법(법률)", "documents/raw/statutes/개인정보보호법(법률).pdf")

    def test_generic_terms_only(self):
        # "법률" is in GENERIC_DOCUMENT_TOKENS, should not match
        assert not source_name_overlap("법률", "개인정보 보호법(법률)", "documents/raw/statutes/개인정보 보호법(법률).pdf")


# ---------------------------------------------------------------------------
# explicit_source_terms — extract file references from query
# ---------------------------------------------------------------------------
class TestExplicitSourceTerms:
    def test_extracts_filename(self):
        terms = explicit_source_terms('이 "개인정보보호법.pdf" 파일의 내용은?')
        assert any("개인정보보호법" in t for t in terms)

    def test_extracts_hwpx(self):
        terms = explicit_source_terms('이 "시행령.hwpx" 파일의 내용은?')
        assert any("시행령" in t for t in terms)

    def test_no_filename(self):
        assert explicit_source_terms("개인정보 처리 방침") == []


# ---------------------------------------------------------------------------
# is_document_summary_query — detect summary requests
# ---------------------------------------------------------------------------
class TestIsDocumentSummaryQuery:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("이 안내서의 주요 내용은 무엇인가요", True),
            ("핵심 내용을 알려주세요", True),
            ("어떤 대상을 다루고 있나요", True),
            ("개인정보 처리 방침", False),
        ],
    )
    def test_detects_summary(self, query, expected):
        assert is_document_summary_query(query) == expected

    def test_guidance_source_boosts_summary(self):
        assert is_document_summary_query("주요 내용", ["documents/raw/guidance/해설.pdf"])

    def test_non_guidance_source_not_summary(self):
        assert not is_document_summary_query("주요 내용", ["documents/raw/statutes/법.pdf"])


# ---------------------------------------------------------------------------
# is_contract_query — detect contract-related queries
# ---------------------------------------------------------------------------
class TestIsContractQuery:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("임대차 계약서의 내용은 무엇인가요", True),
            ("보증금 반환 조건", True),
            ("임대인의 의무", True),
            ("개인정보 처리 방침", False),
        ],
    )
    def test_detects_contract(self, query, expected):
        assert is_contract_query(query, []) == expected

    def test_contract_source_type(self):
        assert is_contract_query("일반 질문", ["documents/raw/contracts/계약서.pdf"])


# ---------------------------------------------------------------------------
# is_case_query — detect precedent/case queries
# ---------------------------------------------------------------------------
class TestIsCaseQuery:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("판례에서 이 문제는 어떻게 판단했나요", True),
            ("대법원 판결 요지", True),
            ("헌재 결정례", True),
            ("개인정보 처리 방침", False),
        ],
    )
    def test_detects_case(self, query, expected):
        assert is_case_query(query, []) == expected

    def test_precedent_source_type(self):
        assert is_case_query("일반 질문", ["documents/raw/precedents/판례.pdf"])

    def test_constitutional_decision_source_type(self):
        assert is_case_query("일반 질문", ["documents/raw/constitutional_decisions/헌재.pdf"])
