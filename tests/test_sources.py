import pytest

from backend.sources import (
    SourceMatcher,
    explicit_source_terms,
    is_case_query,
    is_contract_query,
    is_document_summary_query,
    normalize_source_text,
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


def test_new_law_alias_targets_statute_not_similar_contract():
    records = [
        {
            "name": "상가건물 임대차보호법(법률)",
            "source_path": "documents/raw/statutes/상가건물 임대차보호법(법률).hwpx",
        },
        {
            "name": "상가건물 임대차 표준계약서",
            "source_path": "documents/raw/contracts/상가건물 임대차 표준계약서.pdf",
        },
    ]
    matcher = SourceMatcher(lambda _query, **_params: records)

    assert matcher.constraints("상가임대차법의 갱신 요건") == [
        "documents/raw/statutes/상가건물 임대차보호법(법률).hwpx"
    ]


def test_longest_source_alias_keeps_explicit_standard_contract():
    records = [
        {
            "name": "주택임대차보호법(법률)",
            "source_path": "documents/raw/statutes/주택임대차보호법(법률).hwpx",
        },
        {
            "name": "주택임대차 표준계약서",
            "source_path": "documents/raw/contracts/주택임대차 표준계약서.pdf",
        },
    ]
    matcher = SourceMatcher(lambda _query, **_params: records)

    assert matcher.constraints("주택임대차 표준계약서의 임대인 의무") == [
        "documents/raw/contracts/주택임대차 표준계약서.pdf"
    ]


def test_full_decree_name_overrides_shorter_act_alias():
    records = [
        {
            "name": "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법",
            "source_path": "documents/raw/statutes/인공지능 기본법.hwpx",
        },
        {
            "name": "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법 시행령",
            "source_path": "documents/raw/regulations/인공지능 기본법 시행령.hwpx",
        },
    ]
    matcher = SourceMatcher(lambda _query, **_params: records)

    assert matcher.constraints(
        "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법 시행령에 따르면"
    ) == ["documents/raw/regulations/인공지능 기본법 시행령.hwpx"]


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


# ---------------------------------------------------------------------------
# SourceMatcher — cached source constraints
# ---------------------------------------------------------------------------
class TestSourceMatcher:
    def test_exact_alias_does_not_expand_to_similarly_named_contracts(self):
        records = [
            {
                "name": "주택임대차 표준계약서(원본게시용).pdf",
                "source_path": "documents/raw/contracts/주택임대차 표준계약서(원본게시용).pdf",
            },
            {
                "name": "상가건물_임대차_표준계약서.pdf",
                "source_path": "documents/raw/contracts/상가건물_임대차_표준계약서.pdf",
            },
        ]
        matcher = SourceMatcher(lambda _query, **_parameters: records)

        matches = matcher.constraints(
            "주택임대차 표준계약서에서 임차인은 언제 차임을 지급하나요?"
        )

        assert matches == [
            "documents/raw/contracts/주택임대차 표준계약서(원본게시용).pdf"
        ]

    def test_category_constraints_use_indexed_source_types(self):
        records = [
            {
                "name": "법.pdf",
                "source_path": "documents/raw/statutes/법.pdf",
            },
            {
                "name": "판례.pdf",
                "source_path": "documents/raw/precedents/판례.pdf",
            },
        ]
        matcher = SourceMatcher(lambda _query, **_parameters: records)

        assert matcher.category_constraints("statute") == [
            "documents/raw/statutes/법.pdf"
        ]

    def test_invalidate_refreshes_document_index(self):
        records = [
            [
                {
                    "name": "민법(법률)(제21454호)(20260317).hwpx",
                    "source_path": "documents/raw/statutes/민법(법률)(제21454호)(20260317).hwpx",
                }
            ],
            [
                {
                    "name": "형법(법률)(제21450호)(20260312).hwpx",
                    "source_path": "documents/raw/statutes/형법(법률)(제21450호)(20260312).hwpx",
                }
            ],
        ]

        def run_read(_query, **_parameters):
            return records.pop(0)

        matcher = SourceMatcher(run_read)

        assert matcher.constraints("민법에서 권리능력은?") == [
            "documents/raw/statutes/민법(법률)(제21454호)(20260317).hwpx"
        ]
        matcher.invalidate()
        assert matcher.constraints("형법에서 죄형법정주의는?") == [
            "documents/raw/statutes/형법(법률)(제21450호)(20260312).hwpx"
        ]
