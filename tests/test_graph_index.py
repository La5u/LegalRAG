import pytest

from backend.graph_index import (
    case_number_from_source,
    clean_concept_name,
    clean_relation_rows,
    is_holding_chunk,
    is_legal_reference_concept,
    is_noisy_concept,
)


# ---------------------------------------------------------------------------
# case_number_from_source — extract case number from path/name
# ---------------------------------------------------------------------------
class TestCaseNumberFromSource:
    @pytest.mark.parametrize(
        "source_path, document_name, expected",
        [
            ("precedents/판례_2024다12345.pdf", "판례_2024다12345.pdf", "2024다12345"),
            ("precedents/헌재결정례_2023헌마100.pdf", "헌재결정례_2023헌마100.pdf", "2023헌마100"),
            ("precedents/lbox_precedent_12345.pdf", "lbox_precedent_12345.pdf", "lbox_precedent_12345"),
            ("precedents/other.pdf", "other.pdf", "other"),
            ("precedents/other.pdf", "other", "other"),
        ],
    )
    def test_extracts_case_number(self, source_path, document_name, expected):
        assert case_number_from_source(source_path, document_name) == expected

    def test_falls_back_to_source_path(self):
        result = case_number_from_source("some/path", "")
        assert result == "some/path"


# ---------------------------------------------------------------------------
# is_holding_chunk — detect 판시사항/판결요지/쟁점 chunks
# ---------------------------------------------------------------------------
class TestIsHoldingChunk:
    @pytest.mark.parametrize(
        "content, expected",
        [
            ("쟁점 개인정보의 정의와 범위", True),
            ("판시사항 개인정보보호법의 적용 범위", True),
            ("주문 피고는 원고에게 100만 원을 지급하라", True),
            ("이유 이 사건 기록과 변론을 종합하면", True),
            ("판결요지 개인정보보호법 제15조에 따라", True),
            ("결정요지 헌법상 기본권의 제한은", True),
        ],
    )
    def test_detects_holding(self, content, expected):
        assert is_holding_chunk(content) == expected

    @pytest.mark.parametrize(
        "content, expected",
        [
            ("【판시사항】", False),
            ("【판결요지】", False),
            ("【결정요지】", False),
            ("가. 개인정보의 정의", False),
            ("일반적인 내용입니다.", False),
            ("", False),
        ],
    )
    def test_rejects_non_holding(self, content, expected):
        assert is_holding_chunk(content) == expected


# ---------------------------------------------------------------------------
# clean_concept_name — strip particles, normalize
# ---------------------------------------------------------------------------
class TestCleanConceptName:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("개인정보를", "개인정보"),
            ("개인정보", "개인정보"),
            ("  성명  ", "성명"),
            (None, ""),
            ("", ""),
        ],
    )
    def test_cleans(self, value, expected):
        assert clean_concept_name(value) == expected

    def test_strips_n_prefix(self):
        # "n가" prefix from Kiwi
        assert clean_concept_name("n개인정보") == "개인정보"


# ---------------------------------------------------------------------------
# is_noisy_concept — stop terms, too short, legal references
# ---------------------------------------------------------------------------
class TestIsNoisyConcept:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("", True),
            ("a", True),
            ("개인정보", False),
            ("성명", False),
            ("일반규칙", True),  # CONCEPT_STOP_TERMS
            ("사항", True),  # CONCEPT_STOP_TERMS
            ("제1항", True),  # CONCEPT_STOP_TERMS + legal reference
            ("①", True),  # legal reference
        ],
    )
    def test_noisy_detection(self, value, expected):
        assert is_noisy_concept(value) == expected


# ---------------------------------------------------------------------------
# is_legal_reference_concept — article/clause number noise
# ---------------------------------------------------------------------------
class TestIsLegalReferenceConcept:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("제15조", True),
            ("제37조의2", True),
            ("15조", True),
            ("제1항", True),
            ("제2호", True),
            ("①", True),
            ("가목", True),
            ("개인정보처리자", False),
            ("정보주체", False),
            ("성명", False),
        ],
    )
    def test_legal_reference(self, value, expected):
        assert is_legal_reference_concept(value) == expected


# ---------------------------------------------------------------------------
# clean_relation_rows — filter noisy/identical source-target pairs
# ---------------------------------------------------------------------------
class TestCleanRelationRows:
    def test_removes_noisy_concepts(self):
        rows = [
            {"subject": "개인정보", "object": "사항"},
        ]
        result = clean_relation_rows(rows, "subject", "object")
        assert len(result) == 0

    def test_removes_identical_source_target(self):
        rows = [
            {"subject": "개인정보", "object": "개인정보"},
        ]
        result = clean_relation_rows(rows, "subject", "object")
        assert len(result) == 0

    def test_keeps_valid_rows(self):
        rows = [
            {"subject": "개인정보처리자", "object": "정보주체"},
        ]
        result = clean_relation_rows(rows, "subject", "object")
        assert len(result) == 1
        assert result[0]["subject"] == "개인정보처리자"
        assert result[0]["object"] == "정보주체"

    def test_strips_particles(self):
        rows = [
            {"subject": "개인정보를", "object": "보호한다"},
        ]
        result = clean_relation_rows(rows, "subject", "object")
        assert len(result) == 1
        assert result[0]["subject"] == "개인정보"
