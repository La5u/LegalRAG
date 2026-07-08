import pytest

from backend.document_types import (
    document_type_from_source_path,
    has_structural_legal_units,
    is_precedent_document,
)


# ---------------------------------------------------------------------------
# document_type_from_source_path — infer type from directory structure
# ---------------------------------------------------------------------------
class TestDocumentTypeFromSourcePath:
    @pytest.mark.parametrize(
        "source_path, expected",
        [
            ("documents/raw/statutes/개인정보보호법.pdf", "statute"),
            ("documents/raw/regulations/시행령.pdf", "regulation"),
            ("documents/raw/contracts/표준계약서.pdf", "contract"),
            ("documents/raw/precedents/판례_2024다12345.pdf", "precedent"),
            ("documents/raw/constitutional_decisions/헌재결정.pdf", "constitutional_decision"),
            ("documents/raw/guidance/해설.pdf", "guidance"),
            ("documents/raw/qna/질의회신.pdf", "qna"),
            ("documents/raw/other/기타.pdf", "unknown"),
        ],
    )
    def test_detects_type(self, source_path, expected):
        assert document_type_from_source_path(source_path) == expected


# ---------------------------------------------------------------------------
# has_structural_legal_units — statute/regulation/contract
# ---------------------------------------------------------------------------
class TestHasStructuralLegalUnits:
    @pytest.mark.parametrize(
        "source_path, expected",
        [
            ("documents/raw/statutes/법.pdf", True),
            ("documents/raw/regulations/령.pdf", True),
            ("documents/raw/contracts/계약.pdf", True),
            ("documents/raw/precedents/판례.pdf", False),
            ("documents/raw/guidance/해설.pdf", False),
        ],
    )
    def test_structural_check(self, source_path, expected):
        assert has_structural_legal_units(source_path) == expected


# ---------------------------------------------------------------------------
# is_precedent_document — precedent or constitutional decision
# ---------------------------------------------------------------------------
class TestIsPrecedentDocument:
    @pytest.mark.parametrize(
        "source_path, expected",
        [
            ("documents/raw/precedents/판례.pdf", True),
            ("documents/raw/constitutional_decisions/헌재.pdf", True),
            ("documents/raw/statutes/법.pdf", False),
            ("documents/raw/guidance/해설.pdf", False),
        ],
    )
    def test_precedent_check(self, source_path, expected):
        assert is_precedent_document(source_path) == expected
