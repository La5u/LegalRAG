import pytest

from backend.graph_index import (
    clean_concept_name,
    clean_relation_rows,
    is_legal_reference_concept,
    is_noisy_concept,
)


# ---------------------------------------------------------------------------
# clean_concept_name — strip particles and artifacts
# ---------------------------------------------------------------------------
class TestCleanConceptName:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("개인정보처리자는", "개인정보처리자"),
            ("정보주체에게", "정보주체"),
            ("성명을", "성명"),
            # Already clean
            ("개인정보", "개인정보"),
            # Empty / None
            ("", ""),
            (None, ""),
            # Leading "n" artifact before Korean char
            ("n개인정보", "개인정보"),
            # Leading "n" before non-Korean — kept
            ("n123", "n123"),
        ],
    )
    def test_clean_concept_name(self, value, expected):
        assert clean_concept_name(value) == expected


# ---------------------------------------------------------------------------
# is_noisy_concept — filter noise from concept names
# ---------------------------------------------------------------------------
class TestIsNoisyConcept:
    @pytest.mark.parametrize(
        "value, expected",
        [
            # Empty / None
            ("", True),
            (None, True),
            # Single char
            ("법", True),
            ("수", True),
            # Stop terms
            ("일반규칙", True),
            ("개인", True),
            ("경우", True),
            ("사항", True),
            # Legal references (article/clause numbers)
            ("제15조", True),
            ("제1항", True),
            ("제1호", True),
            ("가목", True),
            # Clean concepts — not noisy
            ("개인정보처리자", False),
            ("정보주체", False),
            ("개인정보", False),
        ],
    )
    def test_noisy_concept(self, value, expected):
        assert is_noisy_concept(value) == expected


# ---------------------------------------------------------------------------
# is_legal_reference_concept — article/clause number noise
# ---------------------------------------------------------------------------
class TestIsLegalReferenceConcept:
    @pytest.mark.parametrize(
        "value, expected",
        [
            # Article numbers
            ("제15조", True),
            ("제37조의2", True),
            ("15조", True),
            # Clause numbers
            ("제1항", True),
            ("제2호", True),
            ("①", True),
            # Sub-item
            ("가목", True),
            # Normal concept — not a reference
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
    def _row(self, source: str, target: str) -> dict:
        return {
            "source": source,
            "target": target,
            "relation": "HAS_DUTY",
            "sentence": "test sentence",
            "source_path": "test.hwpx",
            "document_name": "test.hwpx",
            "chunk_id": "chunk-1",
            "chunk_number": 1,
        }

    def test_clean_concepts_kept(self):
        rows = [self._row("개인정보처리자", "정보주체")]
        result = clean_relation_rows(rows, "source", "target")
        assert len(result) == 1
        assert result[0]["source"] == "개인정보처리자"
        assert result[0]["target"] == "정보주체"

    def test_noisy_source_filtered(self):
        rows = [self._row("법", "정보주체")]
        result = clean_relation_rows(rows, "source", "target")
        assert result == []

    def test_noisy_target_filtered(self):
        rows = [self._row("개인정보처리자", "제15조")]
        result = clean_relation_rows(rows, "source", "target")
        assert result == []

    def test_identical_source_target_filtered(self):
        rows = [self._row("개인정보", "개인정보")]
        result = clean_relation_rows(rows, "source", "target")
        assert result == []

    def test_particle_stripping_affects_equality(self):
        # After stripping, "개인정보처리자는" == "개인정보처리자" → same → filtered
        rows = [self._row("개인정보처리자는", "개인정보처리자")]
        result = clean_relation_rows(rows, "source", "target")
        assert result == []

    def test_multiple_rows_mixed(self):
        rows = [
            self._row("개인정보처리자", "정보주체"),  # clean
            self._row("법", "개인정보"),               # noisy source
            self._row("정보주체", "성명"),              # clean
        ]
        result = clean_relation_rows(rows, "source", "target")
        assert len(result) == 2
        assert result[0]["source"] == "개인정보처리자"
        assert result[1]["source"] == "정보주체"


# ---------------------------------------------------------------------------
# Concept name noise — word endings and article numbers don't leak into nodes
# ---------------------------------------------------------------------------
class TestConceptNameNoiseFiltering:
    """Verify that Korean word endings (particles) and article references
    do not appear as concept names in the knowledge graph."""

    @pytest.mark.parametrize(
        "raw, should_be_noisy",
        [
            # Word endings that should NOT become concepts
            ("개인정보처리자는", False),  # strips to clean form
            ("정보주체에게", False),       # strips to clean form
            # But these are noisy:
            ("는", True),     # single particle
            ("은", True),     # single particle
            ("를", True),     # single particle
            ("을", True),     # single particle
            # Article numbers — must not appear as concepts
            ("제15조", True),
            ("제37조의2", True),
            ("제1항", True),
            ("제1호", True),
            ("가목", True),
            # Real concepts — kept
            ("개인정보처리자", False),
            ("정보주체", False),
            ("수집ㆍ이용", False),
        ],
    )
    def test_word_endings_and_articles_filtered(self, raw, should_be_noisy):
        cleaned = clean_concept_name(raw)
        assert is_noisy_concept(cleaned) == should_be_noisy
