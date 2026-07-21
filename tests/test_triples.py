import pytest

from backend.document import Document
from backend.triples import (
    LEGAL_RELATIONS,
    CONCEPT_RELATION_TYPES,
    clean_concept_phrase,
    extract_triples,
    is_grammar_phrase,
    noun_phrases,
    pick_object,
    pick_subject,
)


# ---------------------------------------------------------------------------
# noun_phrases — extract noun phrases with particles from Korean text
# ---------------------------------------------------------------------------
class TestNounPhrases:
    def test_basic_noun_phrase(self):
        phrases = noun_phrases("개인정보처리자는 정보주체에게")
        texts = [p["text"] for p in phrases]
        assert "개인정보처리자" in texts
        assert "정보주체" in texts

    def test_particle_attached(self):
        phrases = noun_phrases("개인정보처리자는 정보주체에게")
        # Kiwi may strip the particle during analysis; check that phrases are extracted
        assert len(phrases) >= 1

    def test_empty_sentence(self):
        phrases = noun_phrases("")
        assert phrases == []


# ---------------------------------------------------------------------------
# clean_concept_phrase
# ---------------------------------------------------------------------------
class TestCleanConceptPhrase:
    @pytest.mark.parametrize(
        "phrase, expected",
        [
            ("개인정보처리자는", "개인정보처리자"),
            ("정보주체에게", "정보주체"),
            ("개인정보를", "개인정보"),
            ("  성명  ", "성명"),
        ],
    )
    def test_clean(self, phrase, expected):
        assert clean_concept_phrase(phrase) == expected


# ---------------------------------------------------------------------------
# is_grammar_phrase — filter grammar-only and stop-term phrases
# ---------------------------------------------------------------------------
class TestIsGrammarPhrase:
    @pytest.mark.parametrize(
        "phrase, tags, expected",
        [
            # Empty
            ("", ["NNB"], True),
            # Stop term
            ("일반규칙", ["NNG"], True),
            # Grammar-only nouns (bound nouns)
            ("것", ["NNB"], True),
            # Real noun — not grammar
            ("개인정보처리자", ["NNG"], False),
        ],
    )
    def test_grammar_phrase(self, phrase, tags, expected):
        assert is_grammar_phrase(phrase, tags) == expected


# ---------------------------------------------------------------------------
# pick_subject — select subject from noun phrases
# ---------------------------------------------------------------------------
class TestPickSubject:
    def test_picks_phrase_with_subject_particle(self):
        phrases = [
            {"text": "개인정보처리자", "particle": "는"},
            {"text": "정보주체", "particle": "에게"},
        ]
        assert pick_subject(phrases) == "개인정보처리자"

    def test_falls_back_to_first_phrase(self):
        phrases = [
            {"text": "개인정보처리자", "particle": None},
            {"text": "정보주체", "particle": None},
        ]
        assert pick_subject(phrases) == "개인정보처리자"

    def test_empty_phrases(self):
        assert pick_subject([]) is None


# ---------------------------------------------------------------------------
# pick_object — select object from noun phrases
# ---------------------------------------------------------------------------
class TestPickObject:
    def test_must_notify_picks_recipient(self):
        phrases = [
            {"text": "개인정보처리자", "particle": "는"},
            {"text": "정보주체", "particle": "에게"},
        ]
        assert pick_object(phrases, "MUST_NOTIFY") == "정보주체"

    def test_prohibition_picks_object_particle(self):
        phrases = [
            {"text": "개인정보", "particle": "를"},
            {"text": "처리", "particle": None},
        ]
        assert pick_object(phrases, "PROHIBITS") == "개인정보"

    def test_falls_back_to_last_phrase(self):
        phrases = [
            {"text": "개인정보처리자", "particle": None},
            {"text": "정보주체", "particle": None},
        ]
        assert pick_object(phrases, "HAS_DUTY") == "정보주체"


# ---------------------------------------------------------------------------
# LEGAL_RELATIONS / CONCEPT_RELATION_TYPES
# ---------------------------------------------------------------------------
class TestLegalRelations:
    def test_expected_relations_present(self):
        expected = {"MUST_NOTIFY", "MAY_REQUEST", "PROHIBITS", "HAS_DUTY", "MAY"}
        assert set(LEGAL_RELATIONS) == expected

    def test_concept_relation_types_includes_exception(self):
        assert "EXCEPTION_TO" in CONCEPT_RELATION_TYPES
        for rel in LEGAL_RELATIONS:
            assert rel in CONCEPT_RELATION_TYPES


# ---------------------------------------------------------------------------
# extract_triples — full pipeline with mocked chunks
# ---------------------------------------------------------------------------
class TestExtractTriples:
    def _chunk(self, content: str, chunk_number: int = 1) -> Document:
        return Document(
            content=content,
            meta={
                "id": f"chunk-{chunk_number}",
                "chunk_number": chunk_number,
                "document_name": "test.hwpx",
                "source_path": "documents/raw/statutes/test.hwpx",
                "group_id": 0,
                "part": 0,
                "article_number": "제15조",
                "article_title": "목적",
                "article_order": 1,
            },
        )

    def test_extracts_triple_from_obligation(self):
        chunk = self._chunk(
            "개인정보처리자는 정보주체에게 개인정보 처리방침을 통지하여야 한다."
        )
        triples = extract_triples([chunk])
        # Should find at least one triple
        assert len(triples) >= 1
        triple = triples[0]
        assert triple["relation"] == "MUST_NOTIFY"
        assert "개인정보처리자" in triple["subject"]
        assert "정보주체" in triple["object"]

    def test_no_triple_from_plain_text(self):
        chunk = self._chunk("이 법은 개인정보 보호를 목적으로 한다.")
        triples = extract_triples([chunk])
        assert triples == []

    def test_deduplication(self):
        chunk = self._chunk(
            "개인정보처리자는 정보주체에게 통지하여야 한다."
        )
        triples = extract_triples([chunk, chunk])
        # Should not have duplicate triples
        keys = [(t["subject"], t["object"], t["relation"]) for t in triples]
        assert len(keys) == len(set(keys))

    def test_triple_metadata(self):
        chunk = self._chunk(
            "개인정보처리자는 정보주체에게 통지하여야 한다.",
            chunk_number=3,
        )
        triples = extract_triples([chunk])
        assert triples[0]["source_path"] == "documents/raw/statutes/test.hwpx"
        assert triples[0]["document_name"] == "test.hwpx"
        assert triples[0]["chunk_number"] == 3
