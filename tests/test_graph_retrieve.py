import pytest

from backend.document import Document
from backend.graph_retrieve import (
    GraphFact,
    article_terms,
    case_terms_from_query,
    preferred_relations,
    query_terms,
    wants_case_fact,
    wants_definition,
)
from backend.records import chunk_meta


# ---------------------------------------------------------------------------
# query_terms — extract meaningful terms, drop stop words
# ---------------------------------------------------------------------------
class TestQueryTerms:
    def test_extracts_korean_tokens(self):
        terms = query_terms("개인정보처리자의 통지 의무")
        assert any("개인정보처리자" in t for t in terms)
        assert "통지" in terms

    def test_drops_stop_terms(self):
        terms = query_terms("개인정보란 무엇인가요")
        assert "무엇" not in terms

    def test_limits_to_eight(self):
        terms = query_terms("이 법은 개인정보를 보호하고 정보주체의 권리를 보장하기 위한 목적을 가진다")
        assert len(terms) <= 8

    def test_empty_query(self):
        assert query_terms("") == []


# ---------------------------------------------------------------------------
# wants_definition — detect definition queries
# ---------------------------------------------------------------------------
class TestWantsDefinition:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("개인정보란 무엇인가요", True),
            ("개인정보의 정의를 알려주세요", True),
            ("정의", True),
            ("define personal data", True),
            ("정의란 무엇인가", True),
            ("통지 의무", False),
            ("제15조 내용", False),
        ],
    )
    def test_detects_definition(self, query, expected):
        assert wants_definition(query) == expected


# ---------------------------------------------------------------------------
# wants_case_fact — detect precedent/case queries
# ---------------------------------------------------------------------------
class TestWantsCaseFact:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("판례에서 개인정보 보호는 어떻게 되나요", True),
            ("대법원 판결 요지", True),
            ("헌재 결정례", True),
            ("2024다12345 사건", True),
            ("lbox_precedent_12345", True),
            ("개인정보 처리 방침", False),
            ("통지 의무", False),
        ],
    )
    def test_detects_case(self, query, expected):
        assert wants_case_fact(query) == expected


# ---------------------------------------------------------------------------
# case_terms_from_query — extract case numbers
# ---------------------------------------------------------------------------
class TestCaseTermsFromQuery:
    def test_extracts_case_number(self):
        terms = case_terms_from_query("2024다12345 사건의 판결 요지")
        assert "2024다12345" in terms

    def test_extracts_lbox_id(self):
        terms = case_terms_from_query("lbox_precedent_12345의 판시사항")
        assert "lbox_precedent_12345" in terms

    def test_no_case_number(self):
        assert case_terms_from_query("개인정보 처리 방침") == []


# ---------------------------------------------------------------------------
# article_terms — extract article references
# ---------------------------------------------------------------------------
class TestArticleTerms:
    def test_extracts_article(self):
        terms = article_terms("제15조와 제17조의 차이점")
        assert "제15조" in terms
        assert "제17조" in terms

    def test_extracts_article_with_suffix(self):
        terms = article_terms("제15조의2 적용 범위")
        assert "제15조의2" in terms

    def test_no_articles(self):
        assert article_terms("개인정보 처리 방침") == []


# ---------------------------------------------------------------------------
# preferred_relations — infer relation types from query
# ---------------------------------------------------------------------------
class TestPreferredRelations:
    @pytest.mark.parametrize(
        "query, expected_relation",
        [
            ("통지 의무는 무엇인가요", "MUST_NOTIFY"),
            ("요구할 수 있는 권리", "MAY_REQUEST"),
            ("금지 사항", "PROHIBITS"),
            ("의무 사항", "HAS_DUTY"),
            ("예외 규정", "EXCEPTION_TO"),
            ("할 수 있는 경우", "MAY"),
        ],
    )
    def test_detects_relation(self, query, expected_relation):
        relations = preferred_relations(query)
        assert expected_relation in relations

    def test_no_relations(self):
        assert preferred_relations("일반적인 질문") == []


# ---------------------------------------------------------------------------
# GraphFact dataclass
# ---------------------------------------------------------------------------
class TestGraphFact:
    def test_content_format(self):
        fact = GraphFact(
            id="test-id",
            document="test.pdf",
            source_path="documents/raw/statutes/test.pdf",
            chunk=1,
            relation="MUST_NOTIFY",
            path="개인정보처리자 -MUST_NOTIFY-> 정보주체",
            sentence="개인정보처리자는 정보주체에게 통지하여야 한다.",
            score=0.95,
        )
        assert "그래프 사실:" in fact.content
        assert "MUST_NOTIFY" in fact.content
        assert "개인정보처리자는 정보주체에게 통지하여야 한다." in fact.content

    def test_to_document(self):
        fact = GraphFact(
            id="test-id",
            document="test.pdf",
            source_path="documents/raw/statutes/test.pdf",
            chunk=1,
            relation="MUST_NOTIFY",
            path="A -MUST_NOTIFY-> B",
            sentence="test sentence",
            score=0.95,
            article="제15조",
        )
        doc = fact.to_document()
        assert doc.content == fact.content
        assert doc.meta["article_number"] == "제15조"
        assert doc.meta["graph_relation"] == "MUST_NOTIFY"
        assert doc.score == 0.95

    def test_to_response(self):
        fact = GraphFact(
            id="test-id",
            document="test.pdf",
            source_path="documents/raw/statutes/test.pdf",
            chunk=1,
            relation="MUST_NOTIFY",
            path="A -> B",
            sentence="test",
            score=0.95,
        )
        resp = fact.to_response()
        assert resp["relation"] == "MUST_NOTIFY"
        assert resp["path"] == "A -> B"
        assert resp["score"] == 0.95
