import pytest

from backend.document import Document
from backend.reasoning import (
    classify_conclusion,
    contains_any,
    contains_permission,
    is_prohibition_fact,
    is_relevant_fact,
    matches_required_scenario,
    matched_terms,
    relevance_score,
    relevant_query_terms,
    sort_by_relevance,
    supporting_chunks,
    trace_items,
    trace_summary,
    unique_documents,
)
from backend.records import chunk_meta


# ---------------------------------------------------------------------------
# contains_any — simple substring check
# ---------------------------------------------------------------------------
class TestContainsAny:
    def test_match(self):
        assert contains_any("하여야 한다", ("하여야 한다", "해야 한다"))

    def test_no_match(self):
        assert not contains_any("일반 텍스트", ("하여야 한다", "해야 한다"))

    def test_empty_terms(self):
        assert not contains_any("text", ())


# ---------------------------------------------------------------------------
# contains_permission — permission without prohibition
# ---------------------------------------------------------------------------
class TestContainsPermission:
    def test_permission_present(self):
        assert contains_permission("수집할 수 있으며 이를 이용한다")

    def test_permission_blocked_by_prohibition(self):
        assert not contains_permission("해서는 안 된다")

    def test_no_permission(self):
        assert not contains_permission("일반 텍스트")


# ---------------------------------------------------------------------------
# is_prohibition_fact — detect prohibition in document
# ---------------------------------------------------------------------------
class TestIsProhibitionFact:
    def test_prohibition_relation(self):
        doc = Document("some content", {"graph_relation": "PROHIBITS"})
        assert is_prohibition_fact(doc)

    def test_prohibition_text(self):
        doc = Document("해서는 안 된다", {})
        assert is_prohibition_fact(doc)

    def test_permission_not_prohibition(self):
        doc = Document("할 수 있다", {"graph_relation": "MAY"})
        assert not is_prohibition_fact(doc)

    def test_may_in_content_not_prohibition(self):
        doc = Document("-MAY-> something", {})
        assert not is_prohibition_fact(doc)


# ---------------------------------------------------------------------------
# relevant_query_terms — extract meaningful terms from query
# ---------------------------------------------------------------------------
class TestRelevantQueryTerms:
    def test_extracts_korean_tokens(self):
        terms = relevant_query_terms("개인정보처리자의 통지 의무")
        assert "개인정보처리자" in terms

    def test_drops_ignored_terms(self):
        terms = relevant_query_terms("이것은 무엇인가요")
        assert "무엇" not in terms

    def test_drops_short_terms(self):
        terms = relevant_query_terms("이 법")
        assert all(len(t) >= 2 for t in terms)


# ---------------------------------------------------------------------------
# is_relevant_fact — check if document is relevant to query terms
# ---------------------------------------------------------------------------
class TestIsRelevantFact:
    def test_relevant_when_terms_match(self):
        doc = Document("개인정보처리자는 정보주체에게 통지하여야 한다", {})
        terms = ["개인정보처리자", "통지"]
        assert is_relevant_fact(doc, terms)

    def test_irrelevant_when_no_match(self):
        doc = Document("일반적인 내용입니다", {})
        terms = ["개인정보처리자", "통지"]
        assert not is_relevant_fact(doc, terms)

    def test_empty_terms_always_relevant(self):
        doc = Document("any content", {})
        assert is_relevant_fact(doc, [])

    def test_min_two_terms_required(self):
        doc = Document("개인정보처리자만 포함", {})
        terms = ["개인정보처리자", "통지", "의무"]
        assert not is_relevant_fact(doc, terms)


# ---------------------------------------------------------------------------
# matches_required_scenario — scenario-specific filters
# ---------------------------------------------------------------------------
class TestMatchesRequiredScenario:
    def test_pseudonym_info_requires_pseudonym(self):
        assert matches_required_scenario("가명정보 처리 기준", ["가명정보"])
        assert not matches_required_scenario("일반 정보 처리", ["가명정보"])

    def test_minor_requires_minor_or_guardian(self):
        assert matches_required_scenario("미성년자의 경우", ["미성년자"])
        assert matches_required_scenario("법정대리인의 동의", ["미성년자"])
        assert not matches_required_scenario("일반 사항", ["미성년자"])

    def test_contract_requires_contract_or_performance(self):
        assert matches_required_scenario("개인정보 계약 처리", ["개인정보", "계약"])
        assert matches_required_scenario("개인정보 이행 사항", ["개인정보", "계약"])
        assert not matches_required_scenario("개인정보 처리 방침", ["개인정보", "계약"])


# ---------------------------------------------------------------------------
# relevance_score — count matching terms
# ---------------------------------------------------------------------------
class TestRelevanceScore:
    def test_counts_matches(self):
        doc = Document("개인정보처리자는 정보주체에게 통지하여야 한다", {})
        assert relevance_score(doc, ["개인정보처리자", "통지"]) == 2

    def test_no_matches(self):
        doc = Document("일반 텍스트", {})
        assert relevance_score(doc, ["개인정보"]) == 0


# ---------------------------------------------------------------------------
# sort_by_relevance — sort by relevance score
# ---------------------------------------------------------------------------
class TestSortByRelevance:
    def test_sorted_by_score(self):
        docs = [
            Document("일반 텍스트", {}),
            Document("개인정보처리자는 통지하여야 한다", {}),
        ]
        result = sort_by_relevance(docs, ["개인정보처리자", "통지"])
        assert result[0].content == "개인정보처리자는 통지하여야 한다"


# ---------------------------------------------------------------------------
# unique_documents — dedup by (document_name, chunk_number)
# ---------------------------------------------------------------------------
class TestUniqueDocuments:
    def test_deduplicates(self):
        docs = [
            Document("a", {"document_name": "doc.pdf", "chunk_number": 1}),
            Document("b", {"document_name": "doc.pdf", "chunk_number": 2}),
            Document("c", {"document_name": "doc.pdf", "chunk_number": 1}),
        ]
        result = unique_documents(docs)
        assert len(result) == 2

    def test_no_duplicates(self):
        docs = [
            Document("a", {"document_name": "a.pdf", "chunk_number": 1}),
            Document("b", {"document_name": "b.pdf", "chunk_number": 2}),
        ]
        result = unique_documents(docs)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# matched_terms — extract query tokens, drop English stop words
# ---------------------------------------------------------------------------
class TestMatchedTerms:
    def test_extracts_tokens(self):
        terms = matched_terms("개인정보처리자의 통지 의무")
        assert any("개인정보처리자" in t for t in terms)

    def test_drops_english_stop(self):
        terms = matched_terms("what is the meaning of 개인정보")
        assert "what" not in terms

    def test_limits_to_twelve(self):
        terms = matched_terms("이 법은 개인정보를 보호하고 정보주체의 권리를 보장하며 데이터 처리의 투명성을 확보하기 위한 목적을 가진다")
        assert len(terms) <= 12


# ---------------------------------------------------------------------------
# supporting_chunks — extract chunk labels from evidence
# ---------------------------------------------------------------------------
class TestSupportingChunks:
    def test_extracts_labels(self):
        evidence = [
            Document("content 1", {"document_name": "법.pdf", "chunk_number": 1}),
            Document("content 2", {"document_name": "법.pdf", "chunk_number": 2}),
        ]
        result = supporting_chunks(evidence, limit=5)
        assert "법.pdf#1" in result
        assert "법.pdf#2" in result

    def test_skips_graph_facts(self):
        evidence = [
            Document("content", {"document_name": "법.pdf", "chunk_number": 1}),
            Document("그래프 사실:\npath\n근거 문장: sentence", {"document_name": "graph", "chunk_number": 0}),
        ]
        result = supporting_chunks(evidence, limit=5)
        assert len(result) == 1

    def test_respects_limit(self):
        evidence = [
            Document(f"content {i}", {"document_name": "법.pdf", "chunk_number": i})
            for i in range(10)
        ]
        result = supporting_chunks(evidence, limit=3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# trace_items — build trace dicts from documents
# ---------------------------------------------------------------------------
class TestTraceItems:
    def test_builds_items(self):
        docs = [
            Document("some content", {
                "document_name": "법.pdf",
                "chunk_number": 1,
                "graph_relation": "MUST_NOTIFY",
                "article_number": "제15조",
            }),
        ]
        result = trace_items(docs, limit=5)
        assert len(result) == 1
        assert result[0]["document"] == "법.pdf"
        assert result[0]["relation"] == "MUST_NOTIFY"
        assert result[0]["article"] == "제15조"

    def test_respects_limit(self):
        docs = [Document(f"content {i}", {}) for i in range(10)]
        result = trace_items(docs, limit=3)
        assert len(result) == 3

    def test_truncates_text(self):
        docs = [Document("a" * 500, {})]
        result = trace_items(docs, limit=1)
        assert len(result[0]["text"]) <= 240


# ---------------------------------------------------------------------------
# trace_summary — build summary steps
# ---------------------------------------------------------------------------
class TestTraceSummary:
    def test_article_lookup_summary(self):
        evidence = [
            Document("content", {"article_number": "제15조"}),
            Document("content", {"article_number": "제17조"}),
        ]
        conclusion = {"answer_type": "article_text", "reason": "test reason"}
        result = trace_summary("article_lookup", conclusion, evidence)
        assert any("제15조" in step for step in result)
        assert any("test reason" in step for step in result)

    def test_general_summary(self):
        conclusion = {"answer_type": "not_answered", "reason": "no evidence"}
        result = trace_summary("general", conclusion, [])
        assert any("general" in step for step in result)


# ---------------------------------------------------------------------------
# classify_conclusion — symbolic reasoning conclusion
# ---------------------------------------------------------------------------
class TestClassifyConclusion:
    def test_prohibition_returns_no(self):
        prohibition = [Document("금지 사항", {"graph_relation": "PROHIBITS"})]
        result = classify_conclusion("질문", "prohibition", [], prohibition, [], [], [], [])
        assert result["answer_type"] == "no"

    def test_permission_returns_yes(self):
        permission = [Document("할 수 있다", {"graph_relation": "MAY"})]
        result = classify_conclusion("질문", "permission", [], [], permission, [], [], [])
        assert result["answer_type"] == "yes"

    def test_obligation_returns_yes(self):
        obligation = [Document("하여야 한다", {"graph_relation": "HAS_DUTY"})]
        result = classify_conclusion("질문", "obligation", [], [], [], obligation, [], [])
        assert result["answer_type"] == "yes"

    def test_exception_returns_conditional(self):
        exception = [Document("다만 예외로 한다", {})]
        result = classify_conclusion("질문", "general", [], [], [], [], exception, [])
        assert result["answer_type"] == "conditional"

    def test_no_evidence_returns_not_answered(self):
        result = classify_conclusion("질문", "general", [], [], [], [], [], [])
        assert result["answer_type"] == "not_answered"

    def test_article_lookup_with_condition_returns_article_text(self):
        condition = [Document("경우에 해당한다", {})]
        result = classify_conclusion("질문", "article_lookup", [], [], [], [], [], condition)
        assert result["answer_type"] == "article_text"
