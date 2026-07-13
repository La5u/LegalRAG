import pytest

from backend.routing import (
    QueryRoute,
    classify_document_category,
    classify_query,
    has_standalone_phrase,
    route_reason,
)


# ---------------------------------------------------------------------------
# classify_query — all 7 query kinds
# ---------------------------------------------------------------------------
class TestClassifyQuery:
    @pytest.mark.parametrize(
        "query, expected_kind",
        [
            # definition
            ("개인정보처리자란 무엇인가요", "definition"),
            ("개인정보의 정의는?", "definition"),
            ("개인정보란", "definition"),
            # article_lookup
            ("제15조에 규정된 내용은?", "article_lookup"),
            ("몇 조에 해당하나요", "article_lookup"),
            ("어느 조에 규정되어 있나요", "article_lookup"),
            # citation
            ("판례_2023마1234", "citation"),
            ("인용된 판례", "citation"),
            # exception
            ("다만 예외는?", "exception"),
            ("제외되는 경우", "exception"),
            # condition
            ("동의 없이 할 수 있나요", "condition"),
            ("가능한가요", "condition"),
            ("허용되는 경우", "condition"),
            ("이 경우에 해당", "condition"),
            # prohibition
            ("금지하는 행위", "prohibition"),
            ("해서는 아니 되는 경우", "prohibition"),
            # obligation
            ("통지하여야 한다", "obligation"),
            ("해야 하는 의무", "obligation"),
            # general fallback
            ("개인정보 보호법에 대하여", "general"),
        ],
    )
    def test_query_classification(self, query, expected_kind):
        route = classify_query(query)
        assert route.kind == expected_kind

    def test_article_numbers_extracted(self):
        route = classify_query("제15조의2에 대한 설명")
        assert "제15조의2" in route.article_numbers

    def test_search_terms_from_quoted(self):
        route = classify_query('"개인정보처리자"의 의무')
        assert "개인정보처리자" in route.search_terms

    def test_empty_query(self):
        route = classify_query("")
        assert route.kind == "general"

    def test_mixed_language_query(self):
        route = classify_query("What is the 제15조?")
        assert route.kind == "article_lookup"
        assert "제15조" in route.article_numbers

    def test_combined_article_and_exception_intent(self):
        route = classify_query("민법 제5조의 예외와 적용 조건은?")
        assert route.kind == "exception"
        assert route.tasks == ["article_lookup", "exception", "condition"]
        assert route.article_numbers == ["제5조"]

    def test_parent_act_cross_reference_is_not_target_article(self):
        route = classify_query("형사소송규칙에서 법 제6조에 따른 신청 방법은?")

        assert route.article_numbers == []


# ---------------------------------------------------------------------------
# classify_document_category
# ---------------------------------------------------------------------------
class TestClassifyDocumentCategory:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("판례를 검색해주세요", "precedent"),
            ("2023마1234 판결", "precedent"),
            ("헌법재판소 결정", "constitutional_decision"),
            ("헌재 결정", "constitutional_decision"),
            ("임대차 계약서", "contract"),
            ("보증금 관련", "contract"),
            ("안내서에 나와있는", "guidance"),
            ("가이드라인 확인", "guidance"),
            ("시행령 제1조", "regulation"),
            ("대통령령", "regulation"),
            # No category
            ("개인정보 보호법", None),
        ],
    )
    def test_document_category(self, query, expected):
        assert classify_document_category(query) == expected


# ---------------------------------------------------------------------------
# route_reason
# ---------------------------------------------------------------------------
class TestRouteReason:
    @pytest.mark.parametrize(
        "query_type, contains_text",
        [
            ("article_lookup", "specific article"),
            ("definition", "legal term means"),
            ("general", "general retrieval"),
            ("obligation", "duty"),
            ("prohibition", "ban"),
            ("exception", "exception"),
            ("condition", "allowed"),
        ],
    )
    def test_route_reason(self, query_type, contains_text):
        reason = route_reason(query_type)
        assert contains_text.lower() in reason.lower()

    def test_unknown_type_returns_default(self):
        reason = route_reason("unknown_type")
        assert "could not be classified" in reason


# ---------------------------------------------------------------------------
# has_standalone_phrase — Korean word boundary detection via Kiwi
# ---------------------------------------------------------------------------
class TestHasStandalonePhrase:
    @pytest.mark.parametrize(
        "text, phrase, expected",
        [
            # Actual use case: "이란" particle marking a definition
            ("개인정보처리자이란 무엇인가요", "이란", True),
            ("이것이 바로 개인정보이란 것이다", "이란", True),
            # "이란" in compound — Kiwi splits "개인정보이란" as NNP+VCP+ETM,
            # "이란" spans only VCP+ETM, so no two content tokens → standalone
            ("개인정보이란", "이란", True),

            # Compound word: phrase spans 2+ content tokens → embedded
            ("개인정보보호법", "정보보호", False),
            ("개인정보처리자", "정보처리", False),
            # Kiwi splits "개인정보" as 개인(NNG)+정보(NNG) → spans 2 content tokens
            ("개인정보는 중요한 자산이다", "개인정보", False),
            ("이것은 개인정보", "개인정보", False),
            ("개인정보를 보호한다", "개인정보", False),

            # Phrase is a single content token → standalone
            ("이것은 자산이다", "자산", True),
            ("이 법은 개인정보를 보호한다", "이", True),

            # Empty / nonexistent
            ("", "테스트", False),
            (" 아무 텍스트 ", "존재하지않는단어", False),
        ],
    )
    def test_standalone_phrase(self, text, phrase, expected):
        assert has_standalone_phrase(text, phrase) == expected

    def test_returns_false_for_empty_text(self):
        assert has_standalone_phrase("", "이란") is False

    def test_returns_false_for_empty_phrase(self):
        assert has_standalone_phrase(" 아무 텍스트 ", "") is False
        assert has_standalone_phrase("text", "") is False


# ---------------------------------------------------------------------------
# QueryRoute dataclass
# ---------------------------------------------------------------------------
class TestQueryRoute:
    def test_defaults(self):
        route = QueryRoute(kind="general")
        assert route.document_category is None
        assert route.article_numbers == []
        assert route.search_terms == []

    def test_frozen(self):
        route = QueryRoute(kind="definition")
        with pytest.raises(AttributeError):
            route.kind = "general"
