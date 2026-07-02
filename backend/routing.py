from dataclasses import dataclass, field
import re

from backend.constants import ARTICLE_PATTERN, QUERY_STOP_TERMS, TOKEN_PATTERN

QUOTED_PATTERN = re.compile(r"[\"'“”‘’](.*?)[\"'“”‘’]")


def has_standalone_phrase(text: str, phrase: str) -> bool:
    pattern = rf"(?<![가-힣A-Za-z0-9]){re.escape(phrase)}(?![가-힣A-Za-z0-9])"
    return bool(re.search(pattern, text))


@dataclass(frozen=True)
class QueryRoute:
    kind: str
    document_category: str | None = None
    article_numbers: list[str] = field(default_factory=list)
    search_terms: list[str] = field(default_factory=list)
    prefer_legal_units: bool = False
    include_graph_facts: bool = True
    expand_legal_units: bool = False


def classify_document_category(query: str) -> str | None:
    lowered = query.casefold()
    if any(term in lowered for term in ("판례", "판결", "사건번호", "선고")):
        return "precedent"
    if any(term in lowered for term in ("헌법재판소", "헌재", "헌법소원")):
        return "constitutional_decision"
    if any(term in lowered for term in ("계약서", "임대차", "임대인", "임차인", "보증금")):
        return "contract"
    if any(term in lowered for term in ("안내서", "가이드라인", "매뉴얼")):
        return "guidance"
    if any(term in lowered for term in ("시행령", "대통령령", "시행규칙")):
        return "regulation"
    return None


def classify_query(query: str, category: str | None = None) -> QueryRoute:
    lowered = query.casefold()
    article_numbers = ARTICLE_PATTERN.findall(query)
    quoted_terms = [
        term.strip()
        for term in QUOTED_PATTERN.findall(query)
        if term.strip()
    ]
    token_terms = [
        term
        for term in TOKEN_PATTERN.findall(query)
        if term.casefold() not in QUERY_STOP_TERMS
    ]
    search_terms = list(dict.fromkeys(quoted_terms + token_terms))[:10]
    doc_category = category or classify_document_category(query)

    if (
        "정의" in query
        or "란 무엇" in query
        or "이란 무엇" in query
        or has_standalone_phrase(query, "이란")
    ):
        return QueryRoute(
            kind="definition",
            article_numbers=article_numbers,
            search_terms=search_terms,
            document_category=doc_category,
            prefer_legal_units=True,
            expand_legal_units=True,
        )
    if article_numbers or "몇 조" in query or "어느 조" in query or "조에 규정" in query:
        return QueryRoute(
            kind="article_lookup",
            article_numbers=article_numbers,
            search_terms=search_terms,
            document_category=doc_category,
            prefer_legal_units=True,
            include_graph_facts=False,
            expand_legal_units=True,
        )
    if "인용" in query or "사건번호" in query or "판례_" in query or "lbox_" in lowered:
        return QueryRoute(
            kind="citation",
            article_numbers=article_numbers,
            search_terms=search_terms,
            document_category=doc_category,
            prefer_legal_units=True,
        )
    if "예외" in query or "제외" in query or "다만" in query:
        return QueryRoute(
            kind="exception",
            article_numbers=article_numbers,
            search_terms=search_terms,
            document_category=doc_category,
            prefer_legal_units=True,
            expand_legal_units=True,
        )
    if "조건" in query or "경우" in query or "요건" in query or "시점" in query:
        return QueryRoute(
            kind="condition",
            article_numbers=article_numbers,
            search_terms=search_terms,
            document_category=doc_category,
            prefer_legal_units=True,
            expand_legal_units=True,
        )
    if (
        "동의 없이" in query
        or "할 수 있" in query
        or "수 있나요" in query
        or "되나요" in query
        or "가능" in query
        or "허용" in query
    ):
        return QueryRoute(
            kind="condition",
            article_numbers=article_numbers,
            search_terms=search_terms,
            document_category=doc_category,
            prefer_legal_units=True,
            expand_legal_units=True,
        )
    if "금지" in query or "해서는 아니" in query:
        return QueryRoute(
            kind="prohibition",
            article_numbers=article_numbers,
            search_terms=search_terms,
            document_category=doc_category,
            prefer_legal_units=True,
            expand_legal_units=True,
        )
    if "의무" in query or "하여야" in query or "해야" in query:
        return QueryRoute(
            kind="obligation",
            article_numbers=article_numbers,
            search_terms=search_terms,
            document_category=doc_category,
            prefer_legal_units=True,
            expand_legal_units=True,
        )
    return QueryRoute(
        kind="general",
        article_numbers=article_numbers,
        search_terms=search_terms,
        expand_legal_units=bool(article_numbers),
    )
