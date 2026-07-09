from dataclasses import dataclass, field
import re

from backend.constants import ARTICLE_PATTERN, QUERY_STOP_TERMS
from backend.text import get_kiwi, query_terms

QUOTED_PATTERN = re.compile(r"[\"'“”‘’](.*?)[\"'“”‘’]")


def has_standalone_phrase(text: str, phrase: str) -> bool:
    """Check if phrase appears as a standalone word using Kiwi morphological analysis.

    A phrase is considered embedded (not standalone) when it spans across
    multiple content-word tokens. Uses Kiwi token boundaries to verify
    the phrase doesn't sit inside a larger compound."""
    if not phrase or phrase not in text:
        return False

    analyzed = get_kiwi().analyze(text)
    if not analyzed:
        return False
    tokens = analyzed[0][0]

    content_tags = {"NNG", "NNP", "NNB", "NR", "VV", "VA", "VX", "VCN", "SL", "SN"}

    # Find all occurrences of phrase in the original text
    start = 0
    while True:
        idx = text.find(phrase, start)
        if idx == -1:
            break
        end = idx + len(phrase)

        # Check each token that overlaps with the phrase span
        overlapping_content_tokens = []
        for token in tokens:
            # Token overlaps with phrase if it starts before phrase ends
            # and ends after phrase starts
            token_end = token.start + token.len
            if token.start < end and token_end > idx:
                if token.tag in content_tags:
                    overlapping_content_tokens.append(token)

        # If phrase spans 2+ content tokens, it's embedded in a compound
        if len(overlapping_content_tokens) >= 2:
            start = end
            continue

        return True

    return False


@dataclass(frozen=True)
class QueryRoute:
    kind: str
    document_category: str | None = None
    article_numbers: list[str] = field(default_factory=list)
    search_terms: list[str] = field(default_factory=list)
    include_graph_facts: bool = True


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
    search_terms = query_terms(
        query,
        initial_terms=quoted_terms,
        ignored_terms=QUERY_STOP_TERMS,
        max_terms=10,
    )
    doc_category = category or classify_document_category(query)

    if (
        "정의" in query
        or "란 무엇" in query
        or "이란 무엇" in query
        or has_standalone_phrase(query, "이란")
        or re.search(r"[가-힣]란\s*$", query)
    ):
        return QueryRoute(
            kind="definition",
            article_numbers=article_numbers,
            search_terms=search_terms,
            document_category=doc_category,
        )
    if article_numbers or "몇 조" in query or "어느 조" in query or "조에 규정" in query:
        return QueryRoute(
            kind="article_lookup",
            article_numbers=article_numbers,
            search_terms=search_terms,
            document_category=doc_category,
            include_graph_facts=False,
        )
    if "인용" in query or "사건번호" in query or "판례_" in query or "lbox_" in lowered:
        return QueryRoute(
            kind="citation",
            article_numbers=article_numbers,
            search_terms=search_terms,
            document_category=doc_category,
        )
    if "예외" in query or "제외" in query or "다만" in query:
        return QueryRoute(
            kind="exception",
            article_numbers=article_numbers,
            search_terms=search_terms,
            document_category=doc_category,
        )
    if "금지" in query or "해서는 아니" in query:
        return QueryRoute(
            kind="prohibition",
            article_numbers=article_numbers,
            search_terms=search_terms,
            document_category=doc_category,
        )
    if "조건" in query or "경우" in query or "요건" in query or "시점" in query:
        return QueryRoute(
            kind="condition",
            article_numbers=article_numbers,
            search_terms=search_terms,
            document_category=doc_category,
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
        )
    if "의무" in query or "하여야" in query or "해야" in query:
        return QueryRoute(
            kind="obligation",
            article_numbers=article_numbers,
            search_terms=search_terms,
            document_category=doc_category,
        )
    return QueryRoute(
        kind="general",
        article_numbers=article_numbers,
        search_terms=search_terms,
    )


def route_reason(query_type: str | None) -> str:
    reasons = {
        "article_lookup": "The question asks for a specific article or article number.",
        "condition": "The question asks whether something is allowed or depends on conditions.",
        "definition": "The question asks what a legal term means.",
        "exception": "The question asks about an exception or exclusion.",
        "general": "No specific legal route matched, so general retrieval was used.",
        "obligation": "The question asks about a duty or required action.",
        "prohibition": "The question asks about a ban or prohibited action.",
    }
    return reasons.get(query_type, "The route could not be classified more specifically.")
