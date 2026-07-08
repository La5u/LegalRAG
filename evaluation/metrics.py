import re

from backend.constants import CIRCLED_CLAUSE_MARKS, CITATION_PATTERN
from backend.document import Document
from backend.sources import normalize_source_text


CIRCLED_CLAUSES = {
    mark: f"제{index}항"
    for index, mark in enumerate(CIRCLED_CLAUSE_MARKS, start=1)
}


def normalize_metric_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    return text.replace("ㆍ", "·")


def normalize_citation(citation: str) -> str:
    normalized = normalize_metric_text(citation)
    for circled, clause in CIRCLED_CLAUSES.items():
        normalized = normalized.replace(circled, clause)
    return normalized


def normalize_source_name(source: str) -> str:
    return normalize_metric_text(normalize_source_text(source))


def is_article_only(citation: str) -> bool:
    return bool(re.fullmatch(r"제\d+조(?:의\d+)?", citation))


def citation_matches(gold: str, retrieved: str) -> bool:
    normalized_gold = normalize_citation(gold)
    normalized_retrieved = normalize_citation(retrieved)
    if normalized_gold == normalized_retrieved:
        return True
    source_gold = normalize_source_name(gold)
    source_retrieved = normalize_source_name(retrieved)
    if source_gold and source_retrieved and (
        source_gold == source_retrieved
        or source_gold in source_retrieved
        or source_retrieved in source_gold
    ):
        return True
    return (
        is_article_only(normalized_gold)
        and normalized_retrieved.startswith(f"{normalized_gold}제")
    )


def context_precision(retrieved_chunks: list[Document], relevant_terms: list[str]) -> float:
    text_chunks = [
        chunk
        for chunk in retrieved_chunks
        if not chunk.content.startswith(("Graph fact:", "그래프 사실:"))
    ]
    if not text_chunks or not relevant_terms:
        return 0.0

    normalized_terms = [
        normalize_metric_text(term)
        for term in relevant_terms
        if term.strip()
    ]
    if not normalized_terms:
        return 0.0

    def matches(chunk_content: str, term: str) -> bool:
        normalized_content = normalize_metric_text(chunk_content)
        if term in normalized_content:
            return True
        return any(
            citation_matches(term, citation)
            for citation in CITATION_PATTERN.findall(chunk_content)
        )

    relevant_count = sum(
        1
        for chunk in text_chunks
        if any(
            matches(chunk.content, term)
            or citation_matches(term, chunk.meta.get("specific_citation", ""))
            or citation_matches(term, chunk.meta.get("article_number", ""))
            for term in normalized_terms
        )
    )
    return relevant_count / len(text_chunks)
